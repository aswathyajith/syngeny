from inference.models import MODEL_LABEL_MAP, DATASETS, OR_MODEL_LABEL_MAP
import pandas as pd       
import argparse
import os
import json
import random
import numpy as np
from transformers import set_seed
from tqdm import tqdm 
from data.utils import load_data, append_data, save_data
from inference.seeding import derive_seed
from inference.clients.openrouter import OpenRouterClient
from inference.clients.ollama import OllamaClient
from inference.clients.vllm import VLLMClient
from inference.clients.local_transformers import LocalTransformersClient


MODELS = {
    "huggingface": MODEL_LABEL_MAP, 
    "ollama": MODEL_LABEL_MAP,
    "vllm": MODEL_LABEL_MAP,
    "openrouter": OR_MODEL_LABEL_MAP
}

def has_answer_tags(response):
    """Whether the response has an <answer> tag followed by a closing </answer> tag."""

    if not isinstance(response, str):
        return False
    start = response.find("<answer>")
    return start != -1 and response.find("</answer>", start) != -1

def completed_words(output_path):
    """Returns the words in an existing results file that finished generating, backfilling finish_reason.

    Rows with finish_reason "length" need a rerun. Rows from older runs have no finish_reason:
    they count as finished ("stop") if the answer tags are matched, and need a rerun otherwise.
    """

    if not os.path.exists(output_path):
        return []

    df = load_data(output_path)
    if "finish_reason" not in df.columns:
        df["finish_reason"] = None

    backfill = df["finish_reason"].isna() & df["model_response"].apply(has_answer_tags)
    if backfill.any() or "finish_reason" not in load_data(output_path).columns:
        df.loc[backfill, "finish_reason"] = "stop"
        save_data(df, output_path)  # persist the backfilled finish_reason

    rerun = df["finish_reason"].isna() | (df["finish_reason"] == "length")
    if rerun.any():
        print(f"Rerunning {rerun.sum()} of {len(df)} words with truncated or unverifiable responses in {output_path}")
    return df.loc[~rerun, "word"].tolist()

def query_once(client, provider, model, messages, seed, query_kwargs):
    """Queries the client once and returns (response text, response in the API's JSON shape)."""

    if provider == "huggingface":
        # The local client returns text, so wrap it in the API response shape.
        model_response = client.query_text(messages=messages, seed=seed)
        response_json = {
            "model": model,
            "choices": [{
                "message": {"role": "assistant", "content": model_response},
                "finish_reason": client.finish_reason,
            }]
        }
    else:
        response = client.query(messages=messages, seed=seed, **query_kwargs)
        if hasattr(response, "model_dump_json") and callable(getattr(response, "model_dump_json")):
            response_json = json.loads(response.model_dump_json() if response else '{}')
        elif hasattr(response, "__dict__") or isinstance(response, dict):
            response_json = dict(response) if response else {}
        model_response = (response_json.get('choices') or [{}])[0].get('message', {}).get('content', '') if response_json else ''
    return model_response, response_json

def query_model_on_dataset(
        df, 
        model="openai/gpt-4o-mini", 
        provider="openrouter",
        output_path = "results/openai/gpt-4o-mini/sample-10/model_outputs_df.jsonl",
        responses_dir = "responses/openai/gpt-4o-mini/sample-10", 
        max_tokens = 2048,
        max_tokens_limit = 8192,
        temperature = 0.0,
        seed=42,
        max_consecutive_failures = 5,
        stop = None,
        base_url = None
    ):
    
    processed_words = completed_words(output_path)

    df_new = df[~df['word'].isin(processed_words)]  # Filter out already processed words
    if len(df_new) == 0: 
        print(f"No words remaining to test {model} on.")
        return
    
    if provider == "openrouter":
        client = OpenRouterClient(model=model, max_tokens=max_tokens, temperature=temperature, seed=seed)
    elif provider == "ollama":
        client = OllamaClient(model=model, max_tokens=max_tokens, temperature=temperature, seed=seed)
    elif provider == "vllm":
        client = VLLMClient(model=model, base_url=base_url, max_tokens=max_tokens, temperature=temperature, seed=seed)
    elif provider == "huggingface":
        client = LocalTransformersClient(model=model, max_tokens=max_tokens, temperature=temperature, stop=stop, seed=seed)

    # The local client takes stop strings at construction; the API takes them per request.
    query_kwargs = {"stop": stop} if stop and provider != "huggingface" else {}

    os.makedirs(responses_dir, exist_ok=True)
    
    print(f"Total words remaining to process: {len(df_new)}")
    print(f"Responses will be saved to: {responses_dir}")
    model_outputs_df = []
    consecutive_failures = 0

    for row in tqdm(df_new.itertuples(), total=len(df_new)):
        word = row.word
        messages = row.messages
        print(f"Processing word: {word}")
        word_seed = derive_seed(seed, temperature, word)  # independent noise per (seed, temperature, word)

        try:
            # Retry a truncated response with a doubled max_tokens, up to max_tokens_limit.
            attempt_tokens = max_tokens
            while True:
                client.max_tokens = attempt_tokens
                model_response, response_json = query_once(client, provider, model, messages, word_seed, query_kwargs)
                finish_reason = (response_json.get("choices") or [{}])[0].get("finish_reason")
                if finish_reason != "length" or attempt_tokens >= max_tokens_limit:
                    break
                print(f"'{word}' truncated at max_tokens={attempt_tokens}; retrying with {min(attempt_tokens * 2, max_tokens_limit)}")
                attempt_tokens = min(attempt_tokens * 2, max_tokens_limit)
            if finish_reason == "length":
                print(f"WARNING: '{word}' still truncated at max_tokens={attempt_tokens}")
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"Error querying model on '{word}': {e}")
            if consecutive_failures >= max_consecutive_failures:
                print(f"Stopping {model}: {consecutive_failures} consecutive failures.")
                break
            continue

        response_fname = os.path.join(responses_dir, f"{word}.jsonl")

        try:
            with open(response_fname, "a", encoding="utf-8") as file:
                file.write(json.dumps(response_json) + "\n")
                
        except Exception as e:
            print(f"Error saving response: {e}")

        model_outputs_df.append({
            "model": model,
            "word": word,
            "model_response": model_response,
            "finish_reason": finish_reason
        })
        if len(model_outputs_df) % 5 == 0:
            print(f"Processed {len(model_outputs_df)} words. Saving intermediate results.")
            append_data(pd.DataFrame(model_outputs_df), output_path)
            model_outputs_df = []  # Clear the list after saving

    if len(model_outputs_df) > 0:  # Save any remaining data
        append_data(pd.DataFrame(model_outputs_df), output_path)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Run dataset of prompts through a model")
    parser.add_argument("--provider", default="openrouter", help="Model id on provider", choices=["openrouter", "huggingface", "ollama", "vllm"])
    parser.add_argument("--base-url", default=None, help="endpoint for the vllm provider (default: $VLLM_BASE_URL or http://localhost:8000/v1)")
    parser.add_argument("--model-ids", default=None, help="Model ids on provider", nargs='+')
    parser.add_argument("--prompt-path-fmt", default="data/prompts/guess_meaning/p1/{dataset_name}.jsonl")
    parser.add_argument("--responses-dir-fmt", default="responses/{model}/{dataset_name}/tmp={temperature}/seed={seed}")
    parser.add_argument("--results-path-fmt", default="results/{dataset_name}/{model}/tmp={temperature}/results.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens-limit", type=int, default=8192, help="ceiling when retrying truncated responses with a doubled max_tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop", default=None, nargs="+", help="strings that end generation")
    args = parser.parse_args()

    provider = args.provider
    temperature = args.temperature
    max_tokens = args.max_tokens
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)  # python, numpy and torch (incl. CUDA)
    if args.model_ids: 
        models = args.model_ids
    else:
        models = [model_name for model_family, model_maps in MODELS[provider].items() for model_name in list(model_maps.keys())]

    for dataset_name in DATASETS.keys():
        df = load_data(args.prompt_path_fmt.format(dataset_name=dataset_name))
        for model in models:
            query_model_on_dataset(
                df, 
                model=model, 
                provider=provider,
                output_path = args.results_path_fmt.format(model=model, dataset_name=dataset_name, temperature=temperature, seed=seed),
                responses_dir = args.responses_dir_fmt.format(model=model, dataset_name=dataset_name, temperature=temperature, seed=seed), 
                max_tokens=max_tokens,
                max_tokens_limit=args.max_tokens_limit,
                temperature=temperature,
                seed=seed,
                stop=args.stop,
                base_url=args.base_url
            )
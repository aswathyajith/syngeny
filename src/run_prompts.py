from inference.models import HF_MODEL_LABEL_MAP, DATASETS, OR_MODEL_LABEL_MAP
import pandas as pd       
import argparse
import os
import json
import random
import numpy as np
from transformers import set_seed
from tqdm import tqdm 
from data.utils import load_data, append_data
from inference.clients.openrouter import OpenRouterClient
from inference.clients.huggingface import HuggingFaceClient
from inference.clients.local_transformers import LocalTransformersClient


MODELS = {
    "huggingface": HF_MODEL_LABEL_MAP, 
    "huggingface-api": HF_MODEL_LABEL_MAP,
    "openrouter": OR_MODEL_LABEL_MAP
}

def query_model_on_dataset(
        df, 
        model="openai/gpt-4o-mini", 
        provider="openrouter",
        output_path = "results/openai/gpt-4o-mini/sample-10/model_outputs_df.jsonl",
        responses_dir = "responses/openai/gpt-4o-mini/sample-10", 
        max_tokens = 2048,
        temperature = 0.0,
        seed=42,
        max_consecutive_failures = 5,
        stop = None
    ):
    
    processed_words = load_data(output_path).word.tolist() if os.path.exists(output_path) else []

    df_new = df[~df['word'].isin(processed_words)]  # Filter out already processed words
    if len(df_new) == 0: 
        print(f"No words remaining to test {model} on.")
        return
    
    if provider == "openrouter":
        client = OpenRouterClient(model=model, max_tokens=max_tokens, temperature=temperature, seed=seed)
    elif provider == "huggingface-api":
        client = HuggingFaceClient(model=model, max_tokens=max_tokens, temperature=temperature, seed=seed)
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

        try:
            if provider == "huggingface":
                # The local client returns text, so wrap it in the API response shape.
                model_response = client.query_text(messages=messages)
                response_json = {
                    "model": model,
                    "choices": [{"message": {"role": "assistant", "content": model_response}}]
                }
            else:
                response = client.query(messages=messages, **query_kwargs)
                response_json = json.loads(response.model_dump_json() if response else '{}')
                model_response = (response_json.get('choices') or [{}])[0].get('message', {}).get('content', '') if response_json else ''
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
            "model_response": model_response
        })
        if len(model_outputs_df) % 5 == 0:
            print(f"Processed {len(model_outputs_df)} words. Saving intermediate results.")
            append_data(pd.DataFrame(model_outputs_df), output_path)
            model_outputs_df = []  # Clear the list after saving

    if len(model_outputs_df) > 0:  # Save any remaining data
        append_data(pd.DataFrame(model_outputs_df), output_path)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Run dataset of prompts through a model")
    parser.add_argument("--provider", default="openrouter", help="Model id on provider", choices=["openrouter", "huggingface", "huggingface-api"])
    parser.add_argument("--model-ids", default=None, help="Model ids on provider", nargs='+')
    parser.add_argument("--prompt-path-fmt", default="data/prompts/guess_meaning/p1/{dataset_name}.jsonl")
    parser.add_argument("--responses-dir-fmt", default="responses/{model}/{dataset_name}/tmp={temperature}/seed={seed}")
    parser.add_argument("--results-path-fmt", default="results/{dataset_name}/{model}/tmp={temperature}/results.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
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
        models = list(MODELS[provider].keys())

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
                temperature=temperature,
                seed=seed,
                stop=args.stop
            )
import argparse
import os
from huggingface_hub import InferenceClient
import requests
BASE_URL = "http://localhost:11434/v1"

OLLAMA_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5:0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5:1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5:3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5:7b",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5:72b",
    "google/gemma-3-1b-it": "gemma3:1b-it-qat", 
    "google/gemma-3-4b-it": "gemma3:4b-it-qat", 
    "google/gemma-3-12b-it": "gemma3:12b-it-qat",
    "meta-llama/Llama-3.2-1B-Instruct": "llama3.2:1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2:3b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b"
}
class OllamaClient:
    """Queries a model served through Hugging Face's OpenAI-compatible router API."""

    def __init__(
            self,
            model: str,
            api_key: str = "ollama",
            temperature: float = 0.0,
            max_tokens: int = 256,
            max_retries: int = 3,
            base_url: str = BASE_URL,
            seed: int | None = None,
        ):

        self.model = OLLAMA_MODELS[model]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = InferenceClient(base_url=base_url, api_key=api_key)

        # Ollama only loads a model on its first request, so send a dummy one before checking
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=1,
            )
        except Exception as e:
            print(f"Warm-up request failed: {e}")

        # Check if model is being served with ollama
        response = requests.get("http://localhost:11434/api/ps")
        response.raise_for_status()
        models = [hosted_model["model"] for hosted_model in response.json().get("models", [])] # Parse the active models
        assert self.model in models, f"Model {self.model} not running"

    def query(
            self,
            prompt: str = "Hello!",
            system_prompt: str | None = None,
            messages: list[dict] | None = None,
            seed: int | None = None,
            **kwargs,
        ):
        """Sends a single prompt and returns the completion object; `seed` overrides the client's seed for this call."""

        if messages is None:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed if seed is None else seed,
            **kwargs,
        )

    def query_text(self, prompt: str = "Hello!", system_prompt: str | None = None, **kwargs):
        """Sends a single prompt and returns the response text only."""

        response = self.query(prompt, system_prompt=system_prompt, **kwargs)
        return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Query a model on Hugging Face.")
    parser.add_argument(
        "--model",
        # default="meta-llama/Llama-3.1-8B-Instruct",
        default="EleutherAI/pythia-1b",
        help="Hugging Face model id (optionally ':<provider>', e.g. ':together')",
    )
    parser.add_argument(
        "--user-prompt",
        default="Guess the most likely meaning of the pseudoword 'trendipitious'.",
        help="user prompt to send",
    )
    parser.add_argument("--system-prompt", default=None, help="optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="OpenAI-compatible endpoint (e.g. a dedicated Inference Endpoint)",
    )
    args = parser.parse_args()

    client = OllamaClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )

    print(client.query_text(args.user_prompt, system_prompt=args.system_prompt))

if __name__ == "__main__":
    main()

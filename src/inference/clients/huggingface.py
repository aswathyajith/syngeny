import argparse
import os

from openai import OpenAI

BASE_URL = "https://router.huggingface.co/v1"


class HuggingFaceClient:
    """Queries a model served through Hugging Face's OpenAI-compatible router API."""

    def __init__(
            self,
            model: str,
            api_key: str | None = None,
            temperature: float = 0.0,
            max_tokens: int = 256,
            max_retries: int = 3,
            base_url: str = BASE_URL,
            seed: int | None = None,
        ):

        api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("No API key given (set HF_TOKEN or pass api_key=...)")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)

    def query(
            self,
            prompt: str = "Hello!",
            system_prompt: str | None = None,
            messages: list[dict] | None = None,
            **kwargs,
        ):
        """Sends a single prompt and returns the completion object."""

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
            seed=self.seed,
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

    client = HuggingFaceClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )

    print(client.query_text(args.user_prompt, system_prompt=args.system_prompt))

if __name__ == "__main__":
    main()

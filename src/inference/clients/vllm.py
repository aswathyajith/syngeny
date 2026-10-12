import argparse
import os

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"


class VLLMClient:
    """Queries a model served by vLLM's OpenAI-compatible server (e.g. on a remote machine through an SSH tunnel)."""

    def __init__(
            self,
            model: str,
            base_url: str | None = None,
            api_key: str | None = None,
            temperature: float = 0.0,
            max_tokens: int = 256,
            max_retries: int = 3,
            seed: int | None = None,
        ):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = OpenAI(
            base_url=base_url or os.environ.get("VLLM_BASE_URL", BASE_URL),
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            max_retries=max_retries,
        )

        # Check that the server is up and is serving this model (vLLM serves one model per server)
        served = [m.id for m in self.client.models.list().data]
        assert self.model in served, f"Model {self.model} not served at {self.client.base_url}; served: {served}"

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
    parser = argparse.ArgumentParser(description="Query a model served with vLLM.")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="model id the vLLM server is serving")
    parser.add_argument(
        "--user-prompt",
        default="Guess the most likely meaning of the pseudoword 'trendipitious'.",
        help="user prompt to send",
    )
    parser.add_argument("--system-prompt", default=None, help="optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-url", default=None, help="vLLM endpoint (default: $VLLM_BASE_URL or http://localhost:8000/v1)")
    args = parser.parse_args()

    client = VLLMClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(client.query_text(args.user_prompt, system_prompt=args.system_prompt))

if __name__ == "__main__":
    main()

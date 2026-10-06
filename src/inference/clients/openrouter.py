import argparse
import os

from openai import OpenAI

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Queries a model served through OpenRouter's OpenAI-compatible API."""

    def __init__(
            self,
            model: str,
            api_key: str | None = None,
            temperature: float = 0.0,
            max_tokens: int = 256,
            max_retries: int = 3,
        ):

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No API key given (set OPENROUTER_API_KEY or pass api_key=...)")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=BASE_URL, api_key=api_key, max_retries=max_retries)

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
            **kwargs,
        )

    def query_text(self, prompt: str = "Hello!", system_prompt: str | None = None, **kwargs):
        """Sends a single prompt and returns the response text only."""

        response = self.query(prompt, system_prompt=system_prompt, **kwargs)
        return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Query a model on OpenRouter.")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model id")
    parser.add_argument(
        "--user-prompt",
        default="Guess the most likely meaning of the pseudoword 'trendipitious'.",
        help="user prompt to send",
    )
    parser.add_argument("--system-prompt", default=None, help="optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    
    system_prompt = args.system_prompt
    user_prompt = args.user_prompt

    client = OpenRouterClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    print(client.query_text(user_prompt, system_prompt=system_prompt))

if __name__ == "__main__":
    main()

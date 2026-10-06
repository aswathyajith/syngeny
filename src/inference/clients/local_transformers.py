import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def default_device():
    """Picks the best available device."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str):
    """Picks a sensible dtype for the device (CPU is unreliable in half precision)."""

    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class LocalTransformersClient:
    """Queries a Hugging Face model loaded locally with transformers."""

    def __init__(
            self,
            model: str,
            temperature: float = 0.0,
            max_tokens: int = 256,
            device: str | None = None,
            dtype: str | None = None,
            use_chat_template: bool | None = None,
            stop: list[str] | None = None,
        ):

        self.model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_chat_template = use_chat_template
        self.stop = stop
        self.device = device or default_device()
        torch_dtype = DTYPES[dtype] if dtype else default_dtype(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()

    def build_prompt(
            self,
            prompt: str,
            system_prompt: str | None = None,
            messages: list[dict] | None = None,
        ):
        """Renders the prompt, either through the chat template or as plain text.

        `use_chat_template` controls which: True always applies the template,
        False always flattens the roles into one plain-text prompt, and None
        applies the template only if the tokenizer has one.

        Prefer passing it explicitly for base models. Template presence does not
        imply instruction tuning -- Qwen ships a ChatML template on its base
        checkpoints, and prompting those in chat format gives degenerate output.
        """

        if messages is None:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        has_template = self.tokenizer.chat_template is not None
        use_template = has_template if self.use_chat_template is None else self.use_chat_template

        if use_template and not has_template:
            raise ValueError(f"'{self.model_id}' has no chat template to apply.")
        if not use_template:
            return "\n\n".join(message["content"] for message in messages)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def query(
            self,
            prompt: str = "Hello!",
            system_prompt: str | None = None,
            messages: list[dict] | None = None,
            **kwargs,
        ):
        """Generates a continuation and returns the raw output token ids."""

        text = self.build_prompt(prompt, system_prompt=system_prompt, messages=messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # A temperature of 0 has no meaning for sampling; it is greedy decoding.
        sampling = (
            {"do_sample": False}
            if self.temperature == 0.0
            else {"do_sample": True, "temperature": self.temperature}
        )

        # `stop_strings` needs the tokenizer to match the strings against the decoded text.
        stopping = {"stop_strings": self.stop, "tokenizer": self.tokenizer} if self.stop else {}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **sampling,
                **stopping,
                **kwargs,
            )

        self.prompt_length = inputs["input_ids"].shape[-1]
        return output_ids

    def query_text(self, prompt: str = "Hello!", system_prompt: str | None = None, **kwargs):
        """Generates a continuation and returns the new text only."""

        output_ids = self.query(prompt, system_prompt=system_prompt, **kwargs)
        completion_ids = output_ids[0][self.prompt_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Query a Hugging Face model locally.")
    parser.add_argument("--model", default="EleutherAI/pythia-1b", help="Hugging Face model id")
    parser.add_argument(
        "--user-prompt",
        default="Guess the most likely meaning of the pseudoword 'trendipitious'.",
        help="user prompt to send",
    )
    parser.add_argument("--system-prompt", default=None, help="optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: autodetect)")
    parser.add_argument("--dtype", default=None, choices=list(DTYPES), help="default: per device")
    parser.add_argument("--stop", default=None, nargs="+", help="strings that end generation")
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force the chat template on or off; --no-chat-template for base models "
             "(default: on whenever the tokenizer has one)",
    )
    args = parser.parse_args()

    client = LocalTransformersClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        device=args.device,
        dtype=args.dtype,
        use_chat_template=args.chat_template,
        stop=args.stop,
    )

    print(client.query_text(args.user_prompt, system_prompt=args.system_prompt))

if __name__ == "__main__":
    main()

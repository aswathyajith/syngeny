OR_MODEL_LABEL_MAP = {
    "openai/gpt-4o-mini": "GPT-4o Mini",
    # "openai/gpt-5.2": "GPT-5.2",
    # "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    # "openai/gpt-5-nano": "GPT-5 Nano", 
    "openai/gpt-5-mini": "GPT-5 Mini", 
    # "openai/gpt-4.1-nano": "GPT-4.1 Nano", 
    "openai/gpt-4.1-mini": "GPT-4.1 Mini", 
    "openai/gpt-5.4-mini": "GPT-5.4 Mini"
}
OR_MODELS = list(OR_MODEL_LABEL_MAP.keys())

MODEL_LABEL_MAP = {
    "Qwen2.5": {
        "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": "Qwen2.5-72B-Instruct-GPTQ-Int4",
    },
    "Gemma3": {
        "google/gemma-3-1b-it": "Gemma3-1B-IT",
        "google/gemma-3-4b-it": "Gemma3-4B-IT",
        "google/gemma-3-12b-it": "Gemma3-12B-IT"
    },
    "Llama": {
        "meta-llama/Llama-3.2-1B-Instruct": "Llama3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct": "Llama3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B-Instruct",
    }

}


DATASETS = {
    "sample-10": "Fake", 
    "sample-10-real": "Real"
}

HF_DATASETS = {
    
}
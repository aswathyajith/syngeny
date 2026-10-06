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

HF_MODEL_LABEL_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B",
    # "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B",
    # "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5 3B",
    # "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B",
    "EleutherAI/pythia-1b": "Pythia 1B"
}
HF_MODELS = list(HF_MODEL_LABEL_MAP.keys())

DATASETS = {
    "sample-10": "Fake", 
    "sample-10-real": "Real"
}

HF_DATASETS = {
    
}
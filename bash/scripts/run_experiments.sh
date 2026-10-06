#!/bin/bash

./bash/load_env.sh

models=("Qwen/Qwen2.5-0.5B-Instruct" "EleutherAI/pythia-1b")
model=Qwen/Qwen2.5-0.5B-Instruct

python src/run_prompts.py \
    --provider huggingface \
    --model-ids $model \
    --prompt-path-fmt data/processed/p1/{dataset_name}.jsonl \
    --responses-dir-fmt responses/{dataset_name}/{model}/tmp={temperature} \
    --results-path-fmt results/{dataset_name}/{model}/tmp={temperature}/results.jsonl \
    --max-tokens 64

#!/bin/bash

source bash/load_env.sh

seed=42

instruct_models=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
temps=(0.5)
seeds=(42 99)

for seed in "${seeds[@]}"
do
    for temperature in "${temps[@]}"
    do
        for model in "${instruct_models[@]}"
        do
            python src/run_prompts.py \
                --provider huggingface \
                --model-ids $model \
                --prompt-path-fmt data/processed/p1/{dataset_name}.jsonl \
                --responses-dir-fmt responses/{dataset_name}/{model}/tmp={temperature}/seed={seed} \
                --results-path-fmt results/{dataset_name}/{model}/tmp={temperature}/seed={seed}/results.jsonl \
                --seed $seed \
                --temperature $temperature \
                --max-tokens 64
        done
    done
done

# base_models=("Qwen/Qwen2.5-0.5B" "Qwen/Qwen2.5-1.5B")

# for model in "${base_models[@]}"
# do
#     python src/run_prompts.py \
#         --provider huggingface \
#         --model-ids $model \
#         --prompt-path-fmt data/processed/p2/{dataset_name}.jsonl \
#         --responses-dir-fmt responses/{dataset_name}/{model}/tmp={temperature}/seed={seed} \
#         --results-path-fmt results/{dataset_name}/{model}/tmp={temperature}/seed={seed}/results.jsonl \
#         --seed $seed \
#         --max-tokens 64
# done
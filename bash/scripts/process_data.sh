#!/bin/bash


# Build prompts 
python src/data/build_prompts.py \
    --prompt-template-path data/prompt_templates/tasks/guess_meaning/p1 \
    --data-path data/datasets/sample-10-real.csv \
    --save-path data/processed/p1/sample-10-real.jsonl 
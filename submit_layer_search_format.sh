#!/bin/bash

model_name="phi-3"
# model_name="gemma-2-2b-it"

gpu_req="a100_80gb:1"
# gpu_req="rtx_4090:1"
gpu_req="rtx_3090:1"

# arguments
dry_run=false

constraint=exclude_w_exclude_rep
max_generation_length=256
n_examples_per_instruction=10


sbatch --output="${HOME}/bsub_logs/steering/format/search-${model_name}-ex${n_examples_per_instruction}" \
    --job-name="ls-${model_name}" \
    -n 4 \
    --gpus=1 \
    --gres=gpumem:30g \
    --mem-per-cpu=50G \
    --time=23:59:00 \
--wrap="python ifeval_experiments/find_best_layer.py model_name=$model_name max_generation_length=$max_generation_length n_examples_per_instruction=$n_examples_per_instruction
"


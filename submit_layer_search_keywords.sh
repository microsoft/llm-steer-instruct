#!/bin/bash

model_name="phi-3"
model_name="gemma-2-2b-it"

gpu_req="a100_80gb:1"
gpu_req="rtx_4090:1"
gpu_req="rtx_3090:1"

# arguments
dry_run=false

model_name=gemma-2-2b-it
constraint=include
max_generation_length=256
include_instruction=false
steering_weights=[50,75,100,125,150]
specific_instruction=existence
n_examples=20
#   --gpus="${gpu_req}" \
#--gres=gpumem:25g \

sbatch --output="${HOME}/bsub_logs/steering/keywords/search-${model_name}-instr-${include_instructions}" \
    --job-name="ls-${model_name}" \
    -n 4 \
    --gpus=1 \
    --gres=gpumem:30g \
    --mem-per-cpu=50G \
    --time=23:59:00 \
--wrap="python keywords/find_best_layer.py model_name=$model_name constraint=$constraint max_generation_length=$max_generation_length include_instruction=$include_instruction steering_weights=$steering_weights
"


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
include_instruction=false
# steering_weights=[80,90,100,110,120,140]
# steering_weights=[240,250,260,270,280,290,300]
steering_weights=[30,40,50,70,90] # for the forbiddden rep
n_examples=10
#   --gpus="${gpu_req}" \
#--gres=gpumem:25g \

sbatch --output="${HOME}/bsub_logs/steering/keywords/search-${model_name}-instr-${include_instruction}" \
    --job-name="ls-${model_name}" \
    -n 4 \
    --gpus=1 \
    --gres=gpumem:30g \
    --mem-per-cpu=50G \
    --time=23:59:00 \
--wrap="python keywords/find_best_layer.py model_name=$model_name constraint=$constraint max_generation_length=$max_generation_length include_instruction=$include_instruction steering_weights=$steering_weights
"


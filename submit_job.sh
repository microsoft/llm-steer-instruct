#!/bin/bash

model_name="phi-3"

gpu_req="a100_80gb:1"
gpu_req="rtx_4090:1"
gpu_req="rtx_3090:1"

# arguments
dry_run=false
data_path=./data/keyword_test_inclusion.jsonl

source_layer_idx=24
steering_weight=120
steering=add_vector
include_instructions=true

specific_instruction=existence
n_examples=20
#   --gpus="${gpu_req}" \
#--gres=gpumem:25g \

sbatch --output="${HOME}/bsub_logs/steering/keywords/incl-${model_name}-${steering}-instr-${include_instructions}" \
    --job-name="kw-${model_name}" \
    -n 4 \
    --gpus="${gpu_req}" \
    --mem-per-cpu=50G \
    --time=23:59:00 \
--wrap="python keywords/eval_keyword_constraints.py data_path=$data_path source_layer_idx=$source_layer_idx steering_weight=$steering_weight steering=$steering specific_instruction=$specific_instruction n_examples=$n_examples include_instructions=$include_instructions dry_run=$dry_run
"


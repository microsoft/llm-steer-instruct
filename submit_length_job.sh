#!/bin/bash

model_name="phi-3"
# model_name="gemma-2-2b-it"

gpu_req="a100_80gb:1"
gpu_req="rtx_4090:1"
gpu_req="rtx_3090:1"



# arguments
dry_run=false

source_layer_idx=12
steering_weight=20
steering_weights=[$steering_weight]
# steering=add_vector_conciseness
steering=add_vector_length_specific
# steering=none
include_instructions=false
length_rep_file=6sentences_50examples_hs.h5

constraint_type=at_most
# constraint_type=exactly

n_examples=50
#   --gpus="${gpu_req}" \
#--gres=gpumem:25g \

sbatch --output="${HOME}/bsub_logs/steering/length/${model_name}-${steering}-L${source_layer_idx}-ex$n_examples-W$steering_weight-instr-${include_instructions}" \
    --job-name="l-${model_name}" \
    -n 4 \
    --gpus=1 \
    --gres=gpumem:20g \
    --mem-per-cpu=50G \
    --time=23:59:00 \
--wrap="python length_constraints/evaluate_length_constraints.py model_name=$model_name source_layer_idx=$source_layer_idx steering_weights=$steering_weights steering=$steering  n_examples=$n_examples include_instructions=$include_instructions constraint_type=$constraint_type dry_run=$dry_run length_rep_file=$length_rep_file
"


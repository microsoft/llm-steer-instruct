#!/bin/bash

model_name="phi-3"
model_name="gemma-2-2b-it"

gpu_req="a100_80gb:1"
gpu_req="v100:1"
# gpu_req="rtx_4090:1"
# gpu_req="rtx_3090:1"

# arguments
dry_run=false

source_layer_idx=-1
steering=adjust_rs
include_instructions=false
cross_model_steering=false
nonparametric_only=true

if [ "$include_instructions" = true ]; then
    data_path='./data/input_data_single_instr.jsonl'
else
    data_path='./data/input_data_single_instr_no_instr.jsonl'
fi

#   --gpus="${gpu_req}" \
#--gres=gpumem:25g \

sbatch --output="${HOME}/bsub_logs/steering/format/${model_name}-${steering}-instr-${include_instructions}-cross-${cross_model_steering}" \
    --job-name="f-${model_name}" \
    -n 4 \
    --gpus=1 \
    --gres=gpumem:20g \
    --mem-per-cpu=50G \
    --time=23:59:00 \
--wrap="python ifeval_experiments/ifeval_evaluation.py model_name=$model_name data_path=$data_path source_layer_idx=$source_layer_idx steering=$steering  nonparametric_only=$nonparametric_only cross_model_steering=$cross_model_steering dry_run=$dry_run
"


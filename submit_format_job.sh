#!/bin/bash

model_name="phi-3"
# model_name="mistral-7b-instruct"
model_name="gemma-2-2b"

gpu_req="a100_80gb:1"
# gpu_req="v100:1"
# gpu_req="a100-pcie-40gb:1"
# gpu_req="rtx_4090:1"
# gpu_req="rtx_3090:1"

# arguments
dry_run=false

source_layer_idx=-1
steering=adjust_rs

include_instructions=true

cross_model_steering=false

nonparametric_only=true

if [ "$include_instructions" = true ]; then
    data_path='./data/input_data_single_instr.jsonl'
else
    data_path='./data/input_data_single_instr_no_instr.jsonl'
fi



if [ "$model_name" = mistral-7b-instruct ]; then
    max_generation_length=1024
else
    max_generation_length=2048
fi

#max_generation_length=1024

echo $model_name
echo $include_instructions
echo $max_generation_length

#   --gpus="${gpu_req}" \
#--gres=gpumem:25g \

sbatch --output="${HOME}/bsub_logs/steering/format/${model_name}-${steering}-instr-${include_instructions}-cross-${cross_model_steering}" \
    --job-name="f-${model_name}" \
    -n 4 \
    --gpus=1 \
    --gpus="${gpu_req}" \
    --mem-per-cpu=25G \
    --time=23:59:00 \
--wrap="python ifeval_experiments/ifeval_evaluation.py model_name=$model_name data_path=$data_path source_layer_idx=$source_layer_idx steering=$steering  nonparametric_only=$nonparametric_only cross_model_steering=$cross_model_steering dry_run=$dry_run max_generation_length=$max_generation_length
"


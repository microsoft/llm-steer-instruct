#!/bin/bash

model_name="phi-3"
# model_name="gemma-2-2b-it"

gpu_req="a100_80gb:1"
gpu_req="rtx_4090:1"
gpu_req="rtx_3090:1"

source_layer_indices_phi=( 24 26 28 )
source_layer_indices_gemma=( 22 24 )

# specific_instruction=existence_validation
specific_instruction=forbidden_validation_w_forbidden_rep
if [[ "$specific_instruction" == "existence_validation" ]]; then
    data_path=./data/keyword_test_inclusion_likely.jsonl
    steering_weights_phi=( 40 60 80 100 )
    steering_weights_gemma=( 60 80 100 120 )

elif [[ "$specific_instruction" == "forbidden_validation" ]]; then
    data_path=./data/keyword_test_exclusion_likely.jsonl
    steering_weights_phi=( -60 -80 -100 -120 -150)
    steering_weights_gemma=(-120 -150 -200 -250 )

elif [[ "$specific_instruction" == "forbidden_validation_w_forbidden_rep" ]]; then
    data_path=./data/keyword_test_exclusion_likely.jsonl
    steering_weights_phi=( 30 40 50 70 90)
fi

if [[ "$model_name" == "phi-3" ]]; then
    source_layer_indices=$source_layer_indices_phi
    steering_weights=$steering_weights_phi
elif [[ "$model_name" == "gemma-2-2b-it" ]]; then
    source_layer_indices=$source_layer_indices_gemma
    steering_weights=$steering_weights_gemma
fi

# arguments
dry_run=false


steering=add_vector
# steering=0
include_instructions=false


max_generation_length=256
# specific_instruction=forbidden
n_examples=10
#   --gpus="${gpu_req}" \
#--gres=gpumem:25g \

for steering_weight in "${steering_weights_phi[@]}";
do
    echo $steering_weight
    for source_layer_idx in "${source_layer_indices_phi[@]}";
    do

    sbatch --output="${HOME}/bsub_logs/steering/keywords/${specific_instruction}-${model_name}-${steering}-instr-${include_instructions}" \
        --job-name="kw-${model_name}" \
        -n 4 \
        --gpus=1 \
        --gres=gpumem:20g \
        --mem-per-cpu=50G \
        --time=23:59:00 \
    --wrap="python keywords/eval_keyword_constraints.py data_path=$data_path model_name=$model_name source_layer_idx=$source_layer_idx steering_weight=$steering_weight steering=$steering specific_instruction=$specific_instruction n_examples=$n_examples include_instructions=$include_instructions dry_run=$dry_run max_generation_length=$max_generation_length
    "
    done
done



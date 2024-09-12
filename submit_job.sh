#!/bin/bash

model_name="phi-3"

k=100
data_range_start=0
data_range_end=1000
batch_size=2
chunk_size=10
get_rank_of_correct_token=1

gpu_req="a100_80gb:1"


# arguments
dry_run=true
data_path=./data/keyword_test_inclusion.jsonl

source_layer_idx=22
steering_weight=150
steering=add_vector

specific_instruction=existence
n_examples=20
#   --gpus="${gpu_req}" \

sbatch --output="${HOME}/bsub_logs/steering/keywords/${model_name}-${k}" \
    --job-name="kw-${model_name}" \
    -n 1 \
    --gres=gpumem:25g \
    --mem-per-cpu=400G \
    --time=23:59:00 \
--wrap="python keywords/eval_keyword_constraints.py --data_path $data_path --source_layer_idx $source_layer_idx --steering_weight $steering_weight --steering $steering --specific_instruction $specific_instruction --n_examples $n_examples --dry_run $dry_run
"


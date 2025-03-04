
model_names=( gemma-2-2b gemma-2-9b )

modes=( instr_plus_adjust_rs_-1_perplexity_cross_model adjust_rs_-1_perplexity_cross_model instr_plus_adjust_rs_-1_perplexity adjust_rs_-1_perplexity )

for model_name in "${model_names[@]}";
do
    for mode in "${modes[@]}";
    do

        python3 -m eval.evaluation_main --input_data=../data/input_data.jsonl  --input_response_data=../ifeval_experiments/out/$model_name/single_instr/all_base_x_all_instr/$mode/out.jsonl  --output_dir=../ifeval_experiments/out/$model_name/single_instr/all_base_x_all_instr/$mode
    done 
done
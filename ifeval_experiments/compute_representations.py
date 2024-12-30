# %%
import os
if 'ifeval_experiments' in os.getcwd():
    os.chdir('..')

import sys
sys.path.append(os.getcwd())

import torch
import pandas as pd
import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import if_inference
import json
from omegaconf import DictConfig, OmegaConf
import hydra


def extract_representation(model, tokenizer, problem, device, num_final_tokens=8):
    """
    extract the representation of the final token in the direct inference prompt
    """
    eval_prompt = problem

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(model_input['input_ids'])
        del logits
        final_token_rs = torch.stack([cache['resid_post', layer_idx][:, -num_final_tokens:, :].squeeze() for layer_idx in range(model.cfg.n_layers)]).cpu().numpy()
        del cache
    
    return final_token_rs


@hydra.main(config_path='../config', config_name='compute_representations')
def compute_representations(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    if 'all_base_x_all_instructions' in args.data_path:
        with open(args.data_path) as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]

        joined_df = pd.DataFrame(data)
        # drop the column prompt
        joined_df = joined_df.drop(columns=['prompt'])
        # rename model_output to prompt
        joined_df = joined_df.rename(columns={'model_output': 'prompt'})
        # rename prompt_without_instruction to prompt_no_instr
        joined_df = joined_df.rename(columns={'prompt_without_instruction': 'prompt_no_instr'})

        joined_df['instruction_id_list'] = joined_df['single_instruction_id'].apply(lambda x: [x])

        all_instructions = joined_df.single_instruction_id.unique()

    else:
        with open(args.data_path) as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]

        data_df = pd.DataFrame(data)

        with open(args.data_no_instr_path) as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]

        data_no_instr_df = pd.DataFrame(data)
        data_no_instr_df = data_no_instr_df.drop(columns=['instruction_id_list', 'prompt_hash'])

        # join the dataframes using column "key"
        data_df = data_df.set_index('key')
        data_no_instr_df = data_no_instr_df.set_index('key')
        joined_df = data_df.join(data_no_instr_df, lsuffix='', rsuffix='_no_instr')

        all_instructions = list(set([ item for l in data_df.instruction_id_list for item in l]))

    # load tokenizer and model
    model_name = args.model_name
    with open(args.path_to_hf_token) as f:
        hf_token = f.read()
    model, tokenizer = load_model_from_tl_name(model_name, device=args.device, hf_token=hf_token)

    model.to(args.device)

    if 'all_base_x_all_instructions' in args.data_path:
        p_bar = tqdm.tqdm(total=len(joined_df))
    else:
        # compute number of entries in the dataframe with len(instruction_id_list) == 1
        p_bar = tqdm.tqdm(total=len(joined_df[joined_df['instruction_id_list'].apply(lambda x: len(x) == 1)]))

    for instruction_type in all_instructions:
        instr_data_df = joined_df[[[instruction_type] == l for l in joined_df['instruction_id_list'] ]]
        instr_data_df.reset_index(inplace=True, drop=True)

        if args.use_data_subset:
            instr_data_df = instr_data_df.iloc[:int(len(instr_data_df)*args.data_subset_ratio)]

        if args.dry_run:
            instr_data_df = instr_data_df.head(2)

        num_final_tokens = 1
        rows = []

        # Run the model on each input
        for i, r in instr_data_df.iterrows():
            row = dict(r)

            if 'gemma' in model_name and '-it' not in model_name:
                print('Using no-IT Gemma: not using chat template')
                example = f'Q: {row["prompt"]}\nA:'
                example_no_instr = f'Q: {row["prompt_no_instr"]}\nA:'
            else:
                messages = [{"role": "user", "content": row['prompt']}]
                example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                messages_no_instr = [{"role": "user", "content": row['prompt_no_instr']}]
                example_no_instr = tokenizer.apply_chat_template(messages_no_instr, add_generation_prompt=True, tokenize=False)

            out1 = if_inference(model, tokenizer, example, args.device, max_new_tokens=args.max_generation_length)
            last_token_rs = extract_representation(model, tokenizer, example, args.device, num_final_tokens)
            row['output'] = out1
            row['last_token_rs'] = last_token_rs

            out2 = if_inference(model, tokenizer, example_no_instr, args.device, max_new_tokens=args.max_generation_length)
            last_token_rs = extract_representation(model, tokenizer, example_no_instr, args.device, num_final_tokens)
            row['output_no_instr'] = out2
            row['last_token_rs_no_instr'] = last_token_rs

            rows.append(row)
            p_bar.update(1)

        df = pd.DataFrame(rows)

        if args.dry_run:
            break

        folder = f'./representations/{model_name}/single_instr'
        if args.use_data_subset:
            folder += f'_subset_{args.data_subset_ratio}'
        if 'all_base_x_all_instructions' in args.data_path:
            folder += '_all_base_x_all_instr'
        os.makedirs(folder, exist_ok=True)
        # store the df
        try:
            df.to_hdf(f'{folder}/{"".join(instruction_type).replace(":", "_")}.h5', key='df', mode='w')
        except:
            print(f'Error storing {"".join(instruction_type).replace(":", "_")}.h5')

# %%
if __name__ == '__main__':
    compute_representations()
# %%
# %%
import os
import sys
import pandas as pd
import tqdm
import json
from omegaconf import DictConfig, OmegaConf
import hydra

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
os.chdir(project_dir)
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate, extract_representation

config_path = os.path.join(project_dir, 'config/format')


@hydra.main(config_path=config_path, config_name='compute_representations')
def compute_representations(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    with open(f'{project_dir}/{args.data_path}') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_df = pd.DataFrame(data)

    data_df['instruction_id_list'] = data_df['single_instruction_id'].apply(lambda x: [x])

    all_instructions = data_df.single_instruction_id.unique()

    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    data_df = data_df[data_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]

    # load tokenizer and model
    model_name = args.model_name
    model, tokenizer = load_model_from_tl_name(model_name, device=args.device, cache_dir=args.transformers_cache_dir)
    model.to(args.device)

    p_bar = tqdm.tqdm(total=len(data_df))

    for instruction_type in all_instructions:
        instr_data_df = data_df[[[instruction_type] == l for l in data_df['instruction_id_list']]]
        instr_data_df.reset_index(inplace=True, drop=True)

        if args.use_data_subset:
            instr_data_df = instr_data_df.iloc[:int(len(instr_data_df)*args.data_subset_ratio)]

        if args.dry_run:
            instr_data_df = instr_data_df.head(5)

        rows = []

        # Run the model on each input
        for i, r in instr_data_df.iterrows():
            row = dict(r)

            if 'gemma' in model_name and '-it' not in model_name:
                print('Using no-IT Gemma: not using chat template')
                example = f'Q: {row["prompt"]}\nA:'
                example_no_instr = f'Q: {row["prompt_without_instruction"]}\nA:'
            else:
                messages = [{"role": "user", "content": row['prompt']}]
                example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                messages_no_instr = [{"role": "user", "content": row['prompt_without_instruction']}]
                example_no_instr = tokenizer.apply_chat_template(messages_no_instr, add_generation_prompt=True, tokenize=False)

            out1 = generate(model, tokenizer, example, args.device, max_new_tokens=args.max_generation_length)
            last_token_rs = extract_representation(model, tokenizer, example, args.device, args.num_final_tokens)
            row['output'] = out1
            row['last_token_rs'] = last_token_rs

            out2 = generate(model, tokenizer, example_no_instr, args.device, max_new_tokens=args.max_generation_length)
            last_token_rs = extract_representation(model, tokenizer, example_no_instr, args.device, args.num_final_tokens)
            row['output_no_instr'] = out2
            row['last_token_rs_no_instr'] = last_token_rs

            rows.append(row)
            p_bar.update(1)

        df = pd.DataFrame(rows)

        if not args.dry_run:
            folder = f'{script_dir}/representations/{model_name}'
            if args.use_data_subset:
                folder += f'/subset_{args.data_subset_ratio}'
            else:
                folder += '/all'
            os.makedirs(folder, exist_ok=True)

            # store the df
            df.to_hdf(f'{folder}/{"".join(instruction_type).replace(":", "_")}.h5', key='df', mode='w')

# %%
if __name__ == '__main__':
    compute_representations()
# %%
# %%
import os
import sys
import pandas as pd
import tqdm
import json
from omegaconf import DictConfig
import hydra
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(script_dir)
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate, extract_representation

config_path = os.path.join(project_dir, 'config/keywords')


# %%
@hydra.main(config_path=config_path, config_name='compute_keyword_representations')
def compute_representations(args: DictConfig):
        
    # load the data
    with open(f'{project_dir}/{args.base_queries_path}') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_no_instr_df = pd.DataFrame(data)
    data_no_instr_df = data_no_instr_df.drop(columns=['prompt', 'prompt_hash'])
    # rename model_output to prompt_no_instr
    data_no_instr_df = data_no_instr_df.rename(columns={'model_output': 'prompt_no_instr'})

    new_rows = []
   
    phrasings_exclude = [' Do not include the word {}.', ' Make sure not to include the word "{}".', ' Do not use the word {}.', ' Do not say "{}".', ' Please exclude the word "{}".', ' The output should not contain the word "{}".']
    phrasings_include = [' Make sure to include the word "{}".', ' Please include the word "{}".', ' The output should contain the word "{}".', ' The output must contain the word "{}".', ' The output should say the word "{}".']

    if args.keyword_set == 'ifeval_exclude':
        # load ifeval keywords
        with open(f'{project_dir}/data/keywords/ifeval_keywords_exclude.txt') as f:
            word_list = f.readlines()
            word_list = [w.strip() for w in word_list]
    elif args.keyword_set == 'ifeval_include':
        # load ifeval keywords
        with open(f'{project_dir}/data/keywords/ifeval_keywords_include.txt') as f:
            word_list = f.readlines()
            word_list = [w.strip() for w in word_list]
    elif args.keyword_set == 'validation': 
        with open(f'{project_dir}/data/keywords/inclusion_validation.jsonl') as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]
            df = pd.DataFrame(data)
            word_list = list(set([w for l in df['likely_words'] for w in l]))
    else:
        raise ValueError(f'Unknown keyword_set: {args.keyword_set}')

    # exclude the examples that have "keyword" in the instruction_id_list
    data_no_instr_df = data_no_instr_df[~data_no_instr_df['instruction_id_list'].apply(lambda x: any(['keyword' in instr for instr in x]))]

    data_no_instr_df = data_no_instr_df.head(args.n_examples)

    for word in word_list:
        for i, r in data_no_instr_df.iterrows():
            if args.constraint_type == 'include':
                phrasings = phrasings_include
            elif args.constraint_type == 'exclude':
                phrasings = phrasings_exclude
            phrasing = random.choice(phrasings)
            row = dict(r)
            instr = phrasing.format(word)
            row['prompt_with_constraint'] = row['prompt_no_instr'] + instr
            row['word'] = word
            new_rows.append(row)

    data_df = pd.DataFrame(new_rows)

    # load tokenizer and model
    model_name = args.model_name
    model, tokenizer = load_model_from_tl_name(model_name, device=args.device, cache_dir=args.transformers_cache_dir)
    model.to(args.device)

    rows = []
    
    p_bar = tqdm.tqdm(total=len(data_df))

    # Run the model on each input
    for i, r in data_df.iterrows():
        row = dict(r)
        example = row['prompt_with_constraint']
        example_no_instr = row['prompt_no_instr']

        messages = [{"role": "user", "content": example}]
        example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        messages_no_instr = [{"role": "user", "content": example_no_instr}]
        example_no_instr = tokenizer.apply_chat_template(messages_no_instr, add_generation_prompt=True, tokenize=False)

        out1 = generate(model, tokenizer, example, args.device, max_new_tokens=args.max_new_tokens)
        last_token_rs = extract_representation(model, tokenizer, example, args.device, args.num_final_tokens)
        row['output'] = out1
        row['last_token_rs'] = last_token_rs

        out2 = generate(model, tokenizer, example_no_instr, args.device,  max_new_tokens=args.max_new_tokens)
        last_token_rs = extract_representation(model, tokenizer, example_no_instr, args.device, args.num_final_tokens)
        row['output_no_instr'] = out2
        row['last_token_rs_no_instr'] = last_token_rs

        rows.append(row)
        p_bar.update(1)

    df = pd.DataFrame(rows)

    folder = f'{script_dir}/representations/{model_name}'
    os.makedirs(folder, exist_ok=True)
    # store the df
    out_file = f'{folder}/{args.constraint_type}_{args.keyword_set}_{args.n_examples}examples_hs.h5'
    
    print(f'Storing {out_file}')
    df.to_hdf(out_file, key='df', mode='w')


if __name__ == '__main__':
    compute_representations()

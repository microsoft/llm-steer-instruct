# %%
import os
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
elif 'cluster' in os.getcwd():
    os.chdir('/cluster/project/sachan/alessandro/llm-steer-instruct')
    print('We\'re on the sandbox machine')

import sys
sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct')

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import if_inference, generate_with_hooks, direction_ablation_hook, direction_projection_hook
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import random


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


# %%
@hydra.main(config_path='../config', config_name='compute_keyword_representations')
def compute_representations(args: DictConfig):
        
    # load the data
    with open('data/ifeval_wo_instructions.jsonl') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_no_instr_df = pd.DataFrame(data)
    data_no_instr_df = data_no_instr_df.drop(columns=['prompt', 'prompt_hash'])
    # rename model_output to prompt_no_instr
    data_no_instr_df = data_no_instr_df.rename(columns={'model_output': 'prompt_no_instr'})
    # %%
    new_rows = []
   
    phrasings_exclude = [' Do not include the word {}.', ' Make sure not to include the word "{}".', ' Do not use the word {}.', ' Do not say "{}".', ' Please exclude the word "{}".', ' The output should not contain the word "{}".']
    phrasings_include = [' Make sure to include the word "{}".', ' Please include the word "{}".', ' The output should contain the word "{}".', ' The output must contain the word "{}".', ' The output should say the word "{}".']

    if args.word_list.__len__() == 1 and args.word_list[0] == 'ifeval_exclude':
        # load ifeval keywords
        with open('data/ifeval_keywords_exclude.txt') as f:
            word_list = f.readlines()
            word_list = [w.strip() for w in word_list]
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'casing_plus_words':
        with open('data/ifeval_multiple_instr_casing_exclude.jsonl') as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]
            df = pd.DataFrame(data)
            word_list = [w for l in df['kwargs'].apply(lambda x: x[1]['forbidden_words']) for w in l]
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'validation_exclude':
        with open('data/keyword_validation.jsonl') as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]
            df = pd.DataFrame(data)
            word_list = [w for l in df['likely_words'] for w in l]
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'validation_include':
        with open('data/keyword_test_inclusion_likely.jsonl') as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]
            df = pd.DataFrame(data)
            word_list = list(set([w for l in df['likely_words'] for w in l]))
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'ifeval_include':
        # load ifeval keywords
        with open('data/ifeval_keywords_include.txt') as f:
            word_list = f.readlines()
            word_list = [w.strip() for w in word_list]
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'test_include':
        # load ifeval keywords
        with open('data/keyword_test.jsonl') as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]
            df = pd.DataFrame(data)
            word_list = [w for l in df['unlikely_words'] for w in l]
    else:
        word_list = args.word_list

    word_list = word_list
    print(f'LEN {len(word_list)}')
    print(f'word_list: {word_list}')

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

    # %%
    # load tokenizer and model
    model_name = args.model_name
    with open('hf_token.txt') as f:
        hf_token = f.read()
    model, tokenizer = load_model_from_tl_name(model_name, device=args.device, cache_dir=args.transformers_cache_dir, hf_token=hf_token)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(args.device)

    # %%
    rows = []
    
    p_bar = tqdm.tqdm(total=len(data_df))

    # Run the model on each input
    for i, r in data_df.iterrows():
        row = dict(r)
        example = row['prompt_with_constraint']
        example_no_instr = row['prompt_no_instr']

        print(f'example: {example}')
        print(f'example_no_instr: {example_no_instr}')
        print(f'word: {row["word"]}')

        messages = [{"role": "user", "content": example}]
        example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        messages_no_instr = [{"role": "user", "content": example_no_instr}]
        example_no_instr = tokenizer.apply_chat_template(messages_no_instr, add_generation_prompt=True, tokenize=False)

        out1 = if_inference(model, tokenizer, example, args.device, max_new_tokens=args.max_new_tokens)
        last_token_rs = extract_representation(model, tokenizer, example, args.device, args.num_final_tokens)
        row['output'] = out1
        row['last_token_rs'] = last_token_rs

        out2 = if_inference(model, tokenizer, example_no_instr, args.device,  max_new_tokens=args.max_new_tokens)
        last_token_rs = extract_representation(model, tokenizer, example_no_instr, args.device, args.num_final_tokens)
        row['output_no_instr'] = out2
        row['last_token_rs_no_instr'] = last_token_rs

        rows.append(row)
        p_bar.update(1)

    # %%
    df = pd.DataFrame(rows)


    folder = f'keywords/representations/{model_name}'
    os.makedirs(folder, exist_ok=True)
    # store the df

    if args.word_list.__len__() == 1 and args.word_list[0] == 'ifeval_exclude':
        out_file = f'{folder}/{args.constraint_type}_ifeval_exclude_{args.n_examples}examples_hs.h5'
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'casing_plus_words':
        out_file = f'{folder}/{args.constraint_type}_casing_plus_words_{args.n_examples}examples_hs.h5'
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'ifeval_include':
        out_file = f'{folder}/{args.constraint_type}_ifeval_include_{args.n_examples}examples_hs.h5'
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'validation_exclude':
        out_file = f'{folder}/{args.constraint_type}_validation_exclude_{args.n_examples}examples_hs.h5'
    elif args.word_list.__len__() == 1 and args.word_list[0] == 'validation_include':
        out_file = f'{folder}/{args.constraint_type}_validation_include_{args.n_examples}examples_hs.h5'
    else:
        out_file = f'{folder}/{args.constraint_type}_num_words{len(word_list)}_{args.n_examples}examples_hs.h5'
    
    print(f'Storing {out_file}')
    df.to_hdf(out_file, key='df', mode='w')

# %%
if __name__ == '__main__':
    compute_representations()
    exit(0)
# %%

# =============================================================================
# playground
# =============================================================================

model_name = 'phi-3'
device = 0
# load tokenizer and model
model_name = model_name
model, tokenizer = load_model_from_tl_name(model_name, device=device, hf_token=None)

model.to(device)
# %%
base_q = 'Write a a short article about AI.'
instr = ' Make sure to include the word "chocolate".'

example = base_q + instr

messages = [{"role": "user", "content": example}]
input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
output_toks = generate_with_hooks(model, input, fwd_hooks=[], max_tokens_generated=128)
output_str = model.tokenizer.batch_decode(output_toks[:, input.shape[1]:], skip_special_tokens=True)
# %%
print(output_str[0])
# %%

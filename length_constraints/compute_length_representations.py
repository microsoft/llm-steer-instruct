# %%
import os
import sys
if 'Users' in os.getcwd():
    os.chdir('C:\\Users\\t-astolfo\\workspace\\t-astolfo')
    print('We\'re on a Windows machine')
elif 'home' in os.getcwd():
    sys.path.append('/home/t-astolfo/t-astolfo')
    os.chdir('/home/t-astolfo/t-astolfo')

    print('We\'re on the sandbox machine')

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import if_inference, adjust_vectors
import json
import plotly.express as px
import plotly.graph_objects as go
import functools
from transformer_lens import utils as tlutils
import nltk
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


# %%
@hydra.main(config_path='../config', config_name='compute_length_representations')
def compute_representations(args: DictConfig):
        
    # load the data
    with open('data/ifeval_wo_instructions.jsonl') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_no_instr_df = pd.DataFrame(data)
    data_no_instr_df = data_no_instr_df.drop(columns=['prompt', 'instruction_id_list', 'prompt_hash'])
    # rename model_output to prompt_no_instr
    data_no_instr_df = data_no_instr_df.rename(columns={'model_output': 'prompt_no_instr'})
    # %%
    new_rows = []
    if args.constraint_type == 'sentences':
        phrasings_single = [' Answer using 1 sentence.', ' Respond with one sentence.', ' Provide an answer in one sentence.', ' Give your answer in a single sentence.']
        phrasings = [' Answer using {} sentences.', ' Respond with {} sentences.', ' Provide an answer in {} sentences.', ' Give your answer in {} sentences.']
        
        for i, r in data_no_instr_df.iterrows():
            for n_sent in range(1, args.n_sent_max + 1):
                row = dict(r)
                if n_sent == 1:
                    # sample a phrasing
                    instr = np.random.choice(phrasings_single)
                else:
                    phrasing = np.random.choice(phrasings)
                    instr = phrasing.format(n_sent)
                row['prompt_with_constraint'] = row['prompt_no_instr'] + instr
                row['length_constraint'] = n_sent - 1
                new_rows.append(row)
        
    elif args.constraint_type == 'high-level':
        phrasings_exatra_short = [' Be extremely concise.', ' Be extremely brief.', ' Keep it extremely short.', ' Keep it extremely concise.', ' The answer should be extremely concise.', ' The answer should be extremely brief.', ' The answer should be extremely short.']
        phrasings_short = [' Be concise.', ' Be brief.', ' Keep it short.', ' Keep it concise.', ' The answer should be concise.', ' The answer should be brief.', ' The answer should be short.']
        phrasings_medium = [' Don\'t be too concise or too verbose.', ' The answer should be neither too short nor too long.', ' The answer should be neither too concise nor too verbose.', ' The length of the answer should be moderate.']
        phrasings_verbose = [' Be verbose.', ' Provide a long answer.', ' The answer should be verbose.', ' The answer should be long.', ' The answer should be long.']
        phrasings_extra_verbose = [' Be extremely verbose.', ' Provide an extremely long answer.', ' The answer should be extremely verbose.', ' The answer should be extremely long.', ' The answer should be extremely long.']
        phrasings = [phrasings_exatra_short, phrasings_short, phrasings_medium, phrasings_verbose, phrasings_extra_verbose]

        for i, r in data_no_instr_df.iterrows():
            for j, phrasing in enumerate(phrasings):
                row = dict(r)
                instr = np.random.choice(phrasing)
                row['prompt_with_constraint'] = row['prompt_no_instr'] + instr
                row['length_constraint'] = j
                new_rows.append(row)


    data_df = pd.DataFrame(new_rows)


    # %%
    # load gpt2 tokenizer and model
    model_name = args.model_name
    with open('hf_token.txt') as f:
        hf_token = f.read()
    model, tokenizer = load_model_from_tl_name(model_name, device=args.device, cache_dir=None, hf_token=hf_token)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(args.device)

    # %%
    rows = []

    if args.constraint_type == 'sentences':
        limit = args.n_examples * args.n_sent_max
    elif args.constraint_type == 'high-level':
        limit = args.n_examples * len(phrasings)
    else:
        raise ValueError('Invalid constraint type')
    
    print(f'Computing representations for {limit} examples')

    p_bar = tqdm.tqdm(total=limit)

    # Run the model on each input
    for i, r in data_df.head(limit).iterrows():
        row = dict(r)
        example = row['prompt_with_constraint']
        example_no_instr = row['prompt_no_instr']

        # apply the chat template
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


    folder = f'length_constraints/representations/{model_name}'
    os.makedirs(folder, exist_ok=True)
    # store the df
    if args.constraint_type == 'sentences':
        out_file = f'{folder}/{args.n_sent_max}sentences_{args.n_examples}examples_hs.h5'
    elif args.constraint_type == 'high-level':
        out_file = f'{folder}/high_level_{args.n_examples}examples_hs.h5'
    
    print(f'Storing {out_file}')
    df.to_hdf(out_file, key='df', mode='w')
    # %%


# %%
if __name__ == '__main__':
    compute_representations()
# %%
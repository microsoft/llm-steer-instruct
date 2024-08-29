# %%
import os
if 'Users' in os.getcwd():
    os.chdir('C:\\Users\\t-astolfo\\workspace\\t-astolfo')
    print('We\'re on a Windows machine')
elif 'home' in os.getcwd():
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
from utils.generation_utils import if_inference
import json


# %%
# Some environment variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transformer_cache_dir = None

# %%
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
# load the data

with open('data/input_data.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df = pd.DataFrame(data)

with open('data/ifeval_wo_instructions.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_no_instr_df = pd.DataFrame(data)
data_no_instr_df = data_no_instr_df.drop(columns=['prompt', 'instruction_id_list', 'prompt_hash'])
# rename "model_output" to "prompt"
data_no_instr_df = data_no_instr_df.rename(columns={'model_output': 'prompt'})

# join the dataframes using column "key"
data_df = data_df.set_index('key')
data_no_instr_df = data_no_instr_df.set_index('key')
joined_df = data_df.join(data_no_instr_df, lsuffix='', rsuffix='_no_instr')


# select the data with the desired instruction type
#instruct_type = ['detectable_format:json_format']
#instruct_type = ['change_case:english_capital']
instruct_type =['detectable_format:number_bullet_lists']
instruct_type = ['punctuation:no_comma']
instr_data_df = joined_df[[instruct_type == l for l in data_df['instruction_id_list'] ]]
instr_data_df.reset_index(inplace=True, drop=True)


# %%
# load gpt2 tokenizer and model
model_name = 'phi-3'
with open('hf_token.txt') as f:
    hf_token = f.read()
model, tokenizer = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token)
#model = AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

# %%

num_final_tokens = 4
rows = []

# Run the model on each input
for i, r in tqdm.tqdm(instr_data_df.iterrows()):
    row = dict(r)
    example = r.prompt
    print(f'Example: {example}')
    example = row['prompt']
    example_no_instr = row['prompt_no_instr']

    # apply the chat template for phi-3
    messages = [{"role": "user", "content": example}]
    example = tokenizer.decode(tokenizer.apply_chat_template(messages))
    messages_no_instr = [{"role": "user", "content": example_no_instr}]
    example_no_instr = tokenizer.decode(tokenizer.apply_chat_template(messages_no_instr))
    print(f'Example: {example}')
    print(f'Example_no_instr: {example_no_instr}')

    out1 = if_inference(model, tokenizer, example, device)
    last_token_rs = extract_representation(model, tokenizer, example, device, num_final_tokens)
    row['output'] = out1
    print(f'Output: {out1}')
    row['last_token_rs'] = last_token_rs

    out2 = if_inference(model, tokenizer, example_no_instr, device)
    last_token_rs = extract_representation(model, tokenizer, example_no_instr, device, num_final_tokens)
    row['output_no_instr'] = out2
    print(f'Output no instr: {out2}')
    row['last_token_rs_no_instr'] = last_token_rs

    rows.append(row)
    print('---------------------------------')

# %%
df = pd.DataFrame(rows)
n_examples = len(df)

# %%
folder = f'stored_hs/if/{model_name}'
os.makedirs(folder, exist_ok=True)
# store the df
df.to_hdf(f'{folder}/{"".join(instruct_type).replace(":", "_")}_{n_examples}examples_hs_new.h5', key='df', mode='w')
# %%

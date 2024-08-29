# %%
import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import cot_inference, direct_inference

if 'Users' in os.getcwd():
    os.chdir('C:\\Users\\t-astolfo\\workspace\\t-astolfo')
    print('We\'re on a Windows machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on the server')

# %%
# Some environment variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transformer_cache_dir = None

# %%
def direct_extract_representation(model, tokenizer, problem, device, use_qa_pattern=True, num_final_tokens=8):
    """
    extract the representation of the final token in the direct inference prompt
    """
    if use_qa_pattern:
        eval_prompt = 'Q: ' + problem + "\nA: The answer (Arabic numerals) is "
    else:
        eval_prompt = problem + "\nThe answer (Arabic numerals) is "

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(model_input['input_ids'])
        del logits
        final_token_rs = torch.stack([cache['resid_post', layer_idx][:, -num_final_tokens:, :].squeeze() for layer_idx in range(model.cfg.n_layers)]).cpu().numpy()
        del cache
    
    return final_token_rs

def cot_extract_representation(model, tokenizer, problem, device, use_qa_pattern=True, num_final_tokens=8):
    """
    extract the representation of the final token in the cot inference prompt
    """
    if use_qa_pattern:
        eval_prompt = "Q: " + problem + "\nA: Let's think step by step. " 
    else:
        eval_prompt = problem + "\nLet's think step by step. " 

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(model_input['input_ids'])
        del logits
        final_token_rs = torch.stack([cache['resid_post', layer_idx][:, -num_final_tokens:, :].squeeze() for layer_idx in range(model.cfg.n_layers)]).cpu().numpy()
        del cache

    return final_token_rs


# %%
# load gpt2 tokenizer and model
model_name = 'mistral-7b-instruct'
with open('hf_token.txt') as f:
    hf_token = f.read()
model, tokenizer = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token)
#model = AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)
# %%
# Load the GSM8k dataset
dataset = load_dataset('gsm8k', 'main')

# %%
inference_type = 'cot' # 'cot' or 'direct'

n_examples = 100
num_final_tokens = 14
rows = []

# Run the model on each input
for example_idx in tqdm.tqdm(range(n_examples)):
    row = {}
    row['example_idx'] = example_idx
    example = dataset['test']['question'][example_idx]
    print(f'Example {example_idx}: {example}')
    row['example'] = example
    if inference_type == 'direct':
        out1 = direct_inference(model, tokenizer, example, device, True)
        out1_without_cot = out1
        last_token_rs = direct_extract_representation(model, tokenizer, example, device, True, num_final_tokens)
    else:
        out1, out1_without_cot = cot_inference(model, tokenizer, example, device, True)
        last_token_rs = cot_extract_representation(model, tokenizer, example, device, True, num_final_tokens)
    row['prediction_with_cot'] = out1
    row['prediction_without_cot'] = out1_without_cot
    match  = re.search(r'\d+', out1_without_cot)
    pred1  = match.group() if match else -1
    row['prediction_extracted'] = int(pred1)
    row['last_token_rs'] = last_token_rs
    gt_answer = dataset['test']['answer'][example_idx].split('####')[1]
    gt_answer = int(gt_answer)
    row['gt_answer'] = gt_answer
    rows.append(row)
    print('---------------------------------')

# %%
df = pd.DataFrame(rows)

# print accuracy
correct = (df['gt_answer'] == df['prediction_extracted']).sum()
print(f'Accuracy: {correct}/{n_examples} = {correct/n_examples}')
# %%
folder = f'stored_hs/{model_name}'
os.makedirs(folder, exist_ok=True)
# store the df
df.to_hdf(f'{folder}/{inference_type}_{n_examples}examples_hs.h5', key='df', mode='w')
# %%
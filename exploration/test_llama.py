# %%
import sys
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
    print('We\'re on a Windows machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on a remote machine')
# %%
import pandas as pd
import numpy as np
import os
import torch
import plotly.express as px
import sys
import plotly.graph_objects as go
from transformers import AutoTokenizer
import einops
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import adjust_vectors
import functools
from transformer_lens import utils as tlutils
from tqdm import tqdm

# %%
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_name = 'gpt2'

device = 'cpu' if torch.backends.mps.is_available() else 'cuda'
print(f"Using device: {device}")

with open('hf_token.txt') as f:
    hf_token = f.read()
transformer_cache_dir = None
#model, tokenizer = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token)
model_hf, tokenizer_hf = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token, hf_model=True)
model_hf.to(device)
# %%

input_hf = 'List some facts about Lionel Messi.'
# input_hf = tokenizer_hf.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
input_hf = tokenizer_hf(input_hf, return_tensors='pt').to(device)
output_toks_hf = model_hf.generate(**input_hf, max_length=128, do_sample=False)
output_str_hf = tokenizer_hf.batch_decode(output_toks_hf[:, input_hf['input_ids'].shape[1]:], skip_special_tokens=True)
print(output_str_hf[0])
# %%
input = 'List some facts about Lionel Messi.'
messages = [{"role": "user", "content": input}]
input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
logits = model(input.to(device))
logits_hf = model_hf(input.to(device)).logits
# %%
logits - logits_hf
# %%
import plotly.express as px

centered_logits = logits - logits.mean(dim=-1, keepdim=True)
centered_logits_hf = logits_hf - logits_hf.mean(dim=-1, keepdim=True)

# plot histogram of logits difference
logits_diff = (centered_logits - centered_logits_hf).flatten().detach()
fig = px.histogram(x=logits_diff.cpu().numpy(), nbins=100)
fig.show()
# %%

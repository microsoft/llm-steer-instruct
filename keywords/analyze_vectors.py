# %%
import os
import sys
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/keywords')
    print('We\'re on the local machine')
elif 'cluster' in os.getcwd():
    os.chdir('/cluster/project/sachan/alessandro/llm-steer-instruct')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct/keywords')
    print('We\'re on a sandbox machine')

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
model_name = 'phi-3'
# model_name = 'gemma-2-2b-it'
rep_file = f'./keywords/representations/{model_name}/include_validation_include_20examples_hs.h5'
# rep_file = f'./keywords/representations/{model_name}/exclude_validation_exclude_10examples_hs.h5'
device = 'cpu'
all_words_df = pd.read_hdf(rep_file)

# %%
all_words_df.head()
# %%
layer_idx = 26
mean_projections = []
mean_projections_no_instr = []
proj_deltas = []
instr_dirs = {}
for word in all_words_df['word'].unique():
    print(f'Word: {word}')
    df = all_words_df[all_words_df['word'] == word]
    hs_instr = df['last_token_rs'].values
    hs_instr = np.array([example_array for example_array in list(hs_instr)])
    hs_instr = torch.from_numpy(hs_instr)

    hs_no_instr = df['last_token_rs_no_instr'].values
    hs_no_instr = np.array([example_array for example_array in list(hs_no_instr)])
    hs_no_instr = torch.from_numpy(hs_no_instr)

    repr_diffs = hs_instr - hs_no_instr
    mean_repr_diffs = repr_diffs.mean(dim=0)
    last_token_mean_diff = mean_repr_diffs[layer_idx, :]

    instr_dir = last_token_mean_diff / last_token_mean_diff.norm()
    instr_dirs[word] = instr_dir

    # compute projection for inputs with instruction
    proj = hs_instr[:, layer_idx] @ instr_dir
    mean_proj = proj.mean()
    mean_projections.append(mean_proj)

    # compute projection for inputs without instruction
    proj_no_instr = hs_no_instr[:, layer_idx] @ instr_dir
    mean_proj_no_instr = proj_no_instr.mean()
    mean_projections_no_instr.append(mean_proj_no_instr)
    proj_deltas.append(mean_proj - mean_proj_no_instr)
    

# %%
print(f'mean proj with instr {np.sum(mean_projections) / len(mean_projections)}')
print(f'mean proj without instr {np.sum(mean_projections_no_instr) / len(mean_projections_no_instr)}')
print(f'mean proj delta {np.sum(proj_deltas) / len(proj_deltas)}')
# %%
model_name = 'phi-3'

device = 'mps' if torch.backends.mps.is_available() else 'cuda'
print(f"Using device: {device}")

with open('hf_token.txt') as f:
    hf_token = f.read()
transformer_cache_dir = None
model, tokenizer = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token)
# model_hf, tokenizer_hf = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token, hf_model=True)
model.to(device)
# %%
for word, instr_dir in instr_dirs.items():
    logits_projections = instr_dir.to(device) @ model.W_U
    # argsort logits_projections and take the top 10
    top_ids = torch.argsort(logits_projections, descending=True)[:10]
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.cpu().numpy())
    print(f'{word} - Layer {layer_idx}: {top_tokens}')
    print('----------------------')

# %%

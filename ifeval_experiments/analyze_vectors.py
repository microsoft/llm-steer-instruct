# %%
import os
import sys
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/ifeval_experiments')
    print('We\'re on the local machine')
elif 'cluster' in os.getcwd():
    os.chdir('/cluster/project/sachan/alessandro/llm-steer-instruct')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct/ifeval_experiments')
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
rep_dir = f'./ifeval_experiments/representations/{model_name}/single_instr_all_base_x_all_instr/'
device = 'cpu'
files = os.listdir(rep_dir)

# from the files, remove the ones that have "pre_computed" in the name
files = [f for f in files if 'pre_computed' not in f]
df = pd.read_hdf(rep_dir + files[0])

# max_length = min([x.shape[1] for x in results_df['last_token_rs_no_instr'].values])
# %%
# load pred_computed_ivs
file = f'./ifeval_experiments/representations/{model_name}/single_instr_all_base_x_all_instr/pre_computed_ivs_best_layer_validation_no_instr.h5'
df_ivs = pd.read_hdf(file)
best_layers_dict = {r.instruction.replace(':', '_') : r.max_diff_layer_idx for r in df_ivs.itertuples()}


# %%
instr_dirs = {}

for f in tqdm(files):
    if 'length' in f:
        continue
    if 'keywords' in f:
        continue
    if 'detectable_content_' in f:
        continue
    print(f)
    df = pd.read_hdf(rep_dir + f)
    hs_instr = df['last_token_rs'].values
    hs_instr = np.array([example_array for example_array in list(hs_instr)])
    hs_instr = torch.from_numpy(hs_instr)

    hs_no_instr = df['last_token_rs_no_instr'].values
    hs_no_instr = np.array([example_array for example_array in list(hs_no_instr)])
    hs_no_instr = torch.from_numpy(hs_no_instr)

    repr_diffs = hs_instr - hs_no_instr
    mean_repr_diffs = repr_diffs.mean(dim=0)
    last_token_mean_diff = mean_repr_diffs[:, -1, :]

    instr_dir = last_token_mean_diff / last_token_mean_diff.norm()

    f = f.replace('language_response_', '')
    f = f.replace('.h5', '')
    f = f.replace('detectable_format_', '')
    f = f.replace('detectable_content_', '')
    f = f.replace('change_case_', '')
    f = f.replace('punctuation_', '')
    f = f.replace('startend_', '')

    if f in best_layers_dict:
        layer_idx = best_layers_dict[f.replace('.h5', '')]
    else:
        layer_idx = -1
    best_layers_dict[f] = layer_idx

    instr_dirs[f] = instr_dir


# %%
# compute cosine similarity between all pairs of instructions
instr_dirs_df = torch.stack(list(instr_dirs.values()))
# cos_sim = torch.nn.functional.cosine_similarity(instr_dirs_df.unsqueeze(1), instr_dirs_df.unsqueeze(0), dim=2)
cos_sim = instr_dirs_df @ instr_dirs_df.T / (instr_dirs_df.norm(dim=1).unsqueeze(1) @ instr_dirs_df.norm(dim=1).unsqueeze(0))
# %%
# make heatmap of cosine similarities
fig = go.Figure(data=go.Heatmap(
                   z=cos_sim,
                   x=list(instr_dirs.keys()),
                   y=list(instr_dirs.keys()),
                   hoverongaps = False))
fig.update_layout(
    title='Cosine similarity between instruction vectors',
    xaxis_title='Instructions',
    yaxis_title='Instructions')

# incline the x-axis labels
fig.update_layout(xaxis=dict(tickangle=45))
fig.show()
# %%
# =============================================================================
# compute projection onto the vocabulary
# =============================================================================

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

for instruction in instr_dirs.keys():
    layer_idx = best_layers_dict[instruction]
    if layer_idx == -1:
        layer_idx = 30
    instr_dir = instr_dirs[instruction][layer_idx].to(device)
    logits_projections = instr_dir @ model.W_U
    # argsort logits_projections and take the top 10
    top_ids = torch.argsort(logits_projections, descending=True)[:10]
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.cpu().numpy())
    print(f'{instruction} - Layer {layer_idx}: {top_tokens}')
    print('----------------------')

# %%

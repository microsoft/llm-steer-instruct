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
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
from tqdm import tqdm

# %%
model_name = 'phi-3'
rep_dir = f'./ifeval_experiments/representations/{model_name}/single_instr_all_base_x_all_instr/'
device = 'cpu'
files = os.listdir(rep_dir)

# from the files, remove the ones that have "pre_computed" in the name
files = [f for f in files if 'pre_computed' not in f]
# %%
df = pd.read_hdf(rep_dir + files[0])

# max_length = min([x.shape[1] for x in results_df['last_token_rs_no_instr'].values])


# %%
layer_idx = 20
instr_dirs = {}

for f in tqdm(files):
    if 'length' in f:
        continue
    if 'keywords' in f:
        continue
    if 'language' in f:
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

    instr_dir = last_token_mean_diff[layer_idx] / last_token_mean_diff[layer_idx].norm()

    f = f.replace('language_response_', '')
    f = f.replace('.h5', '')
    f = f.replace('detectable_format_', '')
    f = f.replace('detectable_content_', '')
    f = f.replace('change_case_', '')
    f = f.replace('punctuation_', '')
    f = f.replace('startend_', '')

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

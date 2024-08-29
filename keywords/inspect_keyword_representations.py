# %%
import os
os.chdir('/home/t-astolfo/t-astolfo')

import pandas as pd
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import if_inference, generate_with_hooks, direction_ablation_hook, direction_projection_hook
import functools
from transformer_lens import utils as tlutils
# %%

folder = '/home/t-astolfo/t-astolfo/keywords/representations'

model_name = 'phi-3'
# model_name= 'gemma-2-2b-it'
constraint_type = 'include'
num_words = 1
n_examples = 20

# file = f'{folder}/{model_name}/{constraint_type}_num_words{num_words}_{n_examples}examples_hs.h5'
file = f'{folder}/{model_name}/{constraint_type}_ifeval_exclude_{n_examples}examples_hs.h5'
results_df = pd.read_hdf(file)

words = results_df.word.unique()
print(words)
# %%
word = words[6]
print(word)

results_df = results_df[results_df.word == 'station']


# %%
hs_instr = results_df['last_token_rs'].values
hs_instr = torch.tensor([example_array[:, :] for example_array in list(hs_instr)])
hs_no_instr = results_df['last_token_rs_no_instr'].values
hs_no_instr = torch.tensor([example_array[:, :] for example_array in list(hs_no_instr)])

# check if hs has 4 dimensions
print(f'hs_instr shape: {hs_instr.shape}')
if len(hs_instr.shape) == 3:
    hs_instr = hs_instr.unsqueeze(2)
    hs_no_instr = hs_no_instr.unsqueeze(2)
print(f'hs_instr shape: {hs_instr.shape}')


print(f'hs_instr shape: {hs_instr.shape}')
print(f'hs_no_instr shape: {hs_no_instr.shape}')

# print the size in MB of the tensors
print(f'hs_instr size: { hs_instr.element_size() * hs_instr.nelement() / 1024 / 1024 } MB')
# %%
cos_sims = torch.zeros((hs_instr.shape[1], hs_instr.shape[0] * 2, hs_instr.shape[0] * 2, hs_instr.shape[2]))
p_bar = tqdm(total=hs_instr.shape[1])
for layer_idx in range(hs_instr.shape[1]):
    # compute cosine similarity between representations for different examples
    concat = torch.cat([hs_instr[:, layer_idx, :, :], hs_no_instr[:, layer_idx, :, :]], dim=0)
    sim = torch.nn.functional.cosine_similarity(concat.unsqueeze(1), concat.unsqueeze(0), dim=-1)
    cos_sims[layer_idx] = sim
    p_bar.update(1)
cos_sims = cos_sims.permute(0, 3, 2, 1)
# %%
layer_idx = 12
token_idx = -1
fig = px.imshow(cos_sims[layer_idx, token_idx].cpu().numpy(), labels=dict(x='Example idx', y='Example idx', color='Cosine similarity'))
fig.update_layout(title=f'Layer {layer_idx} | Token {token_idx} | Cosine similarity between representations')
# add labels
fig.update_xaxes(title='Example idx')
fig.update_yaxes(title='Example idx')

# %%
# =============================================================================
# Line plot of similarity between representations
# =============================================================================

# compute the average cosine sim along the diagonal of cos_sims
avg_cos_sims = torch.zeros((hs_instr.shape[1], hs_instr.shape[2]))
baseline_sims_instr = torch.zeros((hs_instr.shape[1], hs_instr.shape[2]))
baseline_sims_no_instr = torch.zeros((hs_instr.shape[1], hs_instr.shape[2]))
for layer_idx in range(hs_instr.shape[1]):
    for token_idx in range(hs_instr.shape[2]):
        avg_cos_sims[layer_idx, token_idx] = torch.diagonal(cos_sims[layer_idx, token_idx], offset=-n_examples).mean()
        baseline_sims_instr[layer_idx, token_idx] = cos_sims[layer_idx, token_idx, :n_examples, :n_examples].mean()
        baseline_sims_no_instr[layer_idx, token_idx] = cos_sims[layer_idx, token_idx, n_examples+1:, n_examples+1:].mean()

# %%
token_idx = -1
# make a line plot of the average cosine similarity between representations
df = pd.DataFrame(avg_cos_sims[:, token_idx].cpu().numpy(), columns=['avg_cos_sim'])
fig = go.Figure()
x_labels = np.arange(avg_cos_sims.shape[0])
fig.add_trace(go.Scatter(x=x_labels, y=baseline_sims_instr[:, token_idx].cpu().numpy(), mode='lines', name='Sim. between examples w/ instr'))
fig.add_trace(go.Scatter(x=x_labels, y=baseline_sims_no_instr[:, token_idx].cpu().numpy(), mode='lines', name='Sim. between examples w/o instr'))
fig.add_trace(go.Scatter(x=x_labels, y=avg_cos_sims[:, token_idx].cpu().numpy(), mode='lines', name='Sim between same ex. w/ and w/o instr'))# add labels
fig.update_layout(title=f'Cosine Similarity Between Representations')
fig.update_xaxes(title='Layer idx')
fig.update_yaxes(title='Cosine similarity')
# resize the plot
fig.update_layout(width=400, height=300)
# remove padding
fig.update_layout(margin=dict(l=0, r=10, t=50, b=0))
# place legend at the bottom
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.85,
    xanchor="right",
    x=0.95
))
# save the plot as pdf
# fig.write_image(f'./plots/{model_name.replace('/', '_')}_cosine_similarity.pdf')
fig.show()

# %%
# =============================================================================
# compute the diff between the representations
# =============================================================================

repr_diffs = hs_instr - hs_no_instr
mean_repr_diffs = repr_diffs.mean(dim=0)
last_token_mean_diff = mean_repr_diffs[:, -1, :]

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('hf_token.txt') as f:
    hf_token = f.read()
transformer_cache_dir = None
model, tokenizer = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token)
model_hf, tokenizer_hf = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token, hf_model=True)
model_hf.to(device)
# %%

logits_projections = last_token_mean_diff.to(device) @ model.W_U
for layer_idx in range(mean_repr_diffs.shape[0]):
    # argsort logits_projections and take the top 10
    top_ids = torch.argsort(logits_projections[layer_idx], descending=True)[:10]
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.cpu().numpy())
    print(f'Layer {layer_idx}: {top_tokens}')
# %%
mean_hs_instr = hs_instr.mean(dim=0)[:, -1, :]
logits_projections = mean_hs_instr.to(device) @ model.W_U
for layer_idx in range(mean_hs_instr.shape[0]):
    # argsort logits_projections and take the top 10
    top_ids = torch.argsort(logits_projections[layer_idx], descending=True)[:10]
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.cpu().numpy())
    print(f'Layer {layer_idx}: {top_tokens}')
# %%
mean_hs_no_instr = hs_no_instr.mean(dim=0)[:, -1, :]
logits_projections = mean_hs_no_instr.to(device) @ model.W_U
for layer_idx in range(mean_hs_no_instr.shape[0]):
    # argsort logits_projections and take the top 10
    top_ids = torch.argsort(logits_projections[layer_idx], descending=True)[:10]
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.cpu().numpy())
    print(f'Layer {layer_idx}: {top_tokens}')


# %%
# =============================================================================
# projections along the instruction direction
# =============================================================================

layer_idx = 16

instr_dir = last_token_mean_diff[layer_idx] / last_token_mean_diff[layer_idx].norm()

# average projection along the instruction direction
proj = hs_instr[:, layer_idx, -1, :].to(device) @ instr_dir.to(device)
proj_no_instr = hs_no_instr[:, layer_idx, -1, :].to(device) @ instr_dir.to(device)

# make two overlayed histograms
fig = go.Figure()
fig.add_trace(go.Histogram(x=proj.cpu().numpy(), name='instr', nbinsx=6))
fig.add_trace(go.Histogram(x=proj_no_instr.cpu().numpy(), name='no_instr', nbinsx=6))
# add labels
fig.update_layout(title=f'Layer {layer_idx} - Projection along the instruction direction')
fig.update_xaxes(title='Projection')
fig.update_yaxes(title='Count')
fig.show()

# get average projection along the instruction direction for each layer
avg_proj = proj.mean()
print(f'Average projection along the instruction direction for layer {layer_idx}: {avg_proj}')

# %%
# =============================================================================
# Steering
# =============================================================================

layer_idx = 24

instr_dir = last_token_mean_diff[layer_idx] / last_token_mean_diff[layer_idx].norm()

intervention_dir = instr_dir.to(device)
intervention_layers = list(range(layer_idx, layer_idx+1)) # only one layer

hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir, weight=100)
#hook_fn = functools.partial(direction_projection_hook,direction=intervention_dir, value_along_direction=avg_proj)
fwd_hooks = [(tlutils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_post']]


# %%
base_q = 'What is an interesting place in the city of New York?'
instr = ' Include the word "station".'
#instr = ''
# base_q = "Rewrite the following statement to make it sound more formal, like a President of the United States:\"Hi guys. The work was done to add in a fix for the issue that was observed in the field with the SSO. We are working with our collaborators closely. We will get it done. Thanks ya all.\""
#base_q = "What is a typical issue related to macbooks?"
#base_q = "What is the most famous issue of hydrogen cars"
#base_q = "What is Grand Central?"
example = base_q + instr

messages = [{"role": "user", "content": example}]
input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
output_toks = generate_with_hooks(model, input, fwd_hooks=fwd_hooks, max_tokens_generated=64, verbose=True)
output_str = model.tokenizer.batch_decode(output_toks[:, input.shape[1]:], skip_special_tokens=True)
print(output_str[0])
# %%

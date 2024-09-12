# %%
import os
import sys
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
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

# %%
# Some environment variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transformer_cache_dir = None
# %%
# =============================================================================
# Load and inspect the data
# =============================================================================

model_name = 'gemma-2-9b-it'
folder = f'length_constraints/representations/{model_name}'
n_examples = 50
n_sent_max = 5

file = f'{folder}/high_level_{n_examples}examples_hs.h5'
results_df = pd.read_hdf(file, key='df')

# filter the df keeping the rows for which kwargs == [{'language': 'kn'}]
#results_df = results_df[results_df['kwargs'].apply(lambda x: x[0] =={'language': 'kn'})]

results_df['prompt'] = results_df['prompt_with_constraint']

# sort results_df by length_constraint
results_df = results_df.sort_values(by='length_constraint')
# %%
for i, row in results_df.iterrows():
    print(row['prompt'])
    print(f'output: {row["output"]}')
    print(f'output_no_instr: {row["output_no_instr"]}')
    print('-------')

# %%
max_length = min([x.shape[1] for x in results_df['last_token_rs_no_instr'].values])
hs_instr = results_df['last_token_rs'].values
hs_instr = torch.tensor([example_array[:, :max_length] for example_array in list(hs_instr)])
hs_no_instr = results_df['last_token_rs_no_instr'].values
hs_no_instr = torch.tensor([example_array[:, :max_length] for example_array in list(hs_no_instr)])

print(f'hs_instr shape: {hs_instr.shape}')
print(f'hs_no_instr shape: {hs_no_instr.shape}')

# print the size in MB of the tensors
print(f'hs_instr size: { hs_instr.element_size() * hs_instr.nelement() / 1024 / 1024 } MB')
# %%
cos_sims = torch.zeros((hs_instr.shape[1], hs_instr.shape[0] * 2, hs_instr.shape[0] * 2, hs_instr.shape[2]))
p_bar = tqdm.tqdm(total=hs_instr.shape[1])
for layer_idx in range(hs_instr.shape[1]):
    # compute cosine similarity between representations for different examples
    concat = torch.cat([hs_instr[:, layer_idx, :, :], hs_no_instr[:, layer_idx, :, :]], dim=0)
    sim = torch.nn.functional.cosine_similarity(concat.unsqueeze(1), concat.unsqueeze(0), dim=-1)
    cos_sims[layer_idx] = sim
    p_bar.update(1)
cos_sims = cos_sims.permute(0, 3, 2, 1)
# %%
layer_idx = 16
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
fig.add_trace(go.Scatter(x=x_labels, y=avg_cos_sims[:, token_idx].cpu().numpy(), mode='lines', name='Sim between same ex. w/ and w/o instr'))
# add labels
fig.update_layout(title=f'Token {token_idx} - Average cosine similarity between representations')
fig.update_xaxes(title='Layer idx')
fig.update_yaxes(title='Cosine similarity')
fig.show()
# %%
# =============================================================================
# compute the diff between the representations
# =============================================================================

repr_diffs = hs_instr - hs_no_instr
mean_repr_diffs = repr_diffs.mean(dim=0)
last_token_mean_diff = mean_repr_diffs[:, -1, :]

# %%
# compute cosine similarity between the instruction vectors
layer_idx = 16
cos_sim_matrix = repr_diffs[:, layer_idx, -1] @ repr_diffs[:, layer_idx, -1].transpose(0, 1) / (repr_diffs[:, layer_idx, -1].norm(dim=-1).unsqueeze(1) * repr_diffs[:, layer_idx, -1].norm(dim=-1))



# make a heatmap of the cosine similarity matrix
fig = px.imshow(cos_sim_matrix.cpu().numpy(), labels=dict(x='Example idx', y='Example idx', color='Cosine similarity'))
fig.update_layout(title=f'Layer {layer_idx} - Cosine similarity between instruction vectors')
# add labels
fig.update_xaxes(title='Example idx')
fig.update_yaxes(title='Example idx')
fig.show()

# %%
length_specific_representations = torch.zeros((results_df['length_constraint'].max()+1, hs_instr.shape[-1]))
for i in range(results_df['length_constraint'].max()+1):
    length_specific_rep = repr_diffs[n_examples*i:n_examples*(i+1), layer_idx, -1].mean(dim=0)
    length_specific_representations[i] = length_specific_rep

# %%
# compute cosine similarity between the length specific representations
cos_sim_matrix = torch.zeros((results_df['length_constraint'].max()+1, results_df['length_constraint'].max()+1))
for i in range(results_df['length_constraint'].max()+1):
    for j in range(results_df['length_constraint'].max()+1):
        cos_sim_matrix[i, j] = length_specific_representations[i] @ length_specific_representations[j] / (length_specific_representations[i].norm() * length_specific_representations[j].norm())

# make a heatmap of the cosine similarity matrix
fig = px.imshow(cos_sim_matrix.cpu().numpy(), labels=dict(x='Length constraint', y='Length constraint', color='Cosine similarity'))
fig.update_layout(title=f'Layer {layer_idx} - Cosine similarity between length specific representations')
# add labels
fig.update_xaxes(title='Length constraint')
fig.update_yaxes(title='Length constraint')
fig.show()

# %%
# =============================================================================
# Load the model and inspect the projections
# =============================================================================

device = torch.device(1 if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('hf_token.txt') as f:
    hf_token = f.read()
transformer_cache_dir = None
model, tokenizer = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token)
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
fig.add_trace(go.Histogram(x=proj.cpu().numpy(), name='instr', nbinsx=20, opacity=0.75))
fig.add_trace(go.Histogram(x=proj_no_instr.cpu().numpy(), name='no_instr', nbinsx=20, opacity=0.75))
# overlay both histograms
fig.update_layout(barmode='overlay')
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
# projection along the length specific representations
# =============================================================================

# take the average of the representations 
length_constraints_range = (0, 1)

length_constraint_direction = length_specific_representations[length_constraints_range[0]:length_constraints_range[1]].mean(dim=0) / length_specific_representations[length_constraints_range[0]:length_constraints_range[1]].mean(dim=0).norm()
length_constraint_direction.to(device)

# average projection along the length_constraint_direction
projections = []
for lenght_constraint in range(length_specific_representations.shape[0]):
    proj = hs_instr[n_examples*lenght_constraint:n_examples*(lenght_constraint+1), layer_idx, -1, :] @ length_constraint_direction
    print(f'Layer {layer_idx} - Length constraint {lenght_constraint}: {proj}')
    projections.append(proj)

# make histograms
fig = go.Figure()
for i, proj in enumerate(projections):
    fig.add_trace(go.Histogram(x=proj.cpu().numpy(), name=f'Length constraint {i+1}', opacity=0.75, nbinsx=30))
# overlay both histograms
fig.update_layout(barmode='overlay')
# add labels
fig.update_layout(title=f'Layer {layer_idx} - Projection along the length specific representations')
fig.update_xaxes(title='Projection')
fig.update_yaxes(title='Count')
fig.show()

# %%
# make box plots for each projection
fig = go.Figure()
for i, proj in enumerate(projections):
    fig.add_trace(go.Box(y=proj.cpu().numpy(), name=f'Length constraint={i+1}'))
# add labels
fig.update_layout(title=f'Layer {layer_idx} - Projection along the average representation gor lengths {length_constraints_range[0]+1}-{length_constraints_range[1]}')
fig.update_xaxes(title='Length constraint')
fig.update_yaxes(title='Projection')
fig.show()


# %%
# =============================================================================
# steering the generation
# =============================================================================

from utils.generation_utils import generate_with_hooks

# def generate_with_hooks(
#     model,
#     toks,
#     max_tokens_generated: int = 64,
#     fwd_hooks = [],
#     verbose=False
# ):

#     all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
#     all_toks[:, :toks.shape[1]] = toks

#     if verbose:
#         pbar = tqdm.tqdm(total=max_tokens_generated, desc='Generating tokens')

#     with torch.no_grad():
#         for i in range(max_tokens_generated):
#             with model.hooks(fwd_hooks=fwd_hooks):
#                 logits = model(all_toks[:, :-max_tokens_generated + i])
#                 next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
#                 if next_tokens[0] == model.tokenizer.eos_token_id or next_tokens[0] == 32007:
#                     break
#                 if next_tokens[0] == 235292 and all_toks[0, -max_tokens_generated+i-1] == 235368:
#                     print(f'Stopping the generation as the model is generating a new question (Q:)')
#                     # remove the Q
#                     all_toks[0, -max_tokens_generated+i-1] = 0
#                     break
#                 all_toks[:,-max_tokens_generated+i] = next_tokens
#             if verbose:
#                 pbar.update(1)

#     return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def direction_ablation_hook(
    activation,
    hook,
    direction,
    weight=1.0,
):
    return activation + (direction * weight)

def direction_projection_hook(
    activation,
    hook,
    direction,
    value_along_direction,
):
    adjusted_activations = adjust_vectors(activation.squeeze(), direction, value_along_direction)
    return adjusted_activations.unsqueeze(0)

# %%
layer_idx = 16

# take the average of the representations 
# length_constraints_range = (4, 5)

length_constraint_direction = length_specific_representations[0] / length_specific_representations[0].norm()
length_constraint_direction.to(device)


intervention_dir = length_constraint_direction.to(device)
#intervention_layers = list(range(model.cfg.n_layers)) # all layers
intervention_layers = list(range(layer_idx, layer_idx+1)) # only one layer
#intervention_layers = list(range(0, layer_idx+1))

weight = 0

hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir, weight=weight)
#hook_fn = functools.partial(direction_projection_hook,direction=intervention_dir, value_along_direction=avg_proj)
fwd_hooks = [(tlutils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_post']]

# %%
input = "Can you help me make an advertisement for a new product? It's a diaper that's designed to be more comfortable for babies."
#input = 'Write a rubric for performance review of a software engineer.'
#input = 'List all facts about Lionel Messi.'
#input = 'Compose a startup pitch on a new app called Tipperary that helps people to find the average tip size for each restaurant.'
input = 'Write a poem about inflation.'
input = "What is prospect park?"
input='Q: Write a story about a family that goes camping in the woods.\nA:'
input='Q: Is Pikachu one of the Avengers? Think out loud, then answer.\nA:'
# input='Write a product description about a new, innovative, toothbrush.'
# input='What\'s the story of Myanmarese refugees in Bangladesh?'
#input = tokenizer(input, return_tensors='pt')['input_ids'].to(device)
# input = f'<|user|>\n{input}<|end|>\n<|assistant|>'
messages = [{"role": "user", "content": input}]
# input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
input = tokenizer(input, return_tensors='pt')['input_ids'].to(device)
output_int = generate_with_hooks(model, input, fwd_hooks=fwd_hooks, max_tokens_generated=32, verbose=True, decode_directly=True)
print(output_int[0])
print(len(tokenizer(output_int[0])['input_ids']))
# %%
# =============================================================================
# compare representation to legnth of generation
# =============================================================================

lenght_of_outputs = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
lenght_of_outputs_char = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
lenght_of_outputs_sent = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
proj_values = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
for i, row in results_df.iterrows():
    lenght_of_outputs[row['length_constraint']].append(len(row['output'].split()))
    lenght_of_outputs_char[row['length_constraint']].append(len(row['output']))
    lenght_of_outputs_sent[row['length_constraint']].append(len(nltk.sent_tokenize(row['output'])))
    proj_value = torch.tensor(row['last_token_rs'][layer_idx, -1], device=device) @ length_constraint_direction
    proj_values[row['length_constraint']].append(proj_value)
    
# %%
# make histogram, one for each length constraint
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs_sent.items():
    fig.add_trace(go.Histogram(x=lengths, name=f'Length constraint {length_constraint}', histnorm='probability density', nbinsx=20))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.update_layout(title='Length of generated outputs')
fig.update_xaxes(title='# of Sentences')
fig.update_yaxes(title='Probability density')
fig.show()

# %%
# make histogram for the projection values, one for each length constraint
fig = go.Figure()
for length_constraint, proj_vals in proj_values.items():
    fig.add_trace(go.Histogram(x=proj_vals, name=f'Length constraint {length_constraint}', histnorm='probability density', nbinsx=20))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.update_layout(title='Projection values')
fig.update_xaxes(title='Projection value')
fig.update_yaxes(title='Probability density')
fig.show()
# %%
# make scatter plot of projection values vs length of generated outputs
fig = go.Figure()
for length_constraint, proj_vals in proj_values.items():
    fig.add_trace(go.Scatter(x=lenght_of_outputs_sent[length_constraint], y=proj_vals, mode='markers', name=f'Length constraint {length_constraint}'))
fig.update_layout(title='Projection values vs Length of generated outputs')
fig.update_xaxes(title='# of Sentences')
fig.update_yaxes(title='Projection value')
fig.show()
# %%
# empty cuda cache
torch.cuda.empty_cache()
# %%
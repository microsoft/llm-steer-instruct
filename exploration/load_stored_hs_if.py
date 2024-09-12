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
model_name = 'phi-3'
model_name = 'mistral-7b-instruct'
model_name= 'gemma-2-9b-it'
#folder = f'stored_hs/if/{model_name}'
folder = f'./ifeval_experiments/representations/{model_name}/single_instr_all_base_x_all_instr'
instruct_type = ['detectable_format:json_format']
# instruct_type = ['change_case:english_lowercase']
# instruct_type = ['language:response_language_bn']
# instruct_type = ['detectable_format:number_highlighted_sections']
#instruct_type = ['keywords:frequency_at least']
# instruct_type = ['startend:quotation']
#results_df = pd.read_hdf(f'{folder}/{"".join(instruct_type).replace(":", "_")}_{n_examples}examples_hs_new.h5', key='df')
results_df = pd.read_hdf(f'{folder}/{"".join(instruct_type).replace(":", "_")}.h5', key='df')

# filter the df keeping the rows for which kwargs == [{'language': 'kn'}]
#results_df = results_df[results_df['kwargs'].apply(lambda x: x[0] =={'language': 'kn'})]

n_examples = results_df.shape[0] 

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
layer_idx = 18
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
device = torch.device(1 if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('hf_token.txt') as f:
    hf_token = f.read()
transformer_cache_dir = None
model, tokenizer = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token)
# model_hf, tokenizer_hf = load_model_from_tl_name(model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token, hf_model=True)
# model_hf.to(device)
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

layer_idx = 26

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
# steering the generation
# =============================================================================

def generate_with_hooks(
    model,
    toks,
    max_tokens_generated: int = 64,
    fwd_hooks = [],
):

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    with torch.no_grad():
        for i in range(max_tokens_generated):
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :-max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
                print(f'prev : {all_toks[0, -max_tokens_generated+i-1]} - next_tokens: {next_tokens[0]} -> {model.tokenizer.decode(next_tokens[0])}') 
                if next_tokens[0] == model.tokenizer.eos_token_id or next_tokens[0] == 32007:
                    print(f'breaking at i={i}, token={next_tokens[0]}')
                    break
                if next_tokens[0] == 235292 and all_toks[0, -max_tokens_generated+i-1] == 235368:
                    print(f'The model is generating a new question (Q:)')
                    break
                all_toks[:,-max_tokens_generated+i] = next_tokens

    # truncate the tensor to remove padding
    all_toks = all_toks[:, :toks.shape[1] + i]

    return all_toks

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
    #print(' value along direction before hook:', activation.squeeze() @ direction)
    adjusted_activations = adjust_vectors(activation.squeeze(), direction, value_along_direction)
    #print(' value along direction after hook:', adjusted_activations.squeeze() @ direction)
    return adjusted_activations.unsqueeze(0)

# %%

intervention_dir = instr_dir.to(device)
#intervention_layers = list(range(model.cfg.n_layers)) # all layers
intervention_layers = list(range(layer_idx, layer_idx+1)) # only one layer
#intervention_layers = list(range(0, layer_idx+1))

hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir, weight=90)
hook_fn = functools.partial(direction_projection_hook,direction=intervention_dir, value_along_direction=avg_proj)
fwd_hooks = [(tlutils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_post']]

# %%
input = "Q: Can you help me make an advertisement for a new product? It's a diaper that's designed to be more comfortable for babies.\nA:"
# input = 'Q: Write a rubric for performance review of a software engineer.\nA:'
input = 'Q: List some facts about Lionel Messi.\nA:' # The answer should be in all lowercase.'
#input = 'Compose a startup pitch on a new app called Tipperary that helps people to find the average tip size for each restaurant.'
# input = "Q: What is the history of NYC prospect park?\nA:"
input='Q: Write a story about a family that goes camping in the woods.\nA:'
# input='Q: Write a product description about a new, innovative, toothbrush.\nA:'
# input='Q: What\'s the story of Myanmarese refugees in Bangladesh?\nA:'
# input = 'Q: Write facts about machine learning.\nA:'
#input = tokenizer(input, return_tensors='pt')['input_ids'].to(device)
messages = [{"role": "user", "content": input}]
#input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
input = tokenizer(input, return_tensors='pt')['input_ids'].to(device)
output_toks = generate_with_hooks(model, input, fwd_hooks=[], max_tokens_generated=128)
#output_toks = model.generate(**input, max_length=1024, do_sample=False)
output_str = model.tokenizer.batch_decode(output_toks[:, input.shape[1]:], skip_special_tokens=True)
print(output_str[0])
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

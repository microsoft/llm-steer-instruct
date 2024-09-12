# %%
import pandas as pd
import numpy as np
import os
import torch
import plotly.express as px
import sys
import plotly.graph_objects as go
from transformers import AutoTokenizer
# %%

if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
    print('We\'re on a Windows machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on the server')
# %%
model_name = 'phi-3'
n_examples = 100
folder = f'stored_hs/{model_name}'

df_cot = pd.read_hdf(f'{folder}/cot_{n_examples}examples_hs.h5', key='df')
df_direct = pd.read_hdf(f'{folder}/direct_{n_examples}examples_hs.h5', key='df')

# join the two dataframes on column "example_idx"
df_cot = df_cot.set_index('example_idx')
df_direct = df_direct.set_index('example_idx')
results_df = df_cot.join(df_direct, lsuffix='', rsuffix='_no_instr')
# %%
# compute accuracy
correct_cot = (df_cot['gt_answer'] == df_cot['prediction_extracted']).sum()
print(f'Accuracy: {correct_cot}/{len(df_cot)} = {correct_cot/len(df_cot)}')

correct_direct = (df_direct['gt_answer'] == df_direct['prediction_extracted']).sum()
print(f'Accuracy: {correct_direct}/{len(df_direct)} = {correct_direct/len(df_direct)}')
# %%
hs_last_token_cot = df_cot['last_token_rs'].values
hs_last_token_cot = torch.tensor(list(hs_last_token_cot))
hs_last_token_direct = df_direct['last_token_rs'].values
hs_last_token_direct = torch.tensor(list(hs_last_token_direct))

# %%
# print the size in MB of the tensors
print(f'hs_last_token_cot size: { hs_last_token_cot.element_size() * hs_last_token_cot.nelement() / 1024 / 1024 } MB')

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
for layer_idx in range(hs_instr.shape[1]):
    # compute cosine similarity between representations for different examples
    concat = torch.cat([hs_instr[:, layer_idx, :, :], hs_no_instr[:, layer_idx, :, :]], dim=0)
    sim = torch.nn.functional.cosine_similarity(concat.unsqueeze(1), concat.unsqueeze(0), dim=-1)
    cos_sims[layer_idx] = sim
cos_sims = cos_sims.permute(0, 3, 2, 1)
# %%
layer_idx = 21
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
fig.add_trace(go.Scatter(x=x_labels, y=avg_cos_sims[:, token_idx].cpu().numpy(), mode='lines', name='avg_cos_sim'))
fig.add_trace(go.Scatter(x=x_labels, y=baseline_sims_instr[:, token_idx].cpu().numpy(), mode='lines', name='baseline_sims_instr'))
fig.add_trace(go.Scatter(x=x_labels, y=baseline_sims_no_instr[:, token_idx].cpu().numpy(), mode='lines', name='baseline_sims_no_instr'))
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

layer_idx = 20

instr_dir = last_token_mean_diff[layer_idx] / last_token_mean_diff[layer_idx].norm()

# average projection along the instruction direction
proj = hs_instr[:, layer_idx, -1, :].to(device) @ instr_dir.to(device)
proj_no_instr = hs_no_instr[:, layer_idx, -1, :].to(device) @ instr_dir.to(device)

# make two overlayed histograms
fig = go.Figure()
fig.add_trace(go.Histogram(x=proj.cpu().numpy(), name='instr'))
fig.add_trace(go.Histogram(x=proj_no_instr.cpu().numpy(), name='no_instr'))
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
                all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def direction_ablation_hook(
    activation,
    hook,
    direction,
):
    return activation - (direction * 1)

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
intervention_layers = list(range(model.cfg.n_layers)) # all layers
#intervention_layers = list(range(layer_idx, layer_idx+1)) # only one layer
#intervention_layers = list(range(0, layer_idx+1))

hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir)
#hook_fn = functools.partial(direction_projection_hook,direction=intervention_dir, value_along_direction=avg_proj)
fwd_hooks = [(tlutils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_post']]

# %%
input = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
#input='Write a story about a family that goes camping in the woods.'
#input = tokenizer(input, return_tensors='pt')['input_ids'].to(device)
messages = [{"role": "user", "content": input}]
input = tokenizer.apply_chat_template(messages, return_tensors='pt').to(device)
output_int = generate_with_hooks(model, input, fwd_hooks=fwd_hooks, max_tokens_generated=128)
print(output_int[0])
# %%
0
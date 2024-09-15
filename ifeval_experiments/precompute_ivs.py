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
# %%
import pandas as pd
import numpy as np
import os
import torch
import plotly.express as px
import sys
import plotly.graph_objects as go
from tqdm import tqdm
import json

# %%

# read in ../data/input_data.jsonl
with open('./data/input_data_single_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
input_data_df = pd.DataFrame(data)
all_instructions = input_data_df['instruction_id_list_for_eval'].apply(lambda x: x[0]).unique()
len(all_instructions)


# %%
model_name = 'gemma-2-9b'
dry_run = False
device = 'cpu'
specific_layer = None
search_method = 'validation_accuracy_no_instr'
seed=42
n_examples = 6

rows = []

if 'validation_accuracy' in search_method:

    if 'no_instr' in search_method:
        instr_included = 'no_instr'
    else:
        instr_included = 'instr'
    folder = 'ifeval_experiments/layer_search_out'
    file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}/out_{instr_included}.jsonl'
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    validation_df = pd.DataFrame(results)
    optimal_layers = { instr: -1 for instr in all_instructions }
                
    for instr in all_instructions:
        if instr not in validation_df.single_instruction_id.unique():
            optimal_layers[instr] = -1
            continue

        instr_df = validation_df[validation_df.single_instruction_id == instr]
        max_accuracy = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions.max()
        optimal_layer = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean()[instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions == max_accuracy].index
        optimal_layers[instr] = optimal_layer[0]

for instr in tqdm(all_instructions):
    # check if the file exists
    folder = f'./ifeval_experiments/representations/{model_name}/single_instr_all_base_x_all_instr'

    file =f'{folder}/{"".join(instr).replace(":", "_")}.h5'
    if not os.path.exists(file):
        print(f'File {file} does not exist')
        continue
    results_df = pd.read_hdf(file, key='df')
    if dry_run:
        results_df = results_df.head(10)
    n_examples = results_df.shape[0] 

    row = {}
    row['instruction'] = instr

    # max_length = min([x.shape[1] for x in results_df['last_token_rs_no_instr'].values])
    hs_instr = results_df['last_token_rs'].values
    hs_instr = torch.tensor(np.array([example_array[:, :] for example_array in list(hs_instr)]), device=device)
    hs_no_instr = results_df['last_token_rs_no_instr'].values
    hs_no_instr = torch.tensor(np.array([example_array[:, :] for example_array in list(hs_no_instr)]), device=device)

    # check if hs has 4 dimensions
    if len(hs_instr.shape) == 3:
        hs_instr = hs_instr.unsqueeze(2)
        hs_no_instr = hs_no_instr.unsqueeze(2)

    if specific_layer is not None:
        max_diff_layer_idx = specific_layer

    else:
        if search_method == 'cosine_similarity':
            cos_sims = torch.zeros((hs_instr.shape[1], hs_instr.shape[0] * 2, hs_instr.shape[0] * 2, hs_instr.shape[2]), device=device)
            p_bar = tqdm(total=hs_instr.shape[1], disable=True)
            for layer_idx in range(hs_instr.shape[1]):
                # compute cosine similarity between representations for different examples
                concat = torch.cat([hs_instr[:, layer_idx, :, :], hs_no_instr[:, layer_idx, :, :]], dim=0)
                sim = torch.nn.functional.cosine_similarity(concat.unsqueeze(1), concat.unsqueeze(0), dim=-1)
                cos_sims[layer_idx] = sim
                p_bar.update(1)
            cos_sims = cos_sims.permute(0, 3, 2, 1)

            # compute the average cosine sim along the diagonal of cos_sims
            avg_cos_sims = torch.zeros((hs_instr.shape[1], hs_instr.shape[2]))
            baseline_sims_instr = torch.zeros((hs_instr.shape[1], hs_instr.shape[2]))
            baseline_sims_no_instr = torch.zeros((hs_instr.shape[1], hs_instr.shape[2]))
            for layer_idx in range(hs_instr.shape[1]):
                for token_idx in range(hs_instr.shape[2]):
                    avg_cos_sims[layer_idx, token_idx] = torch.diagonal(cos_sims[layer_idx, token_idx], offset=-n_examples).mean()
                    baseline_sims_instr[layer_idx, token_idx] = cos_sims[layer_idx, token_idx, :n_examples, :n_examples].mean()
                    baseline_sims_no_instr[layer_idx, token_idx] = cos_sims[layer_idx, token_idx, n_examples+1:, n_examples+1:].mean()

            last_tok_avg_cos_sims = avg_cos_sims[:, -1]
            last_tok_baseline_sims_instr = baseline_sims_instr[:, -1]
            last_tok_baseline_sims_no_instr = baseline_sims_no_instr[:, -1]

            diff = last_tok_baseline_sims_instr - last_tok_baseline_sims_no_instr

            # compute index of the layer with the highest diff
            max_diff_layer_idx = diff.argmax().item()
        elif 'validation_accuracy' in search_method:
            max_diff_layer_idx = optimal_layers[instr]
            
        else:
            print('Invalid search method')
            sys.exit(1)

    repr_diffs = hs_instr - hs_no_instr
    mean_repr_diffs = repr_diffs.mean(dim=0)
    last_token_mean_diff = mean_repr_diffs[:, -1, :]

    instr_dir = last_token_mean_diff[max_diff_layer_idx] / last_token_mean_diff[max_diff_layer_idx].norm()

    # average projection along the instruction direction
    proj = hs_instr[:, max_diff_layer_idx, -1, :].to(device) @ instr_dir.to(device)
    proj_no_instr = hs_no_instr[:, max_diff_layer_idx, -1, :].to(device) @ instr_dir.to(device)

    # get average projection along the instruction direction for each layer
    avg_proj = proj.mean()
    avg_proj_no_instr = proj_no_instr.mean()
    
    if max_diff_layer_idx == -1:
        row['max_diff_layer_idx'] = -1
        row['instr_dir'] = torch.zeros(hs_instr.shape[-1]).cpu().numpy()
        row['avg_proj'] = 0
        row['avg_proj_no_instr'] = 0

        print(f'Instruction {instr} is better off without any steering')
    else:
        row['max_diff_layer_idx'] = max_diff_layer_idx
        row['instr_dir'] = instr_dir.cpu().numpy()
        row['avg_proj'] = avg_proj
        row['avg_proj_no_instr'] = avg_proj_no_instr

        print(f'Average projection along the {instr} direction for layer {max_diff_layer_idx}: {avg_proj} - {avg_proj_no_instr}')

    rows.append(row)

# %%
df = pd.DataFrame(rows)

# %%
# store the df in folder + '/pre_computed_ivs_best_layer.h5'
if specific_layer is not None:
    df.to_hdf(f'{folder}/pre_computed_ivs_layer_{specific_layer}.h5', key='df', mode='w')
elif 'validation_accuracy' in search_method:
    df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_validation_{instr_included}.h5', key='df', mode='w')
elif search_method == 'cosine_similarity':
    df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_cosine_similarity.h5', key='df', mode='w')
else:
    raise ValueError('Invalid search method')
# %%

# load the df from folder + '/pre_computed_ivs_best_layer.h5'
# df = pd.read_hdf(f'{folder}/pre_computed_ivs_best_layer.h5', key='df')

# %%

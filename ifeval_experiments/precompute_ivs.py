# %%
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
os.chdir(project_dir)

import sys
sys.path.append(script_dir)
sys.path.append(project_dir)

import json
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

# %%
model_names = ['phi-3', 'mistral-7b-instruct', 'gemma-2-2b-it', 'gemma-2-9b-it']
dry_run = False
device = 'cpu'
specific_layer = None
include_instr_strings = ['_no_instr', '']
cross_model_strings = ['_cross_model', '']
seed=42
n_examples_dict = {'gemma-2-9b' : 6 , 'gemma-2-9b-it' : 6, 'gemma-2-2b' : 10, 'gemma-2-2b-it' : 8}
preplexity_threshold = 2.5


# read in ./data/input_data.jsonl
with open(f'{project_dir}/data/input_data_single_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
input_data_df = pd.DataFrame(data)
all_instructions = list(input_data_df['instruction_id_list_for_eval'].apply(lambda x: x[0]).unique())
len(all_instructions)


for model_name in model_names:
    for include_instr_string in include_instr_strings:
        for cross_model_string in cross_model_strings:

            search_method = f'validation_accuracy_w_perplexity_{cross_model_string}{include_instr_string}'

            n_examples = n_examples_dict[model_name]
            print(n_examples)
                    
            # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
            filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
            all_instructions = list(filter(lambda x: any([f in x for f in filters]), all_instructions))

            w_perplexity = '_with_perplexity' if 'perplexity' in search_method else ''
            cross_model = '_cross_model' if 'cross_model' in search_method else ''
            instr_included = 'instr' if 'instr' in search_method else 'no_instr'

            folder = f'{script_dir}/layer_search_out'
            file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}{cross_model}{w_perplexity}/out_{instr_included}.jsonl'
            with open(file, 'r') as f:
                results = [json.loads(line) for line in f]

            validation_df = pd.DataFrame(results)
            optimal_layers = { instr: -1 for instr in all_instructions }
        
            for instr in all_instructions:
                if instr not in validation_df.single_instruction_id.unique():
                    optimal_layers[instr] = -1
                    continue

                instr_df = validation_df[validation_df.single_instruction_id == instr]
                
                if 'perplexity' in search_method:
                    # add boolean column that is true when perplexity is low
                    instr_df['low_perplexity'] = instr_df.perplexity < preplexity_threshold

                    df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()

                    if model_name == 'gemma-2-9b' or model_name == 'gemma-2-2b':
                        baseline_low_perplexity = df_group_by_layer.loc[-1, 'low_perplexity']
                    else:
                        baseline_low_perplexity = 0

                    # get accuracy for layer -1
                    accuracy_layer_minus_1 = df_group_by_layer.loc[-1, 'follow_all_instructions']

                    df_group_by_layer.loc[df_group_by_layer.low_perplexity > baseline_low_perplexity, 'follow_all_instructions'] = 0

                    # restore accuracy for layer -1
                    df_group_by_layer.loc[-1, 'follow_all_instructions'] = accuracy_layer_minus_1

                    df_group_by_layer.loc[df_group_by_layer.low_perplexity > baseline_low_perplexity, 'follow_all_instructions'] = 0
                    max_accuracy = df_group_by_layer.follow_all_instructions.max()
                    optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
                    optimal_layers[instr] = optimal_layer[0]

                else:
                    max_accuracy = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions.max()
                    optimal_layer = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean()[instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions == max_accuracy].index
                    optimal_layers[instr] = optimal_layer[0]

            rows = []

            for instr in tqdm(all_instructions):
                # check if the file exists
                if model_name == 'gemma-2-2b' and 'cross_model' in search_method:
                    print('Using representations from gemma-2-2b-it')
                    rep_folder = f'{script_dir}/representations/gemma-2-2b-it/single_instr_all_base_x_all_instr'
                elif model_name == 'gemma-2-9b' and 'cross_model' in search_method:
                    print('Using representations from gemma-2-9b-it')
                    rep_folder = f'{script_dir}/representations/gemma-2-9b-it/single_instr_all_base_x_all_instr'
                else:
                    print(f'Using representations from {model_name}')
                    rep_folder = f'{script_dir}/representations/{model_name}/single_instr_all_base_x_all_instr'

                file =f'{rep_folder}/{"".join(instr).replace(":", "_")}.h5'
                
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
                    selected_layer = specific_layer
                else:
                    selected_layer = optimal_layers[instr]
                    
                repr_diffs = hs_instr - hs_no_instr
                mean_repr_diffs = repr_diffs.mean(dim=0)
                last_token_mean_diff = mean_repr_diffs[:, -1, :]

                instr_dir = last_token_mean_diff[selected_layer] / last_token_mean_diff[selected_layer].norm()

                # average projection along the instruction direction
                proj = hs_instr[:, selected_layer, -1, :].to(device) @ instr_dir.to(device)
                proj_no_instr = hs_no_instr[:, selected_layer, -1, :].to(device) @ instr_dir.to(device)

                # get average projection along the instruction direction for each layer
                avg_proj = proj.mean()
                avg_proj_no_instr = proj_no_instr.mean()
                
                if selected_layer == -1:
                    row['selected_layer'] = -1
                    row['instr_dir'] = torch.zeros(hs_instr.shape[-1]).cpu().numpy()
                    row['avg_proj'] = 0
                    row['avg_proj_no_instr'] = 0

                    print(f'Instruction {instr} is better off without any steering')
                else:
                    row['selected_layer'] = selected_layer
                    row['instr_dir'] = instr_dir.cpu().numpy()
                    row['avg_proj'] = avg_proj
                    row['avg_proj_no_instr'] = avg_proj_no_instr

                    print(f'Average projection along the {instr} direction for layer {selected_layer}: {avg_proj} - {avg_proj_no_instr}')

                rows.append(row)

            df = pd.DataFrame(rows)

            # store the df in folder + '/pre_computed_ivs_best_layer.h5'
            folder = f'{script_dir}/representations/{model_name}/single_instr_all_base_x_all_instr'
            if specific_layer is not None:
                df.to_hdf(f'{folder}/pre_computed_ivs_layer_{specific_layer}.h5', key='df', mode='w')
            elif 'perplexity' in search_method:
                df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_validation_perplexity{cross_model}_{instr_included}.h5', key='df', mode='w')
            else:
                df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_validation_{instr_included}.h5', key='df', mode='w')
            
# %%


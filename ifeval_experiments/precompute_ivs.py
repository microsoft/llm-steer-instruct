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
from transformers import AutoTokenizer
from collections import Counter
from eval.evaluation_main import test_instruction_following_loose
import re

# %%

# read in ../data/input_data.jsonl
with open('./data/input_data_single_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
input_data_df = pd.DataFrame(data)
all_instructions = list(input_data_df['instruction_id_list_for_eval'].apply(lambda x: x[0]).unique())
len(all_instructions)


# %%
model_name = 'gemma-2-9b-it'
model_name = 'gemma-2-2b'
# model_name = 'phi-3'
# model_name = 'mistral-7b-instruct'
dry_run = False
device = 'cpu'
specific_layer = None
search_method = 'validation_accuracy_w_perplexity'
seed=42
n_examples = 10

nonparametric_only = True

if nonparametric_only:
    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    all_instructions = list(filter(lambda x: any([f in x for f in filters]), all_instructions))

print(all_instructions)


rows = []

if 'validation_accuracy' in search_method:

    if 'perplexity' in search_method:
        w_perplexity = '_with_perplexity'
    else:
        w_perplexity = ''

    if 'no_instr' in search_method:
        instr_included = 'no_instr'
    else:
        instr_included = 'instr'
    print(f'INSTR: {instr_included}')
    folder = 'ifeval_experiments/layer_search_out'
    file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}{w_perplexity}/out_{instr_included}.jsonl'
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    validation_df = pd.DataFrame(results)
    optimal_layers = { instr: -1 for instr in all_instructions }

    if 'quality_check' in search_method:

        with open('./hf_token.txt') as f:
            hf_token = f.read()
        if model_name == 'phi-3':
            model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'
        elif 'gemma' in model_name:
            model_name_hf = f'google/{model_name}'
        elif model_name == 'mistral-7b-instruct':
            model_name_hf = 'mistralai/Mistral-7B-Instruct-v0.1'
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, token=hf_token)

        broken_outputs = []
        accuracy_with_quality_check = []
        for i, r in validation_df.iterrows():
            # compute accuracy

            response  = r['response']
            tokens = tokenizer.tokenize(response)
            counter = Counter(tokens)
            #remove '▁the' '▁' from the counter
            if '▁the' in counter:
                del counter['▁the']
            if '▁' in counter:
                del counter['▁']
            # take the number of occurrences of the most common token
            most_common = counter.most_common(1)[0][1]
            # get most common token
            if most_common > 50:
                broken_outputs.append(1)
            else:
                broken_outputs.append(0)

            # fix problem with capital word frequency
            if r.single_instruction_id == 'change_case:capital_word_frequency' and r.layer == -1:
                if 'less than' in r.prompt or 'at most' in r.prompt:
                    relation = 'less than'
                elif 'more than' in r.prompt or 'at least' in r.prompt:
                    relation = 'at least'
                # parse the last number in the prompt
                num = int(re.findall(r'\d+', r.prompt)[-1])

                new_kwargs = [{"capital_relation": relation,"capital_frequency": num}]
                r.kwargs = new_kwargs
                prompt_to_response = {}
                prompt_to_response[r['prompt']] = r['response']
                output = test_instruction_following_loose(r, prompt_to_response)
                validation_df.loc[i, 'follow_all_instructions'] = output.follow_all_instructions

        validation_df['broken_output'] = broken_outputs

        
    for instr in all_instructions:
        if instr not in validation_df.single_instruction_id.unique():
            optimal_layers[instr] = -1
            continue

        instr_df = validation_df[validation_df.single_instruction_id == instr]

        if 'quality_check' in search_method:
            df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'broken_output']].groupby('layer').mean()
            # set follow_all_instructions to in df_group_by_layer when broken_output is > 0
            df_group_by_layer.loc[df_group_by_layer.broken_output > 0, 'follow_all_instructions'] = 0
            max_accuracy = df_group_by_layer.follow_all_instructions.max()
            optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
            optimal_layers[instr] = optimal_layer[0]
        
        elif 'perplexity' in search_method:
            # add boolean column that is true when perplexity is low
            instr_df['low_perplexity'] = instr_df.perplexity < 2.5

            # set follow_all_instructions to 0 in df_group_by_layer when there exists an entry with large_perplexity > 0
            df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()
            df_group_by_layer.loc[df_group_by_layer.low_perplexity > 0, 'follow_all_instructions'] = 0
            max_accuracy = df_group_by_layer.follow_all_instructions.max()
            optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
            optimal_layers[instr] = optimal_layer[0]

        else:
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
    if 'quality_check' in search_method:
        df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_validation_quality_check_{instr_included}.h5', key='df', mode='w')
    elif 'perplexity' in search_method:
        df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_validation_perplexity_{instr_included}.h5', key='df', mode='w')
    else:
        df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_validation_{instr_included}.h5', key='df', mode='w')
elif search_method == 'cosine_similarity':
    df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_cosine_similarity.h5', key='df', mode='w')
else:
    raise ValueError('Invalid search method')
# %%

# load the df from folder + '/pre_computed_ivs_best_layer.h5'
# df = pd.read_hdf(f'{folder}/pre_computed_ivs_best_layer.h5', key='df')

# %%

# %%
import os
import sys
import json
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

config_path = os.path.join(project_dir, 'config/format')


@hydra.main(config_path=config_path, config_name='precompute_steering_vectors')
def precompute_vectors(args: DictConfig):
    with open(f'{project_dir}/data/format/ifeval_single_instr_format.jsonl') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    input_data_df = pd.DataFrame(data)
    all_instructions = list(input_data_df['instruction_id_list_for_eval'].apply(lambda x: x[0]).unique())

    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    all_instructions = list(filter(lambda x: any([f in x for f in filters]), all_instructions))

    w_perplexity = '_with_perplexity' if args.use_perplexity else ''
    cross_model = '_cross_model' if args.cross_model_steering else ''
    instr_included = 'instr' if args.include_instructions else 'no_instr'

    folder = f'{script_dir}/layer_search_out'
    file = f'{folder}/{args.model_name}/n_examples{args.n_examples}_seed{args.seed}{cross_model}{w_perplexity}/out_{instr_included}.jsonl'
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    validation_df = pd.DataFrame(results)
    optimal_layers = { instr: -1 for instr in all_instructions }

    for instr in all_instructions:
        if instr not in validation_df.single_instruction_id.unique():
            optimal_layers[instr] = -1
            continue

        instr_df = validation_df[validation_df.single_instruction_id == instr]
        
        if args.use_perplexity:
            # add boolean column that is true when perplexity is low
            instr_df['low_perplexity'] = instr_df.perplexity < args.preplexity_threshold

            df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()

            if args.model_name == 'gemma-2-9b' or args.model_name == 'gemma-2-2b':
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
        if args.model_name == 'gemma-2-2b' and args.cross_model_steering:
            print('Using representations from gemma-2-2b-it')
            rep_folder = f'{script_dir}/representations/gemma-2-2b-it/{args.representations_folder}'
        elif args.model_name == 'gemma-2-9b' and args.cross_model_steering:
            print('Using representations from gemma-2-9b-it')
            rep_folder = f'{script_dir}/representations/gemma-2-9b-it/{args.representations_folder}'
        else:
            rep_folder = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'

        file =f'{rep_folder}/{"".join(instr).replace(":", "_")}.h5'
        
        if not os.path.exists(file):
            print(f'File {file} does not exist')
            continue
        results_df = pd.read_hdf(file, key='df')

        row = {}
        row['instruction'] = instr

        hs_instr = results_df['last_token_rs'].to_list()
        hs_instr = torch.tensor(hs_instr, device=args.device)
        hs_no_instr = results_df['last_token_rs_no_instr'].to_list()
        hs_no_instr = torch.tensor(hs_no_instr, device=args.device)

        # check if hs has 4 dimensions
        if len(hs_instr.shape) == 3:
            hs_instr = hs_instr.unsqueeze(2)
            hs_no_instr = hs_no_instr.unsqueeze(2)

        if args.specific_layer is not None:
            selected_layer = args.specific_layer
        else:
            selected_layer = optimal_layers[instr]
            
        repr_diffs = hs_instr - hs_no_instr
        mean_repr_diffs = repr_diffs.mean(dim=0)
        last_token_mean_diff = mean_repr_diffs[:, -1, :]

        instr_dir = last_token_mean_diff[selected_layer] / last_token_mean_diff[selected_layer].norm()

        # average projection along the instruction direction
        proj = hs_instr[:, selected_layer, -1, :].to(args.device) @ instr_dir.to(args.device)
        proj_no_instr = hs_no_instr[:, selected_layer, -1, :].to(args.device) @ instr_dir.to(args.device)

        # get average projection along the instruction direction for each layer
        avg_proj = proj.mean()
        avg_proj_no_instr = proj_no_instr.mean()
        
        if selected_layer == -1:
            row['selected_layer'] = -1
            row['instr_dir'] = torch.zeros(hs_instr.shape[-1]).cpu().numpy()
            row['avg_proj'] = 0
            row['avg_proj_no_instr'] = 0
        else:
            row['selected_layer'] = selected_layer
            row['instr_dir'] = instr_dir.cpu().numpy()
            row['avg_proj'] = avg_proj
            row['avg_proj_no_instr'] = avg_proj_no_instr

        rows.append(row)

    df = pd.DataFrame(rows)

    # store the df in folder
    folder = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'
    if args.specific_layer is not None:
        df.to_hdf(f'{folder}/pre_computed_ivs_layer{args.specific_layer}.h5', key='df', mode='w')
    else:
        df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_validation{w_perplexity}{cross_model}_{instr_included}.h5', key='df', mode='w')
            
if __name__ == '__main__':
    precompute_vectors()
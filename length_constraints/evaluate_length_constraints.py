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
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct/ifeval_experiments')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct')
    print('We\'re on a sandbox machine')

import numpy as np
import torch
import pandas as pd
import re
import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import if_inference, generate_with_hooks, direction_ablation_hook, direction_projection_hook
import json
import plotly.express as px
import plotly.graph_objects as go
import functools
from transformer_lens import utils as tlutils
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path='../config', config_name='conf_length')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    # Some environment variables
    device = args.device
    print(f"Using device: {device}")

    # load the data
    with open('./data/ifeval_wo_instructions.jsonl') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_no_instr_df = pd.DataFrame(data)
    data_no_instr_df = data_no_instr_df.drop(columns=['prompt', 'instruction_id_list', 'prompt_hash'])
    # rename model_output to prompt_no_instr
    data_no_instr_df = data_no_instr_df.rename(columns={'model_output': 'prompt_no_instr'})

    data_no_instr_df = data_no_instr_df.tail(args.n_examples)

    new_rows = []
    
    for i, r in data_no_instr_df.iterrows():
        for n_sent in range(1, args.n_sent_max+1):
            row = dict(r)
            if args.constraint_type == 'at_least':
                constr = 'at least'
            elif args.constraint_type == 'at_most':
                constr = 'at most'
            elif args.constraint_type == 'exactly':
                constr = 'exactly'
            if n_sent == 1:
                instr = f' Answer using {constr} 1 sentence.'
            else:
                instr = f' Answer using {constr} {n_sent} sentences.'
            if args.include_instructions:
                row['prompt'] = row['prompt_no_instr'] + instr
            else:
                row['prompt'] = row['prompt_no_instr']
            row['length_constraint'] = n_sent - 1
            new_rows.append(row)

    data_df = pd.DataFrame(new_rows)

    # load tokenizer and model
    with open(args.path_to_hf_token) as f:
        hf_token = f.read()

    if args.steering != 'none':
        hf_model = False
    else:
        hf_model = True
    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=args.transformers_cache_dir, hf_token=hf_token, hf_model=hf_model)
    model.to(device)

    if args.dry_run:
        data_df = data_df.head(20)

    out_lines = []

    total = len(data_df)
    if 'add_vector' in args.steering:
        total *= len(args.steering_weights)
    p_bar = tqdm.tqdm(total=total)

    # load the stored representations
    if args.model_name == 'gemma-2-9b':
        print('Loading representations from gemma-2-9b-it')
        file = f'{args.project_dir}/{args.representations_folder}/gemma-2-9b-it/{args.length_rep_file}'
    else:
        file = f'{args.project_dir}/{args.representations_folder}/{args.model_name}/{args.length_rep_file}'
    results_df = pd.read_hdf(file, key='df')

    # TODO ideally this should not be hardcoded
    #steering_weights = {1: 75, 2: 60, 3: 50, 4: 40, 5: 30, 6: 20, 7: 10, 8: 5, 9: 2, 10: 1}
    
    # sort results_df by length_constraint
    results_df = results_df.sort_values(by='length_constraint')

    # max_length = min([x.shape[1] for x in results_df['last_token_rs_no_instr'].values])
    # hs_instr = results_df['last_token_rs'].values
    # hs_instr = torch.tensor([example_array[:, :max_length] for example_array in list(hs_instr)])
    # hs_no_instr = results_df['last_token_rs_no_instr'].values
    # hs_no_instr = torch.tensor([example_array[:, :max_length] for example_array in list(hs_no_instr)])

    # # compute the instrution vector
    # repr_diffs = hs_instr - hs_no_instr

    # layer_idx = args.source_layer_idx
    # instr_dir = last_token_mean_diff[layer_idx] / last_token_mean_diff[layer_idx].norm()

    # take the average of the representations
    # length_constraints_range = (args.length_constraint_rep_min, args.length_constraint_rep_max)
    
    # check if model has cfg attribute
    if hasattr(model, 'cfg'):
        d_model = model.cfg.d_model
    elif hasattr(model, 'config'):
        d_model = model.config.hidden_size

    length_specific_representations = torch.zeros((5, d_model))
    for i in range(5):
        # filter results_df to only include the reelvant length_constraint
        filtered_results_df = results_df[results_df['length_constraint'] == i]
        hs_instr = filtered_results_df['last_token_rs'].values
        hs_instr = torch.tensor([example_array[:, :] for example_array in list(hs_instr)])
        hs_no_instr = filtered_results_df['last_token_rs_no_instr'].values
        hs_no_instr = torch.tensor([example_array[:, :] for example_array in list(hs_no_instr)])
        repr_diffs = hs_instr - hs_no_instr

        length_specific_rep = repr_diffs[:, args.source_layer_idx, -1].mean(dim=0)
        length_specific_representations[i] = length_specific_rep

    # length_constraint_direction = length_specific_representations[length_constraints_range[0]:length_constraints_range[1]].mean(dim=0) / length_specific_representations[length_constraints_range[0]:length_constraints_range[1]].mean(dim=0).norm()
    # instr_dir = length_constraint_direction

    for length_constraint in range(0, args.n_sent_max):
        instr_data_df = data_df[data_df['length_constraint'] == length_constraint]

        if args.steering != 'none':

            if 'length_specific' in args.steering:
                print(f'Using length specific representation for length {length_constraint}')
                instr_dir = length_specific_representations[length_constraint] / length_specific_representations[length_constraint].norm()
            elif 'long-short' in args.steering:
                print(f'Using long-short representation for length {length_constraint}')
                if length_constraint == 0:
                    instr_dir = length_specific_representations[0] / length_specific_representations[0].norm()                 
                elif length_constraint == 1:
                    instr_dir = length_specific_representations[4] / length_specific_representations[4].norm()
                else:
                    print('Long-short representation is only defined for length 0 and 1')
            elif 'long' in args.steering:
                print(f'Using long-short representation for length {length_constraint}')
                if length_constraint == 0:
                    instr_dir = length_specific_representations[4] / length_specific_representations[4].norm()
                else:
                    print('Long-short representation is only defined for length 0')
            elif 'conciseness' in args.steering:
                print(f'Using conciseness representation for length {length_constraint}')
                instr_dir = length_specific_representations[0] / length_specific_representations[0].norm()
            elif 'verbosity' in args.steering:
                print(f'Using verbosity representation for length {length_constraint}')
                instr_dir = length_specific_representations[4] / length_specific_representations[4].norm()

            layer_idx = args.source_layer_idx

            if 'adjust_rs' in args.steering:
                # filter results_df to only include the reelvant length_constraint
                filtered_results_df = results_df[results_df['length_constraint'] == length_constraint]

                max_length = min([x.shape[1] for x in filtered_results_df['last_token_rs_no_instr'].values])
                hs_instr = filtered_results_df['last_token_rs'].values
                hs_instr = torch.tensor([example_array[:, :max_length] for example_array in list(hs_instr)])
              
                # average projection along the instruction direction
                proj = hs_instr[:, layer_idx, -1, :].to(device) @ instr_dir.to(device)

                # get average projection along the instruction direction for each layer
                avg_proj = proj.mean()
                print(f'Average projection along the instruction direction for layer {layer_idx} and length {length_constraint}: {avg_proj}')


        print('Steering weights:', args.steering_weights)


        # Run the model on each input
        for i, r in instr_data_df.iterrows():

            if 'add_vector' not in args.steering:
                steering_weights = [1]
            else:
                steering_weights = args.steering_weights

            for steering_weight in steering_weights:
                print(f'Running example {i} with steering weight {steering_weight}')
                row = dict(r)

                # apply the chat template
                if args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b':
                    example = f'Q: {r["prompt"]}\nA:'
                else:
                    messages = [{"role": "user", "content": r["prompt"]}]
                    example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    
                if args.steering == 'none':
                    out1 = if_inference(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
                elif args.steering != 'none':
                    intervention_dir = instr_dir.to(device)

                    if 'add_vector' in args.steering:
                        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir, weight=steering_weight)
                        row['steering_weight'] = steering_weight
                    elif 'adjust_rs' in args.steering:
                        hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)
                        row['steering_weight'] = avg_proj.item()

                    fwd_hooks = [(tlutils.get_act_name('resid_post', args.source_layer_idx), hook_fn)]
                    encoded_example = tokenizer(example, return_tensors='pt').to(device)
                    print(f'Example: {example}')
                    out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=args.max_generation_length, decode_directly=True)
                    # if out 1 is a list, take the first element
                    if isinstance(out1, list):
                        out1 = out1[0]
                    print(f'Generated: {out1}')
                else:
                    raise ValueError(f"Unknown steering method: {args.steering}")
                                                
                row['response'] = out1

                out_lines.append(row)
                p_bar.update(1)

    # write out_lines as jsonl
    folder = f'{args.project_dir}/{args.output_path}/{args.model_name}/1-{args.n_sent_max}sentences_{args.n_examples}examples/'
    if args.steering != 'none' and not args.include_instructions:
        folder += f'/{args.steering}_{args.source_layer_idx}'
        if args.apply_to_all_layers:
            folder += '_all_layers'
    elif args.steering != 'none' and args.include_instructions:
        folder += f'/{args.constraint_type}_instr_plus_{args.steering}_{args.source_layer_idx}'
        if args.apply_to_all_layers:
            folder += '_all_layers'
    
    if 'high-level' in args.length_rep_file:
        folder += '_high_level'

    if args.steering == 'none' and args.include_instructions:
        folder += f'/no_steering_{args.constraint_type}'
    elif args.steering == 'none' and not args.include_instructions:
        folder += '/no_steering_no_instruction'
    os.makedirs(folder, exist_ok=True)
    out_path = f'{folder}/out'
    out_path += ('_test' if args.dry_run else '')
    out_path +=  '.jsonl'

    # dump the args in the folder
    with open(f'{folder}/args.yaml', 'w') as f:
        OmegaConf.save(args, f)

    # print the max length of the responses
    print(f'Storing responses in {out_path}')

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')

# %%
# os.chdir('/home/t-astolfo/t-astolfo')
# args = OmegaConf.load('./config/conf_length.yaml')
# args['model_name'] = 'phi-3'
# args['dry_run'] = True
# run_experiment(args)
# %%
if __name__ == '__main__':
    run_experiment()
# %%
# %%
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
os.chdir(project_dir)

import sys
sys.path.append(script_dir)
sys.path.append(project_dir)

import numpy as np
import torch
import pandas as pd
import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import functools
from transformer_lens import utils as tlutils
from utils.generation_utils import generate_with_hooks, direction_projection_hook, activation_addition_hook
from ifeval_scripts.evaluation_main import test_instruction_following_loose

config_path = os.path.join(project_dir, 'config/format')


@hydra.main(config_path=config_path, config_name='find_best_layer')
def find_best_layer(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    device = args.device

    # load the data
    with open(f'{project_dir}/{args.data_path}') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_df = pd.DataFrame(data)

    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    data_df = data_df[data_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]

    # load tokenizer and model
    if args.steering != 'none':
        hf_model = False
    else:
        hf_model = True
    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=args.transformers_cache_dir, hf_model=hf_model)
    model.to(device)

    if args.dry_run:
        data_df = data_df.head(2)

    out_lines = []

    all_instructions = list(set([ item for l in data_df.instruction_id_list_for_eval for item in l]))

    # define layer to perform search over
    n_layers = model.cfg.n_layers
    if 'gemma-2-9b' in args.model_name:
        layer_range = range(n_layers // 5, n_layers, 3)
    else:
        layer_range = range(n_layers // 5, n_layers, 2)
    
    # -1 indicates "no steering"
    layer_range = [-1] + list(layer_range)

    num_all_examples = 0
    for instr in all_instructions:
        if 'language' in instr:
            num_all_examples += 1
        else:
            instr_data_df = data_df[[instr in l for l in data_df['instruction_id_list_for_eval'] ]]
            num_all_examples += min(args.n_examples_per_instruction, len(instr_data_df))

    total = num_all_examples * len(layer_range)
    p_bar = tqdm.tqdm(total=total)

    for instruction_type in all_instructions:
        instr_data_df = data_df[[[instruction_type] == l for l in data_df['instruction_id_list_for_eval'] ]]
        instr_data_df.reset_index(inplace=True, drop=True)
        instr_data_df = instr_data_df.sample(n=min(args.n_examples_per_instruction, len(instr_data_df)), random_state=args.seed)
        if 'language' in instruction_type:
            instr_data_df = instr_data_df.head(1)

        instr_data_df['instruction_id_list'] = instr_data_df['instruction_id_list_og']
        instr_data_df['prompt'] = instr_data_df['model_output']

        if args.dry_run:
            instr_data_df = instr_data_df.head(1)

        if args.steering != 'none':
            # load the stored representations
            if args.model_name == 'gemma-2-2b' and args.cross_model_steering:
                print('Loading representations from gemma-2-2b INSTRUCT')
                folder = f'{script_dir}/representations/gemma-2-2b-it/{args.representations_folder}'
            elif args.model_name == 'gemma-2-9b' and args.cross_model_steering:
                print('Loading representations from gemma-2-9b INSTRUCT')
                folder = f'{script_dir}/representations/gemma-2-9b-it/{args.representations_folder}'
            else:
                folder = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'
            file = f'{folder}/{"".join(instruction_type).replace(":", "_")}.h5'
            
            # check if the file exists
            if (not os.path.exists(file)):
                raise ValueError(f"File {file} does not exist")
            else:
                results_df = pd.read_hdf(file, key='df')

                # Convert to tensors
                hs_instr = torch.tensor(results_df['last_token_rs'].tolist())
                hs_no_instr = torch.tensor(results_df['last_token_rs_no_instr'].tolist())

                # compute the instrution vector
                repr_diffs = hs_instr - hs_no_instr
                mean_repr_diffs = repr_diffs.mean(dim=0)
                # check whether mean_repr_diffs has three dimensions
                if len(mean_repr_diffs.shape) == 3:
                    last_token_mean_diff = mean_repr_diffs[:, -1, :]
                else:
                    last_token_mean_diff = mean_repr_diffs

        for layer_idx in layer_range:
                    
            if args.steering != 'none':

                instr_dir = last_token_mean_diff[layer_idx] / last_token_mean_diff[layer_idx].norm()

                if args.steering == 'adjust_rs':
                    # average projection along the instruction direction
                    # check if hs_instr has 4 dimensions
                    if len(hs_instr.shape) == 4:
                        proj = hs_instr[:, layer_idx, -1, :].to(device) @ instr_dir.to(device)
                    else:
                        proj = hs_instr[:, layer_idx, :].to(device) @ instr_dir.to(device)

                    # get average projection along the instruction direction for each layer
                    avg_proj = proj.mean()

            # Run the model on each input
            for i, r in instr_data_df.iterrows():
                if args.include_instructions:
                    example = r['prompt'] # prompt w/ instruction
                else:
                    example = r['prompt_without_instruction'] # prompt w/o instruction

                row = dict(r)
                
                if args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b':
                    example = f'Q: {example}\nA:'
                else:
                    messages = [{"role": "user", "content": example}]
                    example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

                if ('json' in instruction_type) or ('multiple_sections' in instruction_type):
                        # for these instructions we don't want to truncate the output too early as they will fails the instruction following check
                        max_generation_length = 1024
                else:
                    max_generation_length = args.max_generation_length

                if layer_idx == -1:
                    # no steering
                    if (args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b'):
                        encoded_example = tokenizer(example, return_tensors='pt').to(device)
                        out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=[], max_tokens_generated=max_generation_length, return_decoded=True)
                    else:
                        out1 = generate(model, tokenizer, example, device, max_new_tokens=max_generation_length)
                else:
                    intervention_dir = instr_dir.to(device)

                    if args.steering == 'add_vector':
                        hook_fn = functools.partial(activation_addition_hook,direction=intervention_dir, weight=args.steering_weight)
                    elif args.steering == 'adjust_rs':
                        hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)

                    fwd_hooks = [(tlutils.get_act_name('resid_post', layer_idx), hook_fn)]
                    encoded_example = tokenizer(example, return_tensors='pt').to(device)
                    out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=max_generation_length, return_decoded=True)
                
                # if out 1 is a list, take the first element
                if isinstance(out1, list):
                    out1 = out1[0]
                                                
                row['response'] = out1

                # compute accuracy
                prompt_to_response = {}
                prompt_to_response[row['model_output']] = row['response']
                output = test_instruction_following_loose(r, prompt_to_response)
                row['follow_all_instructions'] = output.follow_all_instructions
                row['layer'] = layer_idx
                
                out_lines.append(row)
                p_bar.update(1)

    # write out_lines as jsonl
    folder = f'{script_dir}/{args.output_path}/{args.model_name}'
    folder += f'/n_examples{args.n_examples_per_instruction}_seed{args.seed}'
    folder += '_cross_model' if args.cross_model_steering else ''

    os.makedirs(folder, exist_ok=True)
    out_path = f'{folder}/out'
    if args.include_instructions:
        out_path += '_instr'
    else:
        out_path += '_no_instr'
    out_path += ('_test.jsonl' if args.dry_run else '.jsonl')

    print(f'Writing to {out_path}')

    # dump args to json
    args_file_name = 'args_instr.json' if args.include_instructions else 'args_no_instr.json'
    with open(f'{folder}/{args_file_name}', 'w') as f:
        f.write(OmegaConf.to_yaml(args))

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')

# %%
if __name__ == '__main__':
    find_best_layer()
# %%

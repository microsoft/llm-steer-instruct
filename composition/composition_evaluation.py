# %%
import os
if 'Users' in os.getcwd():
    os.chdir('C:\\Users\\t-astolfo\\workspace\\t-astolfo')
    print('We\'re on a Windows machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on a sandbox machine')

import sys
sys.path.append('/home/t-astolfo/t-astolfo')
sys.path.append('/home/t-astolfo/t-astolfo/ifeval_experiments')


import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import if_inference
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import functools
from transformer_lens import utils as tlutils
from utils.generation_utils import adjust_vectors
from time import time
from eval.evaluation_main import test_instruction_following_loose, test_instruction_following_strict


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
                if next_tokens[0] == model.tokenizer.eos_token_id or next_tokens[0] == 32007:
                    break
                all_toks[:,-max_tokens_generated+i] = next_tokens

    # truncate the tensor to remove padding
    all_toks = all_toks[:, :toks.shape[1] + i]

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

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

@hydra.main(config_path='../config', config_name='eval_multiple_instr')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    os.chdir(args.project_dir)

    # Some environment variables
    device = args.device
    print(f"Using device: {device}")

    transformer_cache_dir = None

    if 'no_instr' in args.data_path:
        raise ValueError('To run without instructions, set the include_instructions argument to False')


    # load the data
    with open(args.data_path) as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_df = pd.DataFrame(data)

    # load tokenizer and model
    with open(args.path_to_hf_token) as f:
        hf_token = f.read()

    if args.steering != 'none':
        hf_model = False
    else:
        hf_model = True
    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token, hf_model=hf_model)
    model.to(device)

    out_lines = []

    total = len(data_df) 

    print(f'Running on {total} examples')
    
    p_bar = tqdm.tqdm(total=total)

    if args.steering != 'none':
        # load the pre-computed IVs 
        folder = f'{args.project_dir}/../ifeval_experiments/representations/{args.model_name}/{args.representations_folder}'
        if args.source_layer_idx == -1:
            instr_included = 'no_instr' if args.include_instructions else 'instr' 
            # use best layer
            file = f'{folder}/pre_computed_ivs_best_layer_validation_{instr_included}.h5'
        else:
            file = f'{folder}/pre_computed_ivs_layer_{args.source_layer_idx}.h5'
        pre_computed_ivs = pd.read_hdf(file, key='df')

    if 'single_instr' in args.data_path:
        raise ValueError('Only multiple instr is supported')
    
    if not args.include_instructions:
        data_df['model_input'] = data_df['prompt_without_instruction']
    else:
        data_df['model_input'] = data_df['prompt']

    # Run the model on each input
    for i, r in data_df.iterrows():
        row = dict(r)
        example = row['model_input']

        if args.steering != 'none':
            # handle multiple instructions

            avg_projs = []
            instr_dirs = []
            layer_indices = []
            for instr in row['instruction_id_list_for_eval']:
                all_instr = pre_computed_ivs['instruction'].unique()
                if instr in all_instr:
                    instr_dir = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['instr_dir'].values[0]
                    instr_dir = torch.tensor(instr_dir, device=device)
                    layer_idx = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['max_diff_layer_idx'].values[0]
                    avg_proj = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['avg_proj'].values[0]
                else:
                    print(f'Instruction {instr} not found in pre-computed IVs')
                    instr_dir = torch.zeros(model.cfg.d_model, device=device)
                    layer_idx = 0
                    avg_proj = 0
                
                instr_dirs.append(instr_dir)
                avg_projs.append(avg_proj)
                layer_indices.append(layer_idx)
        

        # apply the chat template
        messages = [{"role": "user", "content": example}]
        example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if (args.steering == 'none'):
            out1 = if_inference(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
        elif args.steering != 'none':
            fwd_hooks = []
            for i, instr_dir in enumerate(instr_dirs):
                if layer_indices[i] == -1:
                    continue
                if args.steering == 'add_vector':
                    fwd_hooks.append((tlutils.get_act_name('resid_post', layer_indices[i]), functools.partial(direction_ablation_hook, direction=instr_dir, weight=args.steering_weight)))
                elif args.steering == 'adjust_rs':
                    fwd_hooks.append((tlutils.get_act_name('resid_post', layer_indices[i]), functools.partial(direction_projection_hook, direction=instr_dir, value_along_direction=avg_projs[i])))

            #encoded_example = tokenizer.apply_chat_template(messages, return_tensors='pt').to(device)
            encoded_example = tokenizer(example, return_tensors='pt').to(device)
            out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=args.max_generation_length)
            # if out 1 is a list, take the first element
            if isinstance(out1, list):
                out1 = out1[0]
        else:
            raise ValueError(f"Unknown steering method: {args.steering}")
        
        row['response'] = out1

        # compute accuracy
        prompt_to_response = {}
        prompt_to_response[row['prompt']] = row['response']
        output = test_instruction_following_loose(r, prompt_to_response)
        output_strict = test_instruction_following_strict(r, prompt_to_response)
        row['follow_all_instructions'] = output.follow_all_instructions
        row['follow_all_instructions_strict'] = output_strict.follow_all_instructions
        
        out_lines.append(row)
        p_bar.update(1)
    
    # write out_lines as jsonl
    folder = f'{args.output_path}/{args.model_name}'
    if 'all_base_x_all_inst' in args.representations_folder:
        folder += '/all_base_x_all_instr'
    if not args.include_instructions and args.steering == 'none':
        folder += '/no_instr'
    elif not args.include_instructions and args.steering != 'none':
        folder += f'/{args.steering}_{args.source_layer_idx}'
        if args.steering == 'add_vector':
            folder += f'_{args.steering_weight}'
        if args.apply_to_all_layers:
            folder += '_all_layers'
    elif args.steering != 'none':
        folder += f'/instr_plus_{args.steering}_{args.source_layer_idx}'
        if args.steering == 'add_vector':
            folder += f'_{args.steering_weight}'
        if args.apply_to_all_layers:
            folder += '_all_layers'
    else:
        folder += '/standard'
    os.makedirs(folder, exist_ok=True)
    out_path = f'{folder}/out'
    out_path += ('_test' if args.dry_run else '')
    out_path +=  '.jsonl'

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')

# %%
# args = OmegaConf.load('./ifeval_experiments/config/conf.yaml')
# args['model_name'] = 'phi-3'
# run_experiment(args)
# %%
if __name__ == '__main__':
    run_experiment()
# %%
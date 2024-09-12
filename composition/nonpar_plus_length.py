# %%
import os
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
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
import random
from eval.evaluation_main import test_instruction_following_loose
from collections import namedtuple
import nltk


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

@hydra.main(config_path='../config', config_name='nonpar_plus_length')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    os.chdir(args.project_dir)

    random.seed(args.seed)

    # Some environment variables
    device = args.device
    print(f"Using device: {device}")

    transformer_cache_dir = None

    # load the data
    with open(args.data_path) as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_df_no_length = pd.DataFrame(data)

    new_rows = []
    for i, r in data_df_no_length.iterrows():
        n_sent = random.randint(1, args.n_sent_max)
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
            prompt = row['prompt']
        else:
            prompt = row['prompt_without_instruction']
        if args.include_length_instr:
            prompt = prompt + instr
        row['model_input'] = prompt
    
        row['length_constraint'] = n_sent - 1
        new_rows.append(row)

    data_df = pd.DataFrame(new_rows)

    if args.dry_run:
        data_df = data_df.head(3)

    # load tokenizer and model
    with open(args.path_to_hf_token) as f:
        hf_token = f.read()

    if (args.steering != 'none') and ('llama' not in args.model_name.lower()):
        hf_model = False
        print('Using TLens model')
    else:
        hf_model = True
        print('Using HF model')
    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token, hf_model=hf_model)
    model.to(device)

    out_lines = []

    if args.nonparametric_only:
        # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
        filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
        data_df = data_df[data_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]

    total = len(data_df)

    if args.specific_instruction:
        all_instructions = list(set([ item for l in data_df.instruction_id_list_for_eval for item in l]))
    
        all_instructions = [instr for instr in all_instructions if args.specific_instruction in instr]
        total = len(data_df[data_df['instruction_id_list_for_eval'].apply(lambda x: any(y in all_instructions for y in x))])
        print(f'Using only the following instructions: {all_instructions}')

    print(f'Running on {total} examples')

    if args.steering != 'none':
        # load the pre-computed IVs 
        folder = f'{args.project_dir}/../ifeval_experiments/representations/{args.model_name}/{args.representations_folder}'
        if args.source_layer_idx == -1:
            instr_included = 'instr' if args.include_instructions else 'no_instr'
            # use best layer
            file = f'{folder}/pre_computed_ivs_best_layer_validation_{instr_included}.h5'
        else:
            file = f'{folder}/pre_computed_ivs_layer_{args.source_layer_idx}.h5'
        pre_computed_ivs = pd.read_hdf(file, key='df')

    if 'single_instr' in args.data_path:
        instr_data_df = data_df[data_df['instruction_id_list_for_eval'].apply(lambda x: len(x) == 1)]
        instr_data_df.reset_index(inplace=True, drop=True)
    else:
        raise ValueError('Only single_instr is supported for now')
    
    # load length representations
    file = f'{args.project_dir}/{args.length_representations_folder}/{args.model_name}/{args.length_rep_file}'
    results_df = pd.read_hdf(file, key='df')
    
    results_df = results_df.sort_values(by='length_constraint')
    
    if hasattr(model, 'cfg'):
        d_model = model.cfg.d_model
    elif hasattr(model, 'config'):
        d_model = model.config.hidden_size

    length_specific_representations = torch.zeros((args.n_sent_max, d_model))
    for i in range(args.n_sent_max):
        # filter results_df to only include the reelvant length_constraint
        filtered_results_df = results_df[results_df['length_constraint'] == i]
        hs_instr = filtered_results_df['last_token_rs'].values
        hs_instr = torch.tensor([example_array[:, :] for example_array in list(hs_instr)])
        hs_no_instr = filtered_results_df['last_token_rs_no_instr'].values
        hs_no_instr = torch.tensor([example_array[:, :] for example_array in list(hs_no_instr)])
        repr_diffs = hs_instr - hs_no_instr

        length_specific_rep = repr_diffs[:, args.source_layer_idx, -1].mean(dim=0)
        length_specific_representations[i] = length_specific_rep

    if args.length_steering != 'none':
        if 'conciseness' in args.length_steering:
            print(f'Using conciseness representation')
            length_instr_dir = length_specific_representations[0] / length_specific_representations[0].norm()
        elif 'verbosity' in args.length_steering:
            print(f'Using verbosity representation')
            length_instr_dir = length_specific_representations[4] / length_specific_representations[4].norm()

        length_instr_dir = length_instr_dir.to(device)

    p_bar = tqdm.tqdm(total=total)

    # Run the model on each input
    for i, r in instr_data_df.iterrows():
        row = dict(r)
        example = row['model_input']

        if args.steering != 'none':
            instr = row['instruction_id_list_for_eval'][0]
            all_instr = pre_computed_ivs['instruction'].unique()
            if instr in all_instr:
                instr_dir = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['instr_dir'].values[0]
                instr_dir = torch.tensor(instr_dir, device=device)
                layer_idx = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['max_diff_layer_idx'].values[0]
                avg_proj = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['avg_proj'].values[0]
                print(f'Layer used for {instr}: {layer_idx}')
            else:
                print(f'Instruction {instr} not found in pre-computed IVs')
                instr_dir = torch.zeros(model.cfg.d_model)
                layer_idx = -1
                avg_proj = 0

            # check whether avj_proj is a tensor
            # if not isinstance(avg_proj, torch.Tensor):
            avg_proj = torch.tensor(avg_proj, device=device)

        # apply the chat template
        messages = [{"role": "user", "content": example}]
        example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if (args.steering == 'none') and (args.length_steering == 'none'):
            print('No steering')
            out1 = if_inference(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
        else:
            fwd_hooks = []
            if args.steering != 'none':
                intervention_dir = instr_dir.to(device)

                if layer_idx == -1:
                    print(f'Running inference without steering for instruction: {row['instruction_id_list_for_eval'][0]}')
                else:
                    if args.steering == 'add_vector':
                        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir, weight=args.steering_weight)
                    elif args.steering == 'adjust_rs':
                        hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)
                    fwd_hooks.append((tlutils.get_act_name('resid_post', layer_idx), hook_fn))

            if args.length_steering != 'none':
                length_hook_fn = functools.partial(direction_ablation_hook, direction=length_instr_dir, weight=args.length_steering_weight)
                fwd_hooks.append((tlutils.get_act_name('resid_post', args.length_source_layer_idx), length_hook_fn))

            #encoded_example = tokenizer.apply_chat_template(messages, return_tensors='pt').to(device)
            encoded_example = tokenizer(example, return_tensors='pt').to(device)
            out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=args.max_generation_length)
            # if out 1 is a list, take the first element
            if isinstance(out1, list):
                out1 = out1[0]
        
        row['response'] = out1

        # if 'no_instr' in args.data_path:
        #     row['prompt_no_instr'] = row['prompt']
        #     row['prompt'] = row['original_prompt']
        #     row.pop('original_prompt')
        
        # compute accuracy
        prompt_to_response = {}
        prompt_to_response[row['prompt']] = row['response']
        output = test_instruction_following_loose(r, prompt_to_response)
        row['follow_all_instructions'] = output.follow_all_instructions

        # compute length of output
        row['response_length_sent'] = len(nltk.sent_tokenize(row['response']))
        row['response_length_words'] = len(row['response'].split())

        out_lines.append(row)
        p_bar.update(1)
    
    # if 'single_instr' not in args.data_path:
    #     break

    # write out_lines as jsonl
    folder = f'{args.output_path}/{args.model_name}'
    if 'all_base_x_all_inst' in args.representations_folder:
        folder += '/all_base_x_all_instr'

    if args.include_instructions:
        folder += '/instr'
    else:
        folder += '/no_instr'
    if args.include_length_instr:
        folder += '_w_length_instr'
    if args.steering != 'none':
        folder += f'/{args.steering}_{args.source_layer_idx}'
        if args.steering == 'add_vector':
            folder += f'_{args.steering_weight}'
    else:
        folder += '/no_steering'
    if args.length_steering_weight > 0:
        folder += f'_{args.length_steering}_L{args.length_source_layer_idx}_w{args.length_steering_weight}'
    else:
        folder += 'no_length_steering'
    
    os.makedirs(folder, exist_ok=True)
    out_path = f'{folder}/out'
    out_path += ('_test' if args.dry_run else '')
    if args.use_data_subset:
        out_path += f'_subset_{args.data_subset_ratio}'
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
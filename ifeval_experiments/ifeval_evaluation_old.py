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


@hydra.main(config_path='../config', config_name='conf')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    os.chdir(args.project_dir)

    # Some environment variables
    device = args.device
    print(f"Using device: {device}")

    transformer_cache_dir = None

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

    if args.dry_run:
        data_df = data_df.head(5)

    out_lines = []

    all_instructions = list(set([ item for l in data_df.instruction_id_list_for_eval for item in l]))

    total = len(data_df) if not args.use_data_subset else int(len(data_df) * (1 - args.data_subset_ratio))
    p_bar = tqdm.tqdm(total=total)

    for instruction_type in all_instructions:
        # TODO for now, we are only considering one instruction type
        if 'single_instr' in args.data_path:
            instr_data_df = data_df[[[instruction_type] == l for l in data_df['instruction_id_list_for_eval'] ]]
            instr_data_df.reset_index(inplace=True, drop=True)

            if args.use_data_subset:
                instr_data_df = instr_data_df.iloc[int(len(instr_data_df)*args.data_subset_ratio):]
        else:
            instr_data_df = data_df

        if args.dry_run:
            instr_data_df = instr_data_df.head(2)

        if args.steering != 'none':
            # load the stored representations
            folder = f'{args.project_dir}/representations/{args.model_name}/{args.representations_folder}'
            if args.use_data_subset:
                folder += f'_subset_{args.data_subset_ratio}'
            file = f'{folder}/{"".join(instruction_type).replace(":", "_")}.h5'
            # check if the file exists
            if (not os.path.exists(file)) and 'language' in instruction_type:
                print(f"File {file} does not exist")
                avg_proj = 0
                instr_dir = torch.zeros(model.cfg.d_model)
            else:
                results_df = pd.read_hdf(file, key='df')

                max_length = min([x.shape[1] for x in results_df['last_token_rs_no_instr'].values])
                hs_instr = results_df['last_token_rs'].values
                hs_instr = torch.tensor([example_array[:, :max_length] for example_array in list(hs_instr)])
                hs_no_instr = results_df['last_token_rs_no_instr'].values
                hs_no_instr = torch.tensor([example_array[:, :max_length] for example_array in list(hs_no_instr)])

                # compute the instrution vector
                repr_diffs = hs_instr - hs_no_instr
                mean_repr_diffs = repr_diffs.mean(dim=0)
                # check where mean_repr_diffs has three dimensions
                if len(mean_repr_diffs.shape) == 3:
                    last_token_mean_diff = mean_repr_diffs[:, -1, :]
                else:
                    last_token_mean_diff = mean_repr_diffs

                layer_idx = args.source_layer_idx
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
                    print(f'Average projection along the {instruction_type} direction for layer {layer_idx}: {avg_proj}')

        # Run the model on each input
        for i, r in instr_data_df.iterrows():
            row = dict(r)
            example = row['prompt']

            # apply the chat template for phi-3
            #messages = [{"role": "user", "content": example}]
            #example = tokenizer.decode(tokenizer.apply_chat_template(messages))
            # TODO fix apply chat template
            #example = f'<|user|>\n{example}<|end|>\n<|assistant|>'
            messages = [{"role": "user", "content": example}]
            example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            if args.steering == 'none':
                out1 = if_inference(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
            elif args.steering != 'none':
                intervention_dir = instr_dir.to(device)
                if args.apply_to_all_layers:
                    intervention_layers = list(range(0, layer_idx+1))
                else:
                    intervention_layers = list(range(layer_idx, layer_idx+1))

                if args.steering == 'add_vector':
                    hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir, weight=args.steering_weight)
                elif args.steering == 'adjust_rs':
                    hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)

                fwd_hooks = [(tlutils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_post']]
                #encoded_example = tokenizer.apply_chat_template(messages, return_tensors='pt').to(device)
                encoded_example = tokenizer(example, return_tensors='pt').to(device)
                out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=args.max_generation_length)
                # if out 1 is a list, take the first element
                if isinstance(out1, list):
                    out1 = out1[0]
            else:
                raise ValueError(f"Unknown steering method: {args.steering}")
                                            
            row['response'] = out1

            if 'no_instr' in args.data_path:
                row['prompt_no_instr'] = row['prompt']
                row['prompt'] = row['original_prompt']
                row.pop('original_prompt')
            
            out_lines.append(row)
            p_bar.update(1)
        
        if 'single_instr' not in args.data_path:
            break

    # write out_lines as jsonl
    folder = f'{args.output_path}/{args.model_name}'
    if 'single_instr' in args.data_path:
        folder += '/single_instr'
    else:
        folder += '/all_instr'
    if 'no_instr' in args.data_path and args.steering == 'none':
        folder += '/no_instr'
    elif  'no_instr' in args.data_path and args.steering != 'none':
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
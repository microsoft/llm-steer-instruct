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
from eval.evaluation_main import test_instruction_following_strict, test_instruction_following_loose



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

@hydra.main(config_path='../config', config_name='casing_and_exclude_word')
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

    all_instructions = list(set([ item for l in data_df.instruction_id_list_for_eval for item in l]))

    #all_instructions = [instr for instr in all_instructions if args.specific_instruction in instr]
    #data_df = data_df[data_df['instruction_id_list_for_eval'].apply(lambda x: any(y in all_instructions for y in x))]
    print(f'Using only the following instructions: {all_instructions}')
    
    if args.dry_run:
        data_df = data_df.sample(2)
    total = len(data_df)

    print(f'Running on {total} examples')

    if args.steering != 'none':
        # gather keywords needed for steering
        keywords = []
        for i in data_df.index:
            # TODO this assumes that the forbidden words appear in the second element of the kwargs tuple
            if 'forbidden_words' in data_df.loc[i].kwargs[1]:
                keywords.extend(data_df.loc[i].kwargs[1]['forbidden_words'])
            else:
                raise ValueError('Not implemented yet')
            
        print(f'Keywords: {keywords}')

        # load the pre-computed IVs for words 
        if args.specific_instruction == 'forbidden':
            file = f'{args.project_dir}/{args.representations_folder}/{args.model_name}/include_casing_plus_words_{args.n_examples}examples_hs.h5'
        else:
            raise ValueError(f'Unknown specific_instruction: {args.specific_instruction}')
        results_df = pd.read_hdf(file)

        pre_computed_word_ivs = {}

        print(f'words in results_df: {results_df.word.unique()}')
        for word in tqdm.tqdm(keywords, desc='Computing IVs'):

            filtered_df = results_df[results_df.word == word]

            if len(filtered_df) == 0:
                raise ValueError(f'No results found for word {word}')

            hs_instr = filtered_df['last_token_rs'].values
            hs_instr = torch.tensor([example_array[:, :] for example_array in list(hs_instr)])
            hs_no_instr = filtered_df['last_token_rs_no_instr'].values
            hs_no_instr = torch.tensor([example_array[:, :] for example_array in list(hs_no_instr)])

            # check if hs has 4 dimensions
            if len(hs_instr.shape) == 3:
                hs_instr = hs_instr.unsqueeze(2)
                hs_no_instr = hs_no_instr.unsqueeze(2)

            repr_diffs = hs_instr - hs_no_instr
            mean_repr_diffs = repr_diffs.mean(dim=0)
            last_token_mean_diff = mean_repr_diffs[:, -1, :]

            instr_dir = last_token_mean_diff[args.source_layer_idx] / last_token_mean_diff[args.source_layer_idx].norm()

            pre_computed_word_ivs[word] = instr_dir

            # load the pre-computed IVs 
            folder = f'{args.project_dir}/../ifeval_experiments/representations/{args.model_name}/{args.representations_folder_instr}'
            instr_included = 'no_instr' if 'no_instr' in args.data_path else 'instr' 
            # use best layer
            file = f'{folder}/pre_computed_ivs_best_layer_validation_{instr_included}.h5'
            pre_computed_ivs = pd.read_hdf(file, key='df')

    if not args.include_instructions:
        data_df['model_input'] = data_df['prompt_without_instruction']
    else:
        data_df['model_input'] = data_df['prompt']

    p_bar = tqdm.tqdm(total=total)

    # Run the model on each input
    for i, r in data_df.iterrows():
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
            else:
                print(f'Instruction {instr} not found in pre-computed IVs')
                instr_dir = torch.zeros(model.cfg.d_model)
                layer_idx = -1
                avg_proj = -1

            # check whether avj_proj is a tensor
            # if not isinstance(avg_proj, torch.Tensor):
            avg_proj = torch.tensor(avg_proj, device=device)


        # apply the chat templated
        messages = [{"role": "user", "content": example}]
        example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if (args.steering == 'none'):
            out1 = if_inference(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
        elif args.steering != 'none':
            # gather words
            if 'forbidden_words' in r.kwargs[1]:
                keywords = r.kwargs[1]['forbidden_words']
            else:
                raise ValueError('Not implemented yet')
            
            steering_weight = args.steering_weight / len(keywords)

            fwd_hooks = []
            for word in keywords:
                hook_fn_word = functools.partial(direction_ablation_hook,direction=pre_computed_word_ivs[word].to(device), weight=steering_weight)
                fwd_hooks.append((tlutils.get_act_name('resid_post', args.source_layer_idx), hook_fn_word))

            if args.steering == 'add_vector':
                hook_fn = functools.partial(direction_ablation_hook,direction=instr_dir.to(device), weight=args.steering_weight)
            elif args.steering == 'adjust_rs':
                hook_fn = functools.partial(direction_projection_hook, direction=instr_dir.to(device), value_along_direction=avg_proj)
            fwd_hooks.append((tlutils.get_act_name('resid_post', layer_idx), hook_fn))
            
            print(f'Words to steer for: {keywords}')
            print(f'Generating with hooks: {len(fwd_hooks)}')

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
        output = test_instruction_following_strict(r, prompt_to_response)
        row['follow_all_instructions'] = output.follow_all_instructions
        row['follow_instruction_list'] = output.follow_instruction_list
        output_loose = test_instruction_following_loose(r, prompt_to_response)
        row['follow_all_instructions_loose'] = output_loose.follow_all_instructions
        row['follow_instruction_list_loose'] = output_loose.follow_instruction_list
        
        out_lines.append(row)
        p_bar.update(1)
    
    # if 'single_instr' not in args.data_path:
    #     break

    # write out_lines as jsonl
    folder = f'{args.output_path}/{args.model_name}'

    if args.specific_instruction:
        folder += f'/{args.specific_instruction}'
    if not args.include_instructions and args.steering == 'none':
        folder += '/no_instr'
    elif  not args.include_instructions and args.steering != 'none':
        folder += f'/{args.steering}_{args.source_layer_idx}_n_examples{args.n_examples}'
        if args.steering == 'add_vector':
            folder += f'_{args.steering_weight}'
    elif args.steering != 'none':
        folder += f'/instr_plus_{args.steering}_{args.source_layer_idx}_n_examples{args.n_examples}'
        if args.steering == 'add_vector':
            folder += f'_{args.steering_weight}'
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
# %%
import os
import sys
import torch
import pandas as pd
import tqdm
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import functools
from transformer_lens import utils as tlutils

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate, generate_with_hooks, activation_addition_hook, direction_projection_hook
from ifeval_scripts.evaluation_main import test_instruction_following_loose

config_path = os.path.join(project_dir, 'config/format')


@hydra.main(config_path=config_path, config_name='format_evaluation')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    # load the data
    with open(f'{project_dir}/{args.data_path}') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_df = pd.DataFrame(data)

    if args.dry_run:
        data_df = data_df.head(5)

    # load tokenizer and model
    if args.steering == 'none':
        hf_model = args.hf_model
    else:
        hf_model = False
    model, tokenizer = load_model_from_tl_name(args.model_name, device=args.device, cache_dir=args.transformers_cache_dir, hf_model=hf_model)
    model.to(args.device)

    total = len(data_df)
    p_bar = tqdm.tqdm(total=total)

    if args.steering != 'none':
        folder = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'
        if args.source_layer_idx == -1:
            # use best layer
            cross_model_flag = '_cross_model' if args.cross_model_steering else ''
            use_perplexity_flag = '_with_perplexity' if args.use_perplexity else ''
            include_instr_flag = '_instr' if args.include_instructions else '_no_instr'
            file_path = f'{folder}/pre_computed_ivs_best_layer_validation{use_perplexity_flag}{cross_model_flag}{include_instr_flag}.h5'
        else:
            file_path = f'{folder}/pre_computed_ivs_layer_{args.source_layer_idx}.h5'
        pre_computed_ivs = pd.read_hdf(file_path, key='df')

    out_lines = []

    # Run the model on each input
    for i, r in data_df.iterrows():
        row = dict(r)

        if args.include_instructions:
            prompt = row['prompt']
        else:
            prompt = row['prompt_without_instruction']

        if args.steering != 'none':
            instr = row['instruction_id_list_for_eval'][0]
            all_instr = pre_computed_ivs['instruction'].unique()
            if instr in all_instr:
                instr_dir = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['instr_dir'].values[0]
                instr_dir = torch.tensor(instr_dir, device=args.device)
                layer_idx = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['selected_layer'].values[0]
                avg_proj = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['avg_proj'].values[0]
            else:
                print(f'Instruction {instr} not found in pre-computed IVs')
                instr_dir = torch.zeros(model.cfg.d_model)
                layer_idx = -1
                avg_proj = -1

            avg_proj = torch.tensor(avg_proj, device=args.device)

        # format the prompt
        if args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b':
            example = f'Q: {prompt}\nA:'
        else:
            messages = [{"role": "user", "content": prompt}]
            example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if args.steering == 'none' or layer_idx == -1:
            row['steering_layer'] = -1
            if (args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b'):
                encoded_example = tokenizer(example, return_tensors='pt').to(args.device)
                out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=[], max_tokens_generated=args.max_generation_length, return_decoded=True)
            else:
                out1 = generate(model, tokenizer, example, args.device, max_new_tokens=args.max_generation_length)
        
        else:
            intervention_dir = instr_dir.to(args.device)
            row['steering_layer'] = int(layer_idx)

            if args.steering == 'add_vector':
                hook_fn = functools.partial(activation_addition_hook,direction=intervention_dir, weight=args.steering_weight)
            elif args.steering == 'adjust_rs':
                hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)
            else:
                raise ValueError(f"Unknown steering method: {args.steering}")

            fwd_hooks = [(tlutils.get_act_name('resid_post', layer_idx), hook_fn)]
            encoded_example = tokenizer(example, return_tensors='pt').to(args.device)
            
            out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=args.max_generation_length, return_decoded=True)
            
        # if out 1 is a list, take the first element
        if isinstance(out1, list):
            out1 = out1[0]
            
        row['response'] = out1

        # compute accuracy
        prompt_to_response = {}
        prompt_to_response[row['prompt']] = row['response']
        output = test_instruction_following_loose(r, prompt_to_response)
        row['follow_all_instructions'] = output.follow_all_instructions

        out_lines.append(row)
        p_bar.update(1)

    # Build the output folder path
    out_folder = os.path.join(script_dir, args.output_path, args.model_name)
    if not args.include_instructions and args.steering == 'none':
        out_folder = os.path.join(out_folder, 'no_instr')
    elif not args.include_instructions and args.steering != 'none':
        out_folder = os.path.join(out_folder, f"{args.steering}_{args.source_layer_idx}")
        if args.use_perplexity:
            out_folder += '_perplexity'
        if args.steering == 'add_vector':
            out_folder += f"_{args.steering_weight}"
    elif args.steering != 'none':
        out_folder = os.path.join(out_folder, f"instr_plus_{args.steering}_{args.source_layer_idx}")
        if args.use_perplexity:
            out_folder += '_perplexity'
        if args.steering == 'add_vector':
            out_folder += f"_{args.steering_weight}"
    else:
        out_folder = os.path.join(out_folder, 'standard')

    if args.steering == 'none' and (not args.hf_model):
        out_folder += '_no_hf'

    if args.cross_model_steering:
        out_folder += '_cross_model'

    os.makedirs(out_folder, exist_ok=True)

    file_name = 'out.jsonl' if not args.dry_run else 'test.jsonl'
    out_file = os.path.join(out_folder, file_name)

    # dump args in the folder
    with open(os.path.join(out_folder, 'args.json'), 'w') as f:
        f.write(OmegaConf.to_yaml(args))

    with open(out_file, 'w') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')

# %%
if __name__ == '__main__':
    run_experiment()
# %%
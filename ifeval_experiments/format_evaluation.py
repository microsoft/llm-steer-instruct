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
sys.path.append(script_dir)
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate
from utils.generation_utils import generate_with_hooks, activation_addition_hook, direction_projection_hook

config_path = os.path.join(project_dir, 'config')


@hydra.main(config_path='config_path', config_name='format_evaluation')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    device = args.device

    # load the data
    with open(args.data_path) as f:
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
    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=args.transformers_cache_dir, hf_model=hf_model)
    model.to(device)

    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    data_df = data_df[data_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]

    total = len(data_df) if not args.use_data_subset else int(len(data_df) * (1 - args.data_subset_ratio))

    print(f'Running on {total} examples')
    
    p_bar = tqdm.tqdm(total=total)

    out_lines = []

    if args.steering != 'none':
        folder = f'{project_dir}/representations/{args.model_name}/{args.representations_folder}'
        if args.source_layer_idx == -1:
            instr_included = 'no_instr' if 'no_instr' in args.data_path else 'instr' 
            # use best layer
            if args.cross_model_steering:
                file_path = f'{folder}/pre_computed_ivs_best_layer_validation_perplexity_cross_model_{instr_included}.h5'
            else:
                file_path = f'{folder}/pre_computed_ivs_best_layer_validation_perplexity_{instr_included}.h5'
        else:
            file_path = f'{folder}/pre_computed_ivs_layer_{args.source_layer_idx}.h5'
        pre_computed_ivs = pd.read_hdf(file_path, key='df')

    # make sure that all the examples have a single instruction
    instr_data_df = data_df[data_df['instruction_id_list_for_eval'].apply(lambda x: len(x) == 1)]
    instr_data_df.reset_index(inplace=True, drop=True)

    # Run the model on each input
    for i, r in instr_data_df.iterrows():
        row = dict(r)

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

            avg_proj = torch.tensor(avg_proj, device=device)

        # format the prompt
        if args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b':
            example = f'Q: {row["prompt"]}\nA:'
        else:
            messages = [{"role": "user", "content": row["prompt"]}]
            example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if args.steering == 'none' or layer_idx == -1:
            row['steering_layer'] = -1
            if (args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b'):
                encoded_example = tokenizer(example, return_tensors='pt').to(device)
                out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=[], max_tokens_generated=args.max_generation_length, return_decoded=True)
            else:
                out1 = generate(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
        
        else:
            intervention_dir = instr_dir.to(device)
            row['steering_layer'] = int(layer_idx)

            if args.steering == 'add_vector':
                hook_fn = functools.partial(activation_addition_hook,direction=intervention_dir, weight=args.steering_weight)
            elif args.steering == 'adjust_rs':
                hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)
            else:
                raise ValueError(f"Unknown steering method: {args.steering}")

            fwd_hooks = [(tlutils.get_act_name('resid_post', layer_idx), hook_fn)]
            encoded_example = tokenizer(example, return_tensors='pt').to(device)
            
            out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=args.max_generation_length, return_decoded=True)
            
        # if out 1 is a list, take the first element
        if isinstance(out1, list):
            out1 = out1[0]
            
        row['response'] = out1

        if 'no_instr' in args.data_path:
            row['prompt_no_instr'] = row['prompt']
            row['prompt'] = row['original_prompt']
            row.pop('original_prompt')
        
        out_lines.append(row)
        p_bar.update(1)

    # Build the output folder path
    out_folder = os.path.join(args.project_dir, args.output_path, args.model_name)
    out_folder = os.path.join(out_folder, 'single_instr' if 'single_instr' in args.data_path else 'all_instr')
    if 'all_base_x_all_inst' in args.representations_folder:
        out_folder = os.path.join(out_folder, 'all_base_x_all_instr')
    if 'no_instr' in args.data_path and args.steering == 'none':
        out_folder = os.path.join(out_folder, 'no_instr')
    elif 'no_instr' in args.data_path and args.steering != 'none':
        out_folder = os.path.join(out_folder, f"{args.steering}_{args.source_layer_idx}")
        if 'perplexity' in file_path:
            out_folder += '_perplexity'
        if args.steering == 'add_vector':
            out_folder += f"_{args.steering_weight}"
    elif args.steering != 'none':
        out_folder = os.path.join(out_folder, f"instr_plus_{args.steering}_{args.source_layer_idx}")
        if 'perplexity' in file_path:
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

    file_name = 'out.jsonl' if args.dry_run else 'test.jsonl'
    out_file = os.path.join(out_folder, file_name)

    with open(out_file, 'w') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')

# %%
if __name__ == '__main__':
    run_experiment()
# %%
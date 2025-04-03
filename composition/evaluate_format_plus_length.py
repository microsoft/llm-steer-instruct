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
import random
import nltk

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate, generate_with_hooks, activation_addition_hook, direction_projection_hook
from ifeval_scripts.evaluation_main import test_instruction_following_loose

config_path = os.path.join(project_dir, 'config/composition')


@hydra.main(config_path=config_path, config_name='evaluation')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    random.seed(args.seed)

    device = args.device

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

    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=args.transformers_cache_dir)
    model.to(device)

    total = len(data_df)

    if args.steering != 'none':
        # load the pre-computed steering vectors
        folder = f'{project_dir}/format/representations/{args.model_name}/{args.representations_folder}'
        if args.source_layer_idx == -1:
            # use best layer
            use_perplexity_flag = '_with_perplexity' if args.use_perplexity else ''
            include_instr_flag = '_instr' if args.include_instructions else '_no_instr'
            file_path = f'{folder}/pre_computed_ivs_best_layer_validation{use_perplexity_flag}{include_instr_flag}.h5'
        else:
            file_path = f'{folder}/pre_computed_ivs_layer_{args.source_layer_idx}.h5'
        pre_computed_ivs = pd.read_hdf(file_path, key='df')
    
    # load length representations
    length_file = f'{project_dir}/{args.length_representations_folder}/{args.model_name}/{args.length_rep_file}'
    results_df = pd.read_hdf(length_file, key='df')
    
    results_df = results_df.sort_values(by='length_constraint')
    
    if hasattr(model, 'cfg'):
        d_model = model.cfg.d_model
    elif hasattr(model, 'config'):
        d_model = model.config.hidden_size

    length_specific_representations = torch.zeros((args.n_sent_max, d_model))
    for i in range(args.n_sent_max):
        # filter results_df to only include the reelvant length_constraint
        filtered_results_df = results_df[results_df['length_constraint'] == i]
        hs_instr = torch.tensor(filtered_results_df['last_token_rs'].tolist())
        hs_no_instr = torch.tensor(filtered_results_df['last_token_rs_no_instr'].tolist())
        repr_diffs = hs_instr - hs_no_instr

        length_specific_rep = repr_diffs[:, args.source_layer_idx, -1].mean(dim=0)
        length_specific_representations[i] = length_specific_rep

    if args.length_steering != 'none':
        if 'conciseness' in args.length_steering:
            length_instr_dir = length_specific_representations[0] / length_specific_representations[0].norm()
        elif 'verbosity' in args.length_steering:
            length_instr_dir = length_specific_representations[4] / length_specific_representations[4].norm()

        length_instr_dir = length_instr_dir.to(device)

    out_lines = []

    p_bar = tqdm.tqdm(total=total)

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
                instr_dir = torch.tensor(instr_dir, device=device)
                layer_idx = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['selected_layer'].values[0]
                avg_proj = pre_computed_ivs[pre_computed_ivs['instruction'] == instr]['avg_proj'].values[0]
            else:
                instr_dir = torch.zeros(model.cfg.d_model)
                layer_idx = -1
                avg_proj = 0

            avg_proj = torch.tensor(avg_proj, device=device)

        # apply the chat template
        messages = [{"role": "user", "content": prompt}]
        example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if (args.steering == 'none') and (args.length_steering == 'none'):
            out1 = generate(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
        else:
            fwd_hooks = []
            if args.steering != 'none' and layer_idx != -1:
                intervention_dir = instr_dir.to(device)

                if args.steering == 'add_vector':
                    hook_fn = functools.partial(activation_addition_hook,direction=intervention_dir, weight=args.steering_weight)
                elif args.steering == 'adjust_rs':
                    hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)
                fwd_hooks.append((tlutils.get_act_name('resid_post', layer_idx), hook_fn))

            if args.length_steering != 'none':
                length_hook_fn = functools.partial(activation_addition_hook, direction=length_instr_dir, weight=args.length_steering_weight)
                fwd_hooks.append((tlutils.get_act_name('resid_post', args.length_source_layer_idx), length_hook_fn))

            encoded_example = tokenizer(example, return_tensors='pt').to(device)
            out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=args.max_generation_length)

            # if out 1 is a list, take the first element
            if isinstance(out1, list):
                out1 = out1[0]
        
        row['response'] = out1

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

    # write out_lines as jsonl
    folder = os.path.join(script_dir, args.output_path, args.model_name)

    if args.include_instructions:
        folder += '/instr'
    else:
        folder += '/no_instr'
    if args.include_length_instr:
        folder += '_w_length_instr'
    if args.steering != 'none':
        folder += f'/{args.steering}_{args.source_layer_idx}'
        if args.use_perplexity:
            folder += '_perplexity'
        if args.steering == 'add_vector':
            folder += f'_{args.steering_weight}'
    else:
        folder += '/no_steering'
    if args.length_steering != 'none':
        folder += f'_{args.length_steering}_L{args.length_source_layer_idx}_w{args.length_steering_weight}'
    else:
        folder += '_no_length_steering'
    
    os.makedirs(folder, exist_ok=True)
    out_path = f'{folder}/'
    out_path += ('test.jsonl' if args.dry_run else 'out.jsonl')

    # dump args in the folder
    with open(f'{folder}/args.json', 'w') as f:
        f.write(OmegaConf.to_yaml(args))

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')

# %%
if __name__ == '__main__':
    run_experiment()
# %%
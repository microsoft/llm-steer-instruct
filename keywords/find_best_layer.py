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
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(script_dir)
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate_with_hooks, activation_addition_hook, generate

config_path = os.path.join(project_dir, 'config')

@hydra.main(config_path=config_path, config_name='find_best_layer_keywords')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    random.seed(args.seed)

    # Some environment variables
    device = args.device

    # load the data
    with open(args.data_path) as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_df = pd.DataFrame(data)

    new_data_rows = []
    for i, r in data_df.iterrows():
        row = dict(r)
        if 'exclude' in args.constraint:
            keywords = r['likely_words'][:1]
        elif args.constraint == 'include':
            keywords = r['unlikely_words'][:1]
        else:
            raise ValueError(f'Unknown constraint: {args.constraint}')
        for word in keywords:
            row['word'] = word
            new_data_rows.append(row)

    data_df = pd.DataFrame(new_data_rows)

    if args.dry_run:
        data_df = data_df.head(2)

    # load tokenizer and model
    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=args.transformer_cache_dir, hf_model=False)
    model.to(device)

    out_lines = []

    n_layers = model.cfg.n_layers
    layer_range = range(n_layers // 5, n_layers, 2)
    layer_range = [-1] + list(layer_range)

    total = len(data_df) * (len(layer_range) - 1) * (len(args.steering_weights)) + len(data_df)
    p_bar = tqdm.tqdm(total=total)

    # gather keywords needed for steering
    keywords = []
    for i in data_df.index:
        if args.constraint == 'include':
            keywords.extend(data_df.loc[i]['unlikely_words'])
        elif 'exclude' in args.constraint:
            keywords.extend(data_df.loc[i]['likely_words'])

    # load the pre-computed IVs 
    if args.constraint == 'include':
        file = f'{args.project_dir}/representations/{args.model_name}/include_validation_include_{args.n_examples}examples_hs.h5'
    elif args.constraint == 'exclude':
        file = f'{args.project_dir}/representations/{args.model_name}/include_validation_exclude_{args.n_examples}examples_hs.h5'
    elif args.constraint == 'exclude_w_exclude_rep':
            file = f'{args.project_dir}/representations/{args.model_name}/exclude_validation_exclude_{args.n_examples}examples_hs.h5'
    else:
        raise ValueError(f'Unknown specific_instruction: {args.specific_instruction}')
    
    results_df = pd.read_hdf(file)

    for layer_idx in layer_range:

        pre_computed_ivs = {}

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

            instr_dir = last_token_mean_diff[layer_idx] / last_token_mean_diff[layer_idx].norm()

            pre_computed_ivs[word] = instr_dir

        for steering_weight in args.steering_weights:
        
            # Run the model on each input
            for i, r in data_df.iterrows():
                row = dict(r)

                example = r['question'] 
                if args.include_instruction:
                    phrasings_exclude = [' Do not include the word {}.', ' Make sure not to include the word "{}".', ' Do not use the word {}.', ' Do not say "{}".', ' Please exclude the word "{}".', ' The output should not contain the word "{}".']
                    phrasings_include = [' Make sure to include the word "{}".', ' Please include the word "{}".', ' The output should contain the word "{}".', ' The output must contain the word "{}".', ' The output should say the word "{}".']
                    if args.constraint == 'include':
                        # add the instruction to the example
                        phrasing = random.choice(phrasings_include)
                    elif 'exclude' in args.constraint:
                        phrasing = random.choice(phrasings_exclude)
                    example += phrasing.format(r['word'])

                row['model_input'] = example

                messages = [{"role": "user", "content": example}]
                example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                if layer_idx == -1:
                    out1 = generate(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
                else:
                    intervention_dir = pre_computed_ivs[r['word']].to(device)

                    if args.steering == 'add_vector':
                        hook_fn = functools.partial(activation_addition_hook,direction=intervention_dir, weight=steering_weight)
                    else:
                        raise ValueError(f'Keyword steering only supports add_vector, got {args.steering}')

                    fwd_hooks = [(tlutils.get_act_name('resid_post', layer_idx), hook_fn)]
                    encoded_example = tokenizer(example, return_tensors='pt').to(device)
                    out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=args.max_generation_length)
                    # if out 1 is a list, take the first element
                    if isinstance(out1, list):
                        out1 = out1[0]
                                                
                row['response'] = out1
                row['layer'] = layer_idx
                row['steering_weight'] = steering_weight
              
                # compute accuracy
                occurrences = len(re.findall(rf'\b{re.escape(r["word"].lower())}\b', row['response'].lower()))
                keyword_is_present = occurrences > 0

                row['occurrences'] = occurrences
                row['keyword_is_present'] = keyword_is_present
                row['accuracy'] = keyword_is_present if args.constraint == 'include' else not keyword_is_present
                
                out_lines.append(row)
                p_bar.update(1)
            
            if layer_idx == -1:
                break

    # write out_lines as jsonl
    folder = f'{project_dir}/{args.output_path}/{args.model_name}/{args.constraint}'
    folder += f'/n_examples{args.n_examples}_seed{args.seed}'

    os.makedirs(folder, exist_ok=True)
    out_path = f'{folder}/out'
    if args.include_instruction:
        out_path += '_instr'
    else:
        out_path += '_no_instr'
    out_path += ('_test' if args.dry_run else '')
    out_path +=  '.jsonl'

    print(f'Writing to {out_path}')

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    run_experiment()
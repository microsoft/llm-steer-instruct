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
from utils.generation_utils import generate, generate_with_hooks, activation_addition_hook, direction_projection_hook
from eval.evaluation_main import test_instruction_following_strict

config_path = os.path.join(project_dir, 'config')


@hydra.main(config_path=config_path, config_name='keyword_evaluation')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    device = args.device

    # load the data
    with open(f'{args.data_path}') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]

    data_df = pd.DataFrame(data)

    # load tokenizer and model
    if args.steering != 'none':
        hf_model = False
    else:
        hf_model = True

    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=args.transformers_cache_dir, hf_model=hf_model)
    model.to(device)

    all_instructions = list(set([ item for l in data_df.instruction_id_list for item in l]))

    if 'validation' not in args.specific_instruction:
        all_instructions = [instr for instr in all_instructions if args.specific_instruction in instr]
        data_df = data_df[data_df['instruction_id_list'].apply(lambda x: any(y in all_instructions for y in x))]
        print(f'Using only the following instructions: {all_instructions}')
    
    if args.dry_run:
        data_df = data_df.head(2)

    total = len(data_df)

    if args.steering != 'none':
        # gather keywords needed for steering
        keywords = []
        for i in data_df.index:
            if 'forbidden_words' in data_df.loc[i].kwargs[0]:
                keywords.extend(data_df.loc[i].kwargs[0]['forbidden_words'])
            else:
                'keywords' in data_df.loc[i].kwargs[0]
                keywords.extend(data_df.loc[i].kwargs[0]['keywords'])
            
        # load the pre-computed IVs 
        if args.specific_instruction == 'forbidden':
            file = f'{args.project_dir}/representations/{args.model_name}/include_ifeval_exclude_{args.n_examples}examples_hs.h5'
        elif args.specific_instruction == 'forbidden_w_forbidden_rep':
            file = f'{args.project_dir}/representations/{args.model_name}/exclude_ifeval_exclude_{args.n_examples}examples_hs.h5'
        elif args.specific_instruction == 'existence':
            file = f'{args.project_dir}/representations/{args.model_name}/include_ifeval_include_{args.n_examples}examples_hs.h5'
        elif args.specific_instruction == 'existence_validation' or args.specific_instruction == 'forbidden_validation':
            file = f'{args.project_dir}/representations/{args.model_name}/include_validation_{args.n_examples}examples_hs.h5'
        elif args.specific_instruction == 'forbidden_validation_w_forbidden_rep':
            file = f'{args.project_dir}/representations/{args.model_name}/exclude_validation_{args.n_examples}examples_hs.h5'
        else:
            raise ValueError(f'Unknown specific_instruction: {args.specific_instruction}')
        results_df = pd.read_hdf(file)

        pre_computed_ivs = {}
        avg_projections = {}

        for word in tqdm.tqdm(keywords, desc='Computing IVs'):

            filtered_df = results_df[results_df.word == word]

            if len(filtered_df) == 0:
                raise ValueError(f'No results found for word {word}')

            hs_instr = filtered_df['last_token_rs'].values
            hs_instr = torch.tensor(list(hs_instr))
            hs_no_instr = filtered_df['last_token_rs_no_instr'].values
            hs_no_instr = torch.tensor(list(hs_no_instr))

            # check if hs has 4 dimensions
            if len(hs_instr.shape) == 3:
                hs_instr = hs_instr.unsqueeze(2)
                hs_no_instr = hs_no_instr.unsqueeze(2)

            repr_diffs = hs_instr - hs_no_instr
            mean_repr_diffs = repr_diffs.mean(dim=0)
            last_token_mean_diff = mean_repr_diffs[:, -1, :]

            instr_dir = last_token_mean_diff[args.source_layer_idx] / last_token_mean_diff[args.source_layer_idx].norm()

            pre_computed_ivs[word] = instr_dir

            # compute projection for inputs with instruction
            proj = hs_instr[:, args.source_layer_idx] @ instr_dir
            mean_proj = proj.mean()

            avg_projections[word] = mean_proj

    if not args.include_instructions:
        data_df['model_input'] = data_df['prompt_without_instruction']
    else:
        data_df['model_input'] = data_df['prompt']

    out_lines = []
    p_bar = tqdm.tqdm(total=total)

    # Run the model on each input
    for i, r in data_df.iterrows():
        row = dict(r)
        example = row['model_input']

        # format the prompt
        messages = [{"role": "user", "content": example}]
        example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if args.steering == 'none':
            out1 = generate(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
        elif args.steering != 'none':
            # gather words
            if 'forbidden_words' in r.kwargs[0]:
                keywords = r.kwargs[0]['forbidden_words']
            elif 'keywords' in r.kwargs[0]:
                keywords = r.kwargs[0]['keywords']
            else:
                raise ValueError('No keywords found in kwargs')
            
            assert len(keywords) == 1, f'Expected a single keyword, got {keywords}'
            word = keywords[0]

            fwd_hooks = []
            if args.steering == 'add_vector':
                hook_fn = functools.partial(activation_addition_hook,direction=pre_computed_ivs[word].to(device), weight=args.steering_weight)
            elif args.steering == 'adjust_rs':
                # this works significantly worse than add_vector
                hook_fn = functools.partial(direction_projection_hook, direction=pre_computed_ivs[word].to(device), value_along_direction=avg_projections[word])

            fwd_hooks.append((tlutils.get_act_name('resid_post', args.source_layer_idx), hook_fn))

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
        
        out_lines.append(row)
        p_bar.update(1)

    # write out_lines as jsonl
    folder = f'{project_dir}/{args.output_path}/{args.model_name}'

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
    
    # dump args in the folder
    with open(f'{folder}/args.json', 'w') as f:
        f.write(OmegaConf.to_yaml(args))

    out_path = f'{folder}/out'
    out_path += ('_test.jsonl' if args.dry_run else '.jsonl')

    print(f'Storing at: {out_path}')

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    run_experiment()
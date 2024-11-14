# %%
import os
import sys
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/keywords')
    print('We\'re on the local machine')
elif 'cluster' in os.getcwd():
    os.chdir('/cluster/project/sachan/alessandro/llm-steer-instruct')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct/keywords')
    print('We\'re on a sandbox machine')

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
from collections import namedtuple
import random
import re

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


@hydra.main(config_path='../config', config_name='find_best_layer_keywords')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))

    # os.chdir(args.project_dir)

    random.seed(args.seed)

    # Some environment variables
    device = args.device
    print(f"Using device: {device}")

    transformer_cache_dir = args.transformers_cache_dir

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
    with open(args.path_to_hf_token) as f:
        hf_token = f.read()

    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, cache_dir=transformer_cache_dir, hf_token=hf_token, hf_model=False)
    model.to(device)

    if args.dry_run:
        data_df = data_df.head(5)

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
        
    print(f'Keywords: {keywords}')

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
    print(f'words in results_df: {results_df.word.unique()}')

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
                    print('Not steering')
                    out1 = if_inference(model, tokenizer, example, device, max_new_tokens=args.max_generation_length)
                else:
                    intervention_dir = pre_computed_ivs[r['word']].to(device)

                    if args.steering == 'add_vector':
                        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir, weight=steering_weight)
                    elif args.steering == 'adjust_rs':
                        raise ValueError('Not implemented yet')
                        # hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)

                    fwd_hooks = [(tlutils.get_act_name('resid_post', layer_idx), hook_fn)]
                    #encoded_example = tokenizer.apply_chat_template(messages, return_tensors='pt').to(device)
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
    folder = f'{args.project_dir}/{args.output_path}/{args.model_name}/{args.constraint}'
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

# %%
# args = OmegaConf.load('./ifeval_experiments/config/conf.yaml')
# args['model_name'] = 'phi-3'
# run_experiment(args)
# %%
if __name__ == '__main__':
    run_experiment()
    exit(0)
# %%

# compute accuracy 
import sys
sys.path.append('./ifeval_experiments')
from ifeval_experiments.eval.evaluation_main import test_instruction_following_loose

# load out data
import json
import pandas as pd
import numpy as np
import tqdm

out_path = './ifeval_experiments/out/phi-3/single_instr/all_base_x_all_instr/instr_plus_adjust_rs_20/out.jsonl'
with open(out_path) as f:
    out_data = f.readlines()
    out_data = [json.loads(d) for d in out_data]

out_df = pd.DataFrame(out_data)

# load input data
data_path = './data/input_data.jsonl'

with open(data_path) as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df = pd.DataFrame(data)

# %%
eval_outputs = []
for i, r in tqdm.tqdm(out_df.iterrows()):
    prompt_to_response = {}
    prompt_to_response[r.prompt] = r.response
    output = test_instruction_following_loose(r, prompt_to_response)
    eval_outputs.append(output)

follow_all_instructions = [o.follow_all_instructions for o in eval_outputs]
accuracy = sum(follow_all_instructions) / len(eval_outputs)
# %%
accuracy
# %%

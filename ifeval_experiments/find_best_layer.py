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
from utils.generation_utils import generate_with_hooks, direction_projection_hook, direction_ablation_hook
from eval.evaluation_main import test_instruction_following_loose


@hydra.main(config_path='../config', config_name='find_best_layer')
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

    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    data_df = data_df[data_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]

    if args.specific_instruction:
        data_df = data_df[data_df.instruction_id_list.apply(lambda x: args.specific_instruction in x[0])]

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

    print(f'Running on: {all_instructions}')

    n_layers = model.cfg.n_layers
    if 'gemma-2-9b' in args.model_name:
        layer_range = range(n_layers // 5, n_layers, 3)
    else:
        layer_range = range(n_layers // 5, n_layers, 2)
    layer_range = [-1] + list(layer_range)

    num_all_examples = 0
    for instr in all_instructions:
        if 'language' in instr:
            num_all_examples += 1
        else:
            instr_data_df = data_df[[instr in l for l in data_df['instruction_id_list_for_eval'] ]]
            num_all_examples += min(args.n_examples_per_instruction, len(instr_data_df))

    total = num_all_examples * len(layer_range)
    p_bar = tqdm.tqdm(total=total)

    # RowObject = namedtuple('inp', data_df.iloc[0].keys())

    for instruction_type in all_instructions:
        # we are only considering one instruction type
        instr_data_df = data_df[[[instruction_type] == l for l in data_df['instruction_id_list_for_eval'] ]]
        instr_data_df.reset_index(inplace=True, drop=True)
        instr_data_df = instr_data_df.sample(n=min(args.n_examples_per_instruction, len(instr_data_df)), random_state=args.seed)
        if 'language' in instruction_type:
            instr_data_df = instr_data_df.head(1)

        instr_data_df['instruction_id_list'] = instr_data_df['instruction_id_list_og']
        instr_data_df['prompt'] = instr_data_df['model_output']


        if args.dry_run:
            instr_data_df = instr_data_df.head(2)

        if args.steering != 'none':
            # load the stored representations
            if args.model_name == 'gemma-2-2b' and args.cross_model_steering:
                print('Loading representations from gemma-2-2b INSTRUCT')
                folder = f'{args.project_dir}/representations/gemma-2-2b-it/{args.representations_folder}'
            elif args.model_name == 'gemma-2-9b' and args.cross_model_steering:
                print('Loading representations from gemma-2-9b INSTRUCT')
                folder = f'{args.project_dir}/representations/gemma-2-9b-it/{args.representations_folder}'
            else:
                folder = f'{args.project_dir}/representations/{args.model_name}/{args.representations_folder}'
            file = f'{folder}/{"".join(instruction_type).replace(":", "_")}.h5'
            # check if the file exists
            if (not os.path.exists(file)):
                raise ValueError(f"File {file} does not exist")
            else:
                results_df = pd.read_hdf(file, key='df')

                hs_instr = results_df['last_token_rs'].values
                hs_instr = torch.tensor(np.array([example_array[:, :] for example_array in list(hs_instr)]))
                hs_no_instr = results_df['last_token_rs_no_instr'].values
                hs_no_instr = torch.tensor(np.array([example_array[:, :] for example_array in list(hs_no_instr)]))

                # compute the instrution vector
                repr_diffs = hs_instr - hs_no_instr
                mean_repr_diffs = repr_diffs.mean(dim=0)
                # check where mean_repr_diffs has three dimensions
                if len(mean_repr_diffs.shape) == 3:
                    last_token_mean_diff = mean_repr_diffs[:, -1, :]
                else:
                    last_token_mean_diff = mean_repr_diffs

        for layer_idx in layer_range:
                    
            if args.steering != 'none':

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

            print(f'Running on {len(instr_data_df)} examples for instruction {instruction_type} and layer {layer_idx}')

            # Run the model on each input
            for i, r in instr_data_df.iterrows():
                if args.include_instruction:
                    example = r['model_output'] # prompt w/ instruction
                else:
                    example = r['prompt_without_instruction'] # prompt w/o instruction

                row = dict(r)
                
                if args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b':
                    example = f'Q: {example}\nA:'
                else:
                    messages = [{"role": "user", "content": example}]
                    example = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

                if ('json' in instruction_type) or ('multiple_sections' in instruction_type):
                        max_generation_length = 1024
                else:
                    max_generation_length = args.max_generation_length

                if layer_idx == -1:
                    print('Not steering')
                    if (args.model_name == 'gemma-2-2b' or args.model_name == 'gemma-2-9b'):
                        encoded_example = tokenizer(example, return_tensors='pt').to(device)
                        out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=[], max_tokens_generated=max_generation_length, decode_directly=True)
                    else:
                        out1 = if_inference(model, tokenizer, example, device, max_new_tokens=max_generation_length)
                else:
                    intervention_dir = instr_dir.to(device)

                    if args.steering == 'add_vector':
                        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir, weight=args.steering_weight)
                    elif args.steering == 'adjust_rs':
                        hook_fn = functools.partial(direction_projection_hook, direction=intervention_dir, value_along_direction=avg_proj)

                    fwd_hooks = [(tlutils.get_act_name('resid_post', layer_idx), hook_fn)]
                    #encoded_example = tokenizer.apply_chat_template(messages, return_tensors='pt').to(device)
                    encoded_example = tokenizer(example, return_tensors='pt').to(device)
                    out1 = generate_with_hooks(model, encoded_example['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=max_generation_length, decode_directly=True)
                    # if out 1 is a list, take the first element
                if isinstance(out1, list):
                    out1 = out1[0]
                                                
                row['response'] = out1

                # compute accuracy
                prompt_to_response = {}
                prompt_to_response[row['model_output']] = row['response']
                # row['prompt'] = example
                # row["instruction_id_list_for_eval"] = [instruction_type]
                # row["instruction_id_list"] = row["instruction_id_list_og"]
                # r = RowObject(*row.values())
                output = test_instruction_following_loose(r, prompt_to_response, improved_multiple_section_checker=True)
                row['follow_all_instructions'] = output.follow_all_instructions
                row['layer'] = layer_idx

                if 'no_instr' in args.data_path:
                    row['prompt_no_instr'] = row['prompt']
                    row['prompt'] = row['original_prompt']
                    row.pop('original_prompt')
                
                out_lines.append(row)
                p_bar.update(1)


    # write out_lines as jsonl
    folder = f'{args.output_path}/{args.model_name}'
    folder += f'/n_examples{args.n_examples_per_instruction}_seed{args.seed}'

    os.makedirs(folder, exist_ok=True)
    out_path = f'{folder}/out'
    if args.include_instruction:
        out_path += '_instr'
    else:
        out_path += '_no_instr'
    if args.specific_instruction:
        out_path += f'_{args.specific_instruction}'
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

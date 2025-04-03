# %%
import os
import sys
import json
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(script_dir)
sys.path.append(project_dir)

# %%
model_name = 'phi-3'
include_instructions = False
include_length_instr = False
use_perplexity = True
steering_layer = -1
steering_mode = 'adjust_rs'
length_steering = 'conciseness'
length_steering_layer = 12
length_steering_weight = 40

out_folder = f'{script_dir}/format_plus_length_out/{model_name}'
if include_instructions:
    steering_folder = 'instr'
else:
    steering_folder = 'no_instr'
if include_length_instr:
    steering_folder += '_w_length_instr'
if steering_mode != 'none':
    steering_folder += f'/{steering_mode}_{steering_layer}'
    if use_perplexity:
        steering_folder += '_perplexity'
else:
    steering_folder += '/no_steering'
if length_steering != 'none':
    steering_folder += f'_{length_steering}_L{length_steering_layer}_w{length_steering_weight}'
else:
    steering_folder += '_no_length_steering'

path_to_results = f'{out_folder}/{steering_folder}/out.jsonl'
with open(path_to_results, 'r') as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df = pd.DataFrame(results)

length_acc = results_df.response_length_sent <= results_df.length_constraint+1

print(f'{model_name} - {steering_folder}: {results_df.follow_all_instructions.mean()} - {length_acc.mean()}')

# %%

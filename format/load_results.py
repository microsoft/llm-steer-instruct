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
folder = f'{script_dir}/out'
model_name = 'phi-3'
include_instr = False
steering_mode = 'none'
steering_layer = -1

if include_instr and steering_mode != 'none':
    steering_folder = f'instr_plus_{steering_mode}_{steering_layer}_perplexity'
elif include_instr and steering_mode == 'none':
    steering_folder = 'standard'
elif not include_instr and steering_mode != 'none':
    steering_folder = f'{steering_mode}_{steering_layer}_perplexity'
elif not include_instr and steering_mode == 'none':
    steering_folder = 'no_instr'

path_to_results = f'{folder}/{model_name}/{steering_folder}/out.jsonl'
with open(path_to_results, 'r') as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df = pd.DataFrame(results)

print(f'{model_name} - {steering_folder}: {results_df.follow_all_instructions.mean()}')

# %%
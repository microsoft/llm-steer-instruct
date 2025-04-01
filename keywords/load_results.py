# %%
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(script_dir)
sys.path.append(project_dir)

# %%
folder = f'{script_dir}/out'
model_name = 'phi-3'
constraint_type = 'existence'
steering= 'add_vector'
layer = 26
weight = 40
n_examples = 7

file = f'{folder}/{model_name}/{constraint_type}/{steering}_{layer}_n_examples{n_examples}_{weight}/out.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

df_steering = pd.DataFrame(results)

steering = 'no_instr'
file = f'{folder}/{model_name}/{constraint_type}/{steering}/out.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

df_no_steering = pd.DataFrame(results)

steering = 'standard'
file = f'{folder}/{model_name}/{constraint_type}/{steering}/out.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

df_standard = pd.DataFrame(results)

steering = 'instr_plus_add_vector'
file = f'{folder}/{model_name}/{constraint_type}/{steering}_{layer}_n_examples{n_examples}_{weight}/out.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

df_instr_plus_steering = pd.DataFrame(results)

print(f'No steering: {df_no_steering.follow_all_instructions.mean()}')
print(f'Steering: {df_steering.follow_all_instructions.mean()}')
print(f'Standard: {df_standard.follow_all_instructions.mean()}')
print(f'Instr + steering: {df_instr_plus_steering.follow_all_instructions.mean()}')

# %%

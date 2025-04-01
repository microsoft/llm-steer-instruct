# %%
import os
import sys
import pandas as pd
import json
import plotly.graph_objects as go
import nltk
import plotly
import plotly.figure_factory as ff

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

# %%
constraint_type= 'at_most'
n_sent_max = 5
n_examples = 200
output_path = f'{script_dir}/out'
model_name = 'phi-3'
source_layer_idx = 12

# load data without without steering
folder_no_steering = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/no_steering_{constraint_type}'
out_path = f'{folder_no_steering}/out.jsonl'

with open(out_path) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]

results_df_no_steering = pd.DataFrame(results)

steering_type = 'conciseness'

# load data with steering
folder_steering = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/{constraint_type}_instr_plus_add_vector_{steering_type}_{source_layer_idx}'
out_path = f'{folder_steering}/out.jsonl'

with open(out_path) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]

results_df_steering = pd.DataFrame(results)

# %%
lenght_correct_no_steering = []
lenght_of_outputs_char_no_steering = []
lenght_of_outputs_sent_no_steering = []

for i, row in results_df_no_steering.iterrows():
    lenght_of_outputs_sent_no_steering.append(len(nltk.sent_tokenize(row['response'])))
    lenght_of_outputs_char_no_steering.append(len(row['response']))
    if constraint_type == 'at_most':
        lenght_correct_no_steering.append(row['length_constraint']+1 >= len(nltk.sent_tokenize(row['response'])))
    elif constraint_type == 'at_least':
        lenght_correct_no_steering.append(row['length_constraint']+1 <= len(nltk.sent_tokenize(row['response'])))
    elif constraint_type == 'exactly':
        lenght_correct_no_steering.append(row['length_constraint']+1 == len(nltk.sent_tokenize(row['response'])))
    else:
        raise ValueError('Unknown constraint type')

lenght_correct_steering = []
lenght_of_outputs_char_steering = []
lenght_of_outputs_sent_steering = []

for i, row in results_df_steering.iterrows():
    lenght_of_outputs_sent_steering.append(len(nltk.sent_tokenize(row['response'])))
    lenght_of_outputs_char_steering.append(len(row['response']))
    if constraint_type == 'at_most':
        lenght_correct_steering.append(row['length_constraint']+1 >= len(nltk.sent_tokenize(row['response'])))
    elif constraint_type == 'at_least':
        lenght_correct_steering.append(row['length_constraint']+1 <= len(nltk.sent_tokenize(row['response'])))
    elif constraint_type == 'exactly':
        lenght_correct_steering.append(row['length_constraint']+1 == len(nltk.sent_tokenize(row['response'])))

results_df_no_steering['length'] = lenght_of_outputs_sent_no_steering
results_df_no_steering['correct'] = lenght_correct_no_steering
results_df_steering['length'] = lenght_of_outputs_sent_steering
results_df_steering['correct'] = lenght_correct_steering

for length_constraint in results_df_no_steering['length_constraint'].unique():
    print(f'Length constraint: {length_constraint+1}')
    acc_no_steering = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['correct'].mean()
    acc_steering = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['correct'].mean()
    print(f'Accuracy no steering: {acc_no_steering}')
    print(f'Accuracy steering: {acc_steering}')


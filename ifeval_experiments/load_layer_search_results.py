# %%
import os
os.chdir('/home/t-astolfo/t-astolfo')
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
sys.path.append('/home/t-astolfo/t-astolfo')
sys.path.append('/home/t-astolfo/t-astolfo/ifeval_experiments')
from eval.evaluation_main import test_instruction_following_loose

# %%
folder = 'ifeval_experiments/layer_search_out'
model_name = 'mistral-7b-instruct'
# model_name = 'Qwen/Qwen2-1.5B-Instruct'
# model_name='gemma-2-9b'
# model_name = 'phi-3'
# model_name = 'Llama-2-7b-chat'
n_examples = 8
seed = 42
instr = 'instr_detectable_format:multiple_sections'
instr = 'instr'
instr = 'no_instr_lowercase'
instr = 'instr'

file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}/out_{instr}.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

results_df = pd.DataFrame(results)
# %%
# uids = results_df.uid.unique()
# results_df['layer'] = [0 for _ in range(len(results_df))]
# layers = [8, 11, 14, 17, 20, 23, 26, 29]
# for uid in uids:
#     # assign layer to each uid
#     results_df.loc[results_df.uid == uid, 'layer'] = layers
# compute mean of column follow_all_instructions grouped by layer
results_df[['layer', 'follow_all_instructions']].groupby('layer').mean()

# %%
# =============================================================================
# get instruction-specific optimal layer
# =============================================================================

all_instructions = results_df.single_instruction_id.unique()

optimal_layers = { instr: -1 for instr in all_instructions }

for instr in all_instructions:
    instr_df = results_df[results_df.single_instruction_id == instr]
    max_accuracy = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions.max()
    optimal_layer = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean()[instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions == max_accuracy].index
    print(f'Instruction: {instr} | Optimal layer: {optimal_layer}')
    optimal_layers[instr] = optimal_layer[0]
# %%
optimal_layers
# %%
# compute mean of column follow_all_instructions using optimal layer
accuracy_values = []
for instr in all_instructions:
    instr_df = results_df[results_df.single_instruction_id == instr]
    optimal_layer = optimal_layers[instr]
    accuracy_values.append(instr_df[instr_df.layer == optimal_layer].follow_all_instructions.mean())
# %%
sum(accuracy_values) / len(accuracy_values)
# %%
# plot distribution of optimal layers
optimal_layers_values = list(optimal_layers.values())
fig = px.histogram(x=optimal_layers_values, nbins=30)
fig.show()
# %%
# print instructions for which the optimal layer is -1
instrs_no_optimal_layer = []
for instr in optimal_layers:
    if optimal_layers[instr] == -1:
        print(instr)
        instrs_no_optimal_layer.append(instr)

# %%
# print some outputs
#instr = instrs_no_optimal_layer[2]
instr = [i for i in all_instructions if 'json' in i][0]
instr_df = results_df[results_df.single_instruction_id == instr]
instr_df = instr_df[instr_df.layer == 16]
for i, r in instr_df.iterrows():
    print(f"uid: {r.uid} | layer: {r.layer} | follow_all_instructions: {r.follow_all_instructions}")
    print(f'Prompt: {r.prompt}')
    print(f'Response: {r.response}')
    print('-----------------------')
# %%
# =============================================================================
# plot accuracy per layer
# =============================================================================


# make line plot of the accuracy per layer
# instruction = all_instructions[2]

fig = go.Figure()
for instruction in all_instructions:
    layer_accuracy = results_df[results_df.single_instruction_id == instruction][['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions
    fig.add_trace(go.Scatter(x=layer_accuracy.index, y=layer_accuracy, mode='lines+markers', name=instruction))

# add title
fig.update_layout(title_text='Layer Selection: Accuracy per Layer')

# add axis labels
fig.update_xaxes(title_text='Layer')
fig.update_yaxes(title_text='Accuracy')
fig.show()
# %%
# =============================================================================
# use improved section checker
# =============================================================================

model_name = 'phi-3'
n_examples = 8
seed = 42
instr = 'instr_detectable_format:multiple_sections'
instr='no_instr'

file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}/out_{instr}.jsonl'

with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

results_df = pd.DataFrame(results)
# %%
updated_scores = []
for i, r in results_df.iterrows():
    # compute accuracy
    prompt_to_response = {}
    prompt_to_response[r['model_output']] = r['response']
    # row['prompt'] = example
    # row["instruction_id_list_for_eval"] = [instruction_type]
    # row["instruction_id_list"] = row["instruction_id_list_og"]
    # r = RowObject(*row.values())
    output = test_instruction_following_loose(r, prompt_to_response, improved_multiple_section_checker=True)
    updated_scores.append(output.follow_all_instructions)
results_df['follow_all_instructions_improved'] = updated_scores
# %%
results_df[results_df.single_instruction_id.apply(lambda x: 'multiple_sec' in x)][['layer', 'follow_all_instructions_improved']].groupby('layer').mean()
# %%
results_df[['layer', 'follow_all_instructions']].groupby('layer').mean()
# %%
results_df[results_df.single_instruction_id.apply(lambda x: 'multiple_sec' in x)]
# %%

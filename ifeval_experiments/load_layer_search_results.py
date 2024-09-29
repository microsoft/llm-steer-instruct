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
model_name='gemma-2-2b-it'
# model_name = 'phi-3'
# model_name = 'Llama-2-7b-chat'
n_examples = 8
seed = 42
instr = 'instr_detectable_format:multiple_sections'
instr = 'instr'
instr = 'no_instr_lowercase'
instr = 'no_instr'

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
instr = [i for i in all_instructions if 'title' in i][0]
instr_df = results_df[results_df.single_instruction_id == instr]
instr_df = instr_df[instr_df.layer == 23]
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
# =============================================================================
# Show all results of layer search for all model and all instructions
# =============================================================================

model_names = ['phi-3', 'mistral-7b-instruct', 'gemma-2-2b-it', 'gemma-2-9b-it', ]
settings = ['instr', 'no_instr', ]
n_examples_dict = {'phi-3' : (8, 8), 'mistral-7b-instruct': (8, 10), 'gemma-2-2b-it': (8, 8), 'gemma-2-9b-it': (6, 6)}
seed = 42
all_results = {}

for model_name in model_names:
    for setting_idx, setting in enumerate(settings):
        n_examples = n_examples_dict[model_name][0]
        print(f'Processing {model_name} | {setting} | {n_examples} examples | seed {seed}')
        file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}/out_{setting}.jsonl'
        with open(file, 'r') as f:
            results = [json.loads(line) for line in f]

        results_df = pd.DataFrame(results)
        if model_name not in all_results:
            all_results[model_name] = {}
        all_results[model_name][setting] = results_df
# %%
best_layer_rows = []
for model_name in model_names:
    for setting in settings:
        results_df = all_results[model_name][setting]
        all_instructions = results_df.single_instruction_id.unique()
        optimal_layers = { instr: -1 for instr in all_instructions }

        for instr in all_instructions:
            instr_df = results_df[results_df.single_instruction_id == instr]
            max_accuracy = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions.max()
            optimal_layer = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean()[instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions == max_accuracy].index
            optimal_layers[instr] = optimal_layer[0]

            instr_df = results_df[results_df.single_instruction_id == instr]
            optimal_layer = optimal_layers[instr]
            accuracy_value = instr_df[instr_df.layer == optimal_layer].follow_all_instructions.mean()

            best_layer_rows.append({'model_name': model_name, 'setting': setting, 'instruction': instr, 'optimal_layer': optimal_layer, 'accuracy': accuracy_value})
# %%
best_layer_df = pd.DataFrame(best_layer_rows)
best_layer_df.head()

# %%
# make optimal_layer entries -1 for detectable_format:multiple_sections and detectable_format:title in the no_instr setting
best_layer_df.loc[(best_layer_df.instruction == 'detectable_format:multiple_sections') & (best_layer_df.setting == 'no_instr'), 'optimal_layer'] = -1
best_layer_df.loc[(best_layer_df.instruction == 'detectable_format:title') & (best_layer_df.setting == 'no_instr'), 'optimal_layer'] = -1

# %%
# make table where every row is an instruction and every column is a model_name_setting
table = pd.pivot_table(best_layer_df, values='optimal_layer', index='instruction', columns=['model_name', 'setting'])
table
# %%
# remove rows with 'language' in the instruction
no_language_table = table[table.index.str.contains('language')]
no_language_table
# %%
# make table with setttin == 'no_instr'
no_instr_table = no_language_table.loc[:, (slice(None), 'no_instr')]
no_instr_table

# %%
no_instr_table = no_language_table.loc[:, (slice(None), 'instr')]
no_instr_table

# %%
paper_table = no_language_table
for i,r in paper_table.iterrows():
    instruction_name = i.split(':')[-1].replace('_', ' ').title()
    if 'Language' in instruction_name:
        instruction_name = instruction_name.replace('Response ', '')
    phi_no_instr = int(r['phi-3', 'no_instr']) if r['phi-3', 'no_instr'] != -1 else '-'
    phi_instr = int(r['phi-3', 'instr']) if r['phi-3', 'instr'] != -1 else '-'

    gemma2b_no_instr = int(r['gemma-2-2b-it', 'no_instr']) if r['gemma-2-2b-it', 'no_instr'] != -1 else '-'
    gemma2b_instr = int(r['gemma-2-2b-it', 'instr']) if r['gemma-2-2b-it', 'instr'] != -1 else '-'

    mistral_no_instr = int(r['mistral-7b-instruct', 'no_instr']) if r['mistral-7b-instruct', 'no_instr'] != -1 else '-'
    mistral_instr = int(r['mistral-7b-instruct', 'instr']) if r['mistral-7b-instruct', 'instr'] != -1 else '-'

    gemma9b_no_instr = int(r['gemma-2-9b-it', 'no_instr']) if r['gemma-2-9b-it', 'no_instr'] != -1 else '-'
    gemma9b_instr = int(r['gemma-2-9b-it', 'instr']) if r['gemma-2-9b-it', 'instr'] != -1 else '-'


    print(f'{instruction_name} & {phi_no_instr} & {phi_instr} & {gemma2b_no_instr} & {gemma2b_instr} & {mistral_no_instr} & {mistral_instr} & {gemma9b_no_instr} & {gemma9b_instr} \\\\')
# %%
len(paper_table)
# %%

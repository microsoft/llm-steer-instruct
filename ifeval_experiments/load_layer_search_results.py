# %%
import os
import sys
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/ifeval_experiments')
    print('We\'re on the local machine')
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from eval.evaluation_main import test_instruction_following_loose
from collections import Counter
import re

# %%
folder = 'ifeval_experiments/layer_search_out'
model_name = 'mistral-7b-instruct'
# model_name = 'Qwen/Qwen2-1.5B-Instruct'
model_name='gemma-2-2b'
# model_name='gemma-2-9b-it'
# model_name = 'phi-3'
# model_name = 'Llama-2-7b-chat'
n_examples = 10
seed = 42
instr = 'instr_detectable_format:multiple_sections'
instr = 'instr'
# instr = 'no_instr_lowercase'
# instr = 'no_instr'

file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}/out_{instr}.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

results_df = pd.DataFrame(results)
# %%

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
    if 'language' in instruction:
        continue
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
# check for broken outputs
# =============================================================================
from transformers import AutoTokenizer
with open('./hf_token.txt') as f:
    hf_token = f.read()
if model_name == 'phi-3':
    model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'
elif model_name == 'gemma-2-2b-it':
    model_name_hf = 'google/gemma-2-2b-it'
elif model_name == 'gemma-2-9b-it':
    model_name_hf = 'google/gemma-2-9b-it'
elif model_name == 'mistral-7b-instruct':
    model_name_hf = 'mistralai/Mistral-7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name_hf, token=hf_token)

# %%

broken_outputs = []
accuracy_with_quality_check = []
for i, r in results_df.iterrows():
    # compute accuracy

    response  = r['response']
    tokens = tokenizer.tokenize(response)
    counter = Counter(tokens)
    #remove '▁the' '▁' from the counter
    if '▁the' in counter:
        del counter['▁the']
    if '▁' in counter:
        del counter['▁']
    # take the number of occurrences of the most common token
    most_common = counter.most_common(1)[0][1]
    # get most common token
    if most_common > 50:
        broken_outputs.append(1)
    else:
        # if r.single_instruction_id == 'detectable_format:multiple_sections' and r.layer == 8:
        #     print(f'layer: {r.layer}')    
        #     print(f'Broken output: {response}')
        #     # print most common token
        #     print(counter.most_common(1))
        broken_outputs.append(0)

    # fix problem with capital word frequency
    if r.single_instruction_id == 'change_case:capital_word_frequency' and r.layer == -1:
        if 'less than' in r.prompt or 'at most' in r.prompt:
            relation = 'less than'
        elif 'more than' in r.prompt or 'at least' in r.prompt:
            relation = 'at least'
        # parse the last number in the prompt
        num = int(re.findall(r'\d+', r.prompt)[-1])

        new_kwargs = [{"capital_relation": relation,"capital_frequency": num}]
        r.kwargs = new_kwargs
        prompt_to_response = {}
        prompt_to_response[r['prompt']] = r['response']
        output = test_instruction_following_loose(r, prompt_to_response)
        # print(f'=============\nchanging follow_all_instructions from {r.follow_all_instructions} to {output.follow_all_instructions}\nfor output:{r.response}\nkargs: {r.kwargs}')
        # update follow_all_instructions
        results_df.loc[i, 'follow_all_instructions'] = output.follow_all_instructions

results_df['broken_output'] = broken_outputs

# %%
# for each instruction, make a line plot of accuracy and avg broken output per layer
for instruction in all_instructions:
    if 'language' in instruction:
        continue
    fig = go.Figure()
    layer_accuracy = results_df[results_df.single_instruction_id == instruction][['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions
    fig.add_trace(go.Scatter(x=layer_accuracy.index, y=layer_accuracy, mode='lines+markers', name='accuracy'))
    layer_broken_output = results_df[results_df.single_instruction_id == instruction][['layer', 'broken_output']].groupby('layer').mean().broken_output
    fig.add_trace(go.Scatter(x=layer_broken_output.index, y=layer_broken_output, mode='lines+markers', name='broken_output'))
    fig.update_layout(title_text=f' Accuracy and Broken Output  for {instruction}')
    fig.update_xaxes(title_text='Layer')

    fig.show()

# %%
# get update optimal layers for each instruction
optimal_layers = { instr: -1 for instr in all_instructions }
new_optimal_layers = { instr: -1 for instr in all_instructions }
for instr in all_instructions:
    instr_df = results_df[results_df.single_instruction_id == instr]
    df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'broken_output']].groupby('layer').mean()
    max_accuracy = df_group_by_layer.follow_all_instructions.max()
    optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
    optimal_layers[instr] = optimal_layer[0]

    # set follow_all_instructions to in df_group_by_layer when broken_output is > 0
    df_group_by_layer.loc[df_group_by_layer.broken_output > 0, 'follow_all_instructions'] = 0
    max_accuracy = df_group_by_layer.follow_all_instructions.max()
    optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
    new_optimal_layers[instr] = optimal_layer[0]
# %%
# make df with two columns: optimal_layer and new_optimal_layer
optimal_layers_df = pd.DataFrame(list(optimal_layers.items()), columns=['instruction', 'optimal_layer'])
optimal_layers_df['new_optimal_layer'] = list(new_optimal_layers.values())

# add column with difference between optimal_layer and new_optimal_layer
optimal_layers_df['diff'] = optimal_layers_df.optimal_layer - optimal_layers_df.new_optimal_layer
optimal_layers_df

# %%
# =============================================================================
# OLD: use improved section checker
# =============================================================================

updated_scores = []
broken_outputs = []
for i, r in results_df.iterrows():
    # compute accuracy
    prompt_to_response = {}
    prompt_to_response[r['prompt']] = r['response']
    # row['prompt'] = example
    # row["instruction_id_list_for_eval"] = [instruction_type]
    # row["instruction_id_list"] = row["instruction_id_list_og"]
    # r = RowObject(*row.values())
    output = test_instruction_following_loose(r, prompt_to_response, improved_multiple_section_checker=True)

    response  = r['response']
    tokens = tokenizer.tokenize(response)
    counter = Counter(tokens)
    # take the number of occurrences of the most common token
    most_common = counter.most_common(1)[0][1]
    # get most common token
    if most_common > 50 and counter.most_common(1)[0][0] != '▁the' and counter.most_common(1)[0][0] != '▁':
        if r.follow_all_instructions:
            print(f'layer: {r.layer}')    
            print(f'Broken output: {response}')
            # print most common token
            print(counter.most_common(1))
        broken_outputs.append(1)
    else:
        broken_outputs.append(0)
    updated_scores.append(output.follow_all_instructions)
    if output.follow_all_instructions != r.follow_all_instructions:
        if r.single_instruction_id != 'detectable_format:multiple_sections':
            print(f'=============\ninstruction: {r.single_instruction_id} | layer: {r.layer} | follow_all_instructions: {r.follow_all_instructions} -> improved: {output.follow_all_instructions} | kwargs {r.kwargs} \nprompt {r.prompt}\nresponse: {r.response}=============')
results_df['follow_all_instructions_improved'] = updated_scores
results_df['broken_output'] = broken_outputs
# %%
fig = go.Figure()
for instruction in all_instructions:
    if 'language' in instruction:
        continue
    layer_accuracy = results_df[results_df.single_instruction_id == instruction][['layer', 'follow_all_instructions_improved']].groupby('layer').mean().follow_all_instructions_improved
    fig.add_trace(go.Scatter(x=layer_accuracy.index, y=layer_accuracy, mode='lines+markers', name=instruction))

# add title
fig.update_layout(title_text='Layer Selection Improved: Accuracy per Layer')

# add axis labels
fig.update_xaxes(title_text='Layer')
fig.update_yaxes(title_text='Accuracy')
fig.show()

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

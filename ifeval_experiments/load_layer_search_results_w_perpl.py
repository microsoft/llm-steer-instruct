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
import torch
from tqdm import tqdm
# %%
folder = 'ifeval_experiments/layer_search_out'
model_name = 'gemma-2-2b'
# model_name = 'Qwen/Qwen2-1.5B-Instruct'
# model_name='mistral-7b-instruct'
# model_name='gemma-2-2b-it'
# model_name = 'phi-3'
# model_name = 'Llama-2-7b-chat'
n_examples = 10
seed = 42
instr_present = 'instr'
# instr_present = 'no_instr_lowercase'
# instr_present = 'no_instr'

w_perplexity = '_with_perplexity'

file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}{w_perplexity}/out_{instr_present}.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

results_df = pd.DataFrame(results)

# %%
all_instructions = results_df.single_instruction_id.unique()

# add boolean column that is true when perplexity is low
results_df['low_perplexity'] = results_df.perplexity < 2.5

# %%
# get update optimal layers for each instruction
optimal_layers = { instr: -1 for instr in all_instructions }
new_optimal_layers = { instr: -1 for instr in all_instructions }
new_optimal_layers_perpl = { instr: -1 for instr in all_instructions }
for instr in all_instructions:
    instr_df = results_df[results_df.single_instruction_id == instr]
    df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'broken_output']].groupby('layer').mean()
    max_accuracy = df_group_by_layer.follow_all_instructions.max()
    optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
    optimal_layers[instr] = optimal_layer[0]

    # set follow_all_instructions to in df_group_by_layer when broken_output is > 0
    # df_group_by_layer.loc[df_group_by_layer.broken_output > 0, 'follow_all_instructions'] = 0
    # max_accuracy = df_group_by_layer.follow_all_instructions.max()
    # optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
    # new_optimal_layers[instr] = optimal_layer[0]

    # set follow_all_instructions to 0 in df_group_by_layer when there exists an entry with low_perplexity > 0
    df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()
    
    # get baseline low_perplexity as the df_group_by_layer.low_perplexity for layer -1
    baseline_low_perplexity = df_group_by_layer.loc[-1, 'low_perplexity']
    # baseline_low_perplexity = 0

    # get accuracy for layer -1
    accuracy_layer_minus_1 = df_group_by_layer.loc[-1, 'follow_all_instructions']

    df_group_by_layer.loc[df_group_by_layer.low_perplexity > baseline_low_perplexity, 'follow_all_instructions'] = 0

    # restore accuracy for layer -1
    df_group_by_layer.loc[-1, 'follow_all_instructions'] = accuracy_layer_minus_1

    max_accuracy = df_group_by_layer.follow_all_instructions.max()
    optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
    new_optimal_layers[instr] = optimal_layer[0]


    # get uids of low perplexity outputs for layer -1
    low_perplexity_uids = instr_df[instr_df.layer == -1][instr_df.low_perplexity].uid

    # set low_perplexity to 0 for outputs with low_perplexity_uids
    instr_df.loc[instr_df.uid.isin(low_perplexity_uids), 'low_perplexity'] = False
    instr_df.loc[instr_df.uid.isin(low_perplexity_uids), 'follow_all_instructions'] = False

    # set follow_all_instructions to 0 in df_group_by_layer when there exists an entry with low_perplexity > 0
    df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()
    
    # get baseline low_perplexity as the df_group_by_layer.low_perplexity for layer -1
    # baseline_low_perplexity = df_group_by_layer.loc[-1, 'low_perplexity']
    baseline_low_perplexity = 0

    # get accuracy for layer -1
    # accuracy_layer_minus_1 = df_group_by_layer.loc[-1, 'follow_all_instructions']

    df_group_by_layer.loc[df_group_by_layer.low_perplexity > baseline_low_perplexity, 'follow_all_instructions'] = 0

    # restore accuracy for layer -1
    # df_group_by_layer.loc[-1, 'follow_all_instructions'] = accuracy_layer_minus_1

    max_accuracy = df_group_by_layer.follow_all_instructions.max()
    optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
    new_optimal_layers_perpl[instr] = optimal_layer[0]
    

# make df with two columns: optimal_layer and new_optimal_layer
optimal_layers_df = pd.DataFrame(list(optimal_layers.items()), columns=['instruction', 'optimal_layer'])
optimal_layers_df['new_optimal_layer'] = list(new_optimal_layers.values())
optimal_layers_df['new_optimal_layer_perpl'] = list(new_optimal_layers_perpl.values())

# add column with difference between optimal_layer and new_optimal_layer
optimal_layers_df['diff'] = optimal_layers_df.new_optimal_layer - optimal_layers_df.new_optimal_layer_perpl
optimal_layers_df_no_language = optimal_layers_df[~optimal_layers_df.instruction.str.contains('language')]
optimal_layers_df_no_language



#%%
optimal_layers_df_language = optimal_layers_df[optimal_layers_df.instruction.str.contains('language')]
optimal_layers_df_language

# %%
# for each instruction, make a line plot of accuracy and avg broken output and perplexity per layer

color1= px.colors.qualitative.Plotly[2]
color2= px.colors.qualitative.Plotly[6]

pretty_instruction_names = {'detectable_format:multiple_sections': 'Multiple Sect.', 'detectable_format:title': 'Title', 'change_case:english_capital': 'English Capital', 'detectable_format:json_format': 'JSON Format', 'change_case:english_lowercase' : 'Lowercase'}

for instruction in all_instructions:
    if 'language' in instruction:
        continue
    # if 'lowercase' in instruction or 'multiple_sections' in instruction:
    #     pass
    # else:
    #     continue
    fig = go.Figure()
    layer_accuracy = results_df[results_df.single_instruction_id == instruction][['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions
    fig.add_trace(go.Scatter(x=layer_accuracy.index, y=layer_accuracy, mode='lines+markers', name='Accuracy', line=dict(color=color1)))
    # layer_broken_output = results_df[results_df.single_instruction_id == instruction][['layer', 'broken_output']].groupby('layer').mean().broken_output
    # fig.add_trace(go.Scatter(x=layer_broken_output.index, y=layer_broken_output, mode='lines+markers', name='broken_output'))
    layer_perplexity = results_df[results_df.single_instruction_id == instruction][['layer', 'low_perplexity']].groupby('layer').mean().low_perplexity
    fig.add_trace(go.Scatter(x=layer_perplexity.index, y=layer_perplexity, mode='lines+markers', name='Fraction of Low-perplexity Outputs', line=dict(color=color2)))
    
    if 'gemma' in model_name and 'json' in instruction:
        letter = '(c) '
    elif model_name == 'phi-3' and 'multiple' in instruction:
        letter = '(a) '
    elif model_name == 'phi-3' and 'lowercase' in instruction:
        letter = '(b) '
    else:
        letter = ''
    prett_name = pretty_instruction_names.get(instruction, instruction)
    fig.update_layout(title_text=f'{letter}Accuracy vs. Low Perpl.: {prett_name}')
    # make title smaller
    fig.update_layout(title=dict(font=dict(size=15)))

    fig.update_xaxes(title_text='Layer Index')
    fig.update_yaxes(title_text='Value')

    # resize plot
    fig.update_layout(width=350, height=250)

    # remove padding
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

    # move legend to the bottom
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-.65,
        xanchor="right",
        x=.85
    ))

    # if 'gemma' in model_name and 'json' in instruction:
    #     # save plot as pdf 
    #     fig.write_image(f'./plots_for_paper/format/validation/{model_name}_{instr_present}_{instruction.split(":")[-1]}.pdf')
    # elif model_name == 'phi-3' and ( 'multiple' in instruction or 'lowercase' in instruction):
    #     fig.write_image(f'./plots_for_paper/format/validation/{model_name}_{instr_present}_{instruction.split(":")[-1]}.pdf')

    fig.show()

# %%
# make scatter plot of perplexity vs broken_output
fig = px.scatter(results_df, x='perplexity', y='broken_output', hover_data=['layer', 'follow_all_instructions'])
fig.show()

# compute correlation between perplexity and broken_output
results_df[['perplexity', 'broken_output']].corr()

#%% 
# sort df by perplexity
sorted_results_df = results_df.sort_values(by='perplexity', ascending=False)
# print top 10 highest perplexity that are not broken
for i, r in sorted_results_df.iterrows():
    # if  r.perplexity > 6 and r.broken_output == 1:
    # change_case:english_capital
    # detectable_format:json_format
    if r.single_instruction_id == 'language:response_language_ur' and r.layer == 28:
        print(f'Perplexity: {r.perplexity} | Broken output: {r.broken_output} | Layer {r.layer}\nPrompt: {r.prompt} \nResponse: {r.response}\n======================\n')
    
    


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
        file = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}{w_perplexity}/out_{setting}.jsonl'
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

        # add boolean column that is true when perplexity is low
        results_df['low_perplexity'] = results_df.perplexity < 2.5


        for instr in all_instructions:
            instr_df = results_df[results_df.single_instruction_id == instr]
            # set follow_all_instructions to 0 in df_group_by_layer when there exists an entry with large_perplexity > 0
            df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()
            df_group_by_layer.loc[df_group_by_layer.low_perplexity > 0, 'follow_all_instructions'] = 0
            max_accuracy = df_group_by_layer.follow_all_instructions.max()
            optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
            optimal_layers[instr] = optimal_layer[0]

            instr_df = results_df[results_df.single_instruction_id == instr]
            optimal_layer = optimal_layers[instr]
            accuracy_value = instr_df[instr_df.layer == optimal_layer].follow_all_instructions.mean()

            best_layer_rows.append({'model_name': model_name, 'setting': setting, 'instruction': instr, 'optimal_layer': optimal_layer, 'accuracy': accuracy_value})
# %%
best_layer_df = pd.DataFrame(best_layer_rows)
best_layer_df.head()

# %%
# # make optimal_layer entries -1 for detectable_format:multiple_sections and detectable_format:title in the no_instr setting
# best_layer_df.loc[(best_layer_df.instruction == 'detectable_format:multiple_sections') & (best_layer_df.setting == 'no_instr'), 'optimal_layer'] = -1
# best_layer_df.loc[(best_layer_df.instruction == 'detectable_format:title') & (best_layer_df.setting == 'no_instr'), 'optimal_layer'] = -1

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

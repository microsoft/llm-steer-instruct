# %%
import os
if 'Users' in os.getcwd():
    os.chdir('C:\\Users\\t-astolfo\\workspace\\t-astolfo')

    print('We\'re on a Windows machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on a sandbox machine')

import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
# %%

model_name = 'Llama-2-7b-chat'
model_name = 'phi-3'
# model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_name = 'mistral-7b-instruct'
#model_name = 'gemma-2-2b-it'
single_instr = 'single_instr/all_base_x_all_instr'
mode = 'no_instr'
# mode = 'standard'
subset = ''

eval_type = 'loose'

path_to_results = f'./ifeval_experiments/out/phi-3/single_instr/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df = pd.DataFrame(results)

# path_to_results = f'./ifeval_experiments/out/{model_name}/eval_results_{eval_type}.jsonl'
# with open(path_to_results) as f:
#     results = f.readlines()
#     results = [json.loads(r) for r in results]
# results_df_steering = pd.DataFrame(results)


mode = 'adjust_rs_20'
path_to_results = f'./ifeval_experiments/out/phi-3/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_steering = pd.DataFrame(results)

model_name = 'Llama-2-7b-chat'


mode = 'standard_no_hf'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_standard = pd.DataFrame(results)

mode = 'instr_plus_adjust_rs_-1'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_instr_plus_steering = pd.DataFrame(results)

# load ./data/input_data_single_instr_no_instr.jsonl
with open('./data/input_data_single_instr_no_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
data_df = pd.DataFrame(data)
# rename prompt to prompt_no_instr
data_df = data_df.rename(columns={'prompt': 'prompt_no_instr'})

# rename original prompt to prompt
data_df = data_df.rename(columns={'original_prompt': 'prompt'})

# drop instruction_id_list
data_df = data_df.drop(columns=['instruction_id_list'])

nonparametric_only = True
if nonparametric_only:
    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    results_df = results_df[results_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df.reset_index(drop=True, inplace=True)
    results_df_steering = results_df_steering[results_df_steering.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df_steering.reset_index(drop=True, inplace=True)
    results_df_standard = results_df_standard[results_df_standard.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df_standard.reset_index(drop=True, inplace=True)
    results_df_instr_plus_steering = results_df_instr_plus_steering[results_df_instr_plus_steering.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df_instr_plus_steering.reset_index(drop=True, inplace=True)


print(f'Max length of responses: {max([len(r.split()) for r in results_df.response])}')
print(f'Max length of responses steering: {max([len(r.split()) for r in results_df_steering.response])}')
print(f'Max length of responses standard: {max([len(r.split()) for r in results_df_standard.response])}')
print(f'Max length of responses instr_plus_steering: {max([len(r.split()) for r in results_df_instr_plus_steering.response])}')

# %%
# join data_df with results_df on prompt
results_df = results_df.merge(data_df, on='prompt')
results_df_steering = results_df_steering.merge(data_df, on='prompt')
results_df_standard = results_df_standard.merge(data_df, on='prompt')
results_df_instr_plus_steering = results_df_instr_plus_steering.merge(data_df, on='prompt')


# %%
all_instruct = list(set([ item for l in results_df.instruction_id_list for item in l]))
all_categories = [i.split(':')[0] for i in all_instruct]
# %%

# join the results_df_standard with the steering results on column 'key'
analysis_df = results_df_standard.join(results_df_instr_plus_steering.set_index('key'), on='key', rsuffix='_steering')
#analysis_df = results_df.join(results_df_steering.set_index('key'), on='key', rsuffix='_steering')

analysis_df.follow_all_instructions.sum(), analysis_df.follow_all_instructions_steering.sum()
# %%
w2r = [0 for i in range(0, len(analysis_df))]
w2w = [0 for i in range(0, len(analysis_df))]
r2w = [0 for i in range(0, len(analysis_df))]
r2r = [0 for i in range(0, len(analysis_df))]

for i, r in analysis_df.iterrows():
    if r.follow_all_instructions and r.follow_all_instructions_steering:
        r2r[i] = 1
    elif r.follow_all_instructions and not r.follow_all_instructions_steering:
        r2w[i] = 1
    elif not r.follow_all_instructions and r.follow_all_instructions_steering:
        w2r[i] = 1
    elif not r.follow_all_instructions and not r.follow_all_instructions_steering:
        w2w[i] = 1

analysis_df['w2r'] = w2r
analysis_df['w2w'] = w2w
analysis_df['r2w'] = r2w
analysis_df['r2r'] = r2r

analysis_df['r2r'].sum(), analysis_df['r2w'].sum(), analysis_df['w2r'].sum(), analysis_df['w2w'].sum()
# %%

# Define custom colors
custom_colors = plotly.colors.qualitative.Set1[2], plotly.colors.qualitative.Set2[7], plotly.colors.qualitative.Set1[0], plotly.colors.qualitative.Set2[7]

w2r_fraction = analysis_df['w2r'].sum() / analysis_df.shape[0]
w2w_fraction = analysis_df['w2w'].sum() / analysis_df.shape[0]
r2w_fraction = analysis_df['r2w'].sum() / analysis_df.shape[0]
r2r_fraction = analysis_df['r2r'].sum() / analysis_df.shape[0]

# make pie chart of the distribution of the different transitions
fig = px.pie(values=[w2r_fraction, w2w_fraction, r2w_fraction, r2r_fraction], names=['Wrong -> Correct', 'Wrong -> Wrong', 'Correct -> Wrong', 'Correct -> Correct'])
# add labels
fig.update_traces(textinfo='percent+label')
fig.update_layout(title='Distribution of Accuracy Change upon Steering')
# remove legend
fig.update_layout(showlegend=False)
# remove margins
fig.update_layout(margin=dict(l=0, r=0, t=50, b=20))
# Add custom colors
fig.update_traces(marker=dict(colors=custom_colors), textinfo='percent+label')

# %%
# Define custom colors for histograms
w2r_color = plotly.colors.qualitative.Set1[2]  # Custom color for Wrong -> Correct
r2w_color = plotly.colors.qualitative.Set1[0]  # Custom color for Correct -> Wrong

from collections import Counter

from collections import Counter
import plotly.graph_objects as go

# Filter and count categories
filtered_df_w2r = analysis_df[analysis_df['w2r'] == 1]
w2r_categories = [i[0] for i in filtered_df_w2r.instruction_id_list]
w2r_counts = Counter(w2r_categories)

filtered_df_r2w = analysis_df[analysis_df['r2w'] == 1]
r2w_categories = [i[0] for i in filtered_df_r2w.instruction_id_list]
r2w_counts = Counter(r2w_categories)

# Sort categories by counts
sorted_w2r = sorted(w2r_counts.items(), key=lambda x: x[1], reverse=True)
sorted_r2w = sorted(r2w_counts.items(), key=lambda x: x[1], reverse=False)

# Extract sorted categories and counts
sorted_w2r_categories, sorted_w2r_values = zip(*sorted_w2r)
sorted_r2w_categories, sorted_r2w_values = zip(*sorted_r2w)

# Make a bar chart of the sorted categories
fig = go.Figure()
fig.add_trace(go.Bar(x=sorted_w2r_categories, y=sorted_w2r_values, name='Wrong -> Correct', marker_color=w2r_color))
fig.add_trace(go.Bar(x=sorted_r2w_categories, y=sorted_r2w_values, name='Correct -> Wrong', marker_color=r2w_color))
fig.update_layout(barmode='group')
fig.update_layout(title='W->C and C->W Changes for Different Instrucitons')
# Incline the x-axis labels
fig.update_layout(xaxis_tickangle=45)

fig.show()
# %%
# print some examples of the transitions
filtered_df = analysis_df[analysis_df['w2r'] == 1]
# filter for json instructions
# filtered_df = filtered_df[filtered_df.instruction_id_list.apply(lambda x: 'title' in x[0])]

for i, r in filtered_df.iterrows():
    print(f'Prompt: {r.prompt}')
    print(f'Instruction: {r.instruction_id_list}')
    print(f'Instruction (steering): {r.instruction_id_list_steering}')
    print(f'Output: {r.response}\n---------------------')
    print(f'Output (steering): {r.response_steering}')
    print('==========================================')
# %%
# print the max length of response and response_steering
max_length = max([len(r.response.split()) for i, r in analysis_df.iterrows()])
max_length_steering = max([len(r.response_steering.split()) for i, r in analysis_df.iterrows()])
max_length, max_length_steering
# %%
# make histogram of lengths of responses
lengths_standard = [len(r.split()) for r in analysis_df.response]
lengths_steering = [len(r.split()) for r in analysis_df.response_steering]

fig = go.Figure()
fig.add_trace(go.Histogram(x=lengths_standard, name='Standard', opacity=0.75))
fig.add_trace(go.Histogram(x=lengths_steering, name='Steering', opacity=0.75))
fig.update_layout(barmode='overlay')
fig.update_layout(title='Length of Responses')
fig.show()

# %%

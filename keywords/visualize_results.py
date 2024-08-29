# %%
import os
os.chdir('/home/t-astolfo/t-astolfo')
import json
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import re

# %%
def correct_loose_score(df):
    new_follow_all_instructions = []
    for i, r in df.iterrows():
        keyword = r.kwargs[0]['forbidden_words'][0]
        # check whether the keyword is in the response using regex
        if re.search(rf'\b{keyword.lower()}\b', r.response.lower()):
            # if r.follow_all_instructions:
                #print('Error found!!!')
                #print(f'Keyword: {keyword} | Response: {r.response}')
            new_follow_all_instructions.append(False)
        else:
            new_follow_all_instructions.append(True)

    df['follow_all_instructions'] = new_follow_all_instructions
    return df
# %%

folder = 'keywords/out'
model_name = 'phi-3'
constrain_type = 'existence'
steering= 'add_vector'
layer = 24
weight = 120
n_examples = 20

file = f'{folder}/{model_name}/{constrain_type}/{steering}_{layer}_n_examples{n_examples}_{weight}/out.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

df_steering = pd.DataFrame(results)

steering = 'no_instr'
file = f'{folder}/{model_name}/{constrain_type}/{steering}/out.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

df_no_steering = pd.DataFrame(results)

steering = 'standard'
file = f'{folder}/{model_name}/{constrain_type}/{steering}/out.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

df_standard = pd.DataFrame(results)

steering = 'instr_plus_add_vector'
file = f'{folder}/{model_name}/{constrain_type}/{steering}_{layer}_n_examples{n_examples}_{weight}/out.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

df_instr_plus_steering = pd.DataFrame(results)

# %%
print(f'No steering: {df_no_steering.follow_all_instructions.mean()}')
print(f'Steering: {df_steering.follow_all_instructions.mean()}')
print(f'Standard: {df_standard.follow_all_instructions.mean()}')
print(f'Instr + steering: {df_instr_plus_steering.follow_all_instructions.mean()}')
# %%
df_no_steering = correct_loose_score(df_no_steering)
df_steering = correct_loose_score(df_steering)
df_standard = correct_loose_score(df_standard)
df_instr_plus_steering = correct_loose_score(df_instr_plus_steering)

print(f'No steering: {df_no_steering.follow_all_instructions.mean()}')
print(f'Steering: {df_steering.follow_all_instructions.mean()}')
print(f'Standard: {df_standard.follow_all_instructions.mean()}')
print(f'Instr + steering: {df_instr_plus_steering.follow_all_instructions.mean()}')

# %%
# =============================================================================
# Make plots
# =============================================================================


# make bar plot of the accuracy values
fig = go.Figure()
fig.add_trace(go.Bar(x=['w/o instr.', 'w/o instr. + Steering', 'w/ instr.', 'w/ instr. + steering'], y=[df_no_steering.follow_all_instructions.mean(), df_steering.follow_all_instructions.mean(), df_standard.follow_all_instructions.mean(), df_instr_plus_steering.follow_all_instructions.mean()], marker_color=plotly.colors.qualitative.Plotly[4:]))
fig.update_layout(title=f'Accuracy of {model_name.title()} on Keyword Exclusion')
fig.update_yaxes(title='Accuracy')
fig.update_xaxes(title='Setting')
# set min y value to 0
fig.update_yaxes(range=[0, 1])
# resize the plot
fig.update_layout(width=400, height=300)
# remove padding
fig.update_layout(margin=dict(l=0, r=10, t=50, b=20))
fig.show()


# %%
# =============================================================================
# Error analysis
# =============================================================================

df1 = df_no_steering
df2 = df_steering

# join the two dataframes on index
analysis_df = df1.copy()
analysis_df['follow_all_instructions_steering'] = df2['follow_all_instructions']
analysis_df['follow_all_instructions_no_steering'] = df1['follow_all_instructions']
analysis_df['response_steering'] = df2['response']

w2r = [0 for i in range(0, len(analysis_df))]
w2w = [0 for i in range(0, len(analysis_df))]
r2w = [0 for i in range(0, len(analysis_df))]
r2r = [0 for i in range(0, len(analysis_df))]

for i, r in analysis_df.iterrows():
    if r.follow_all_instructions_no_steering and r.follow_all_instructions_steering:
        r2r[i] = 1
    if r.follow_all_instructions_no_steering and not r.follow_all_instructions_steering:
        r2w[i] = 1
    if not r.follow_all_instructions_no_steering and r.follow_all_instructions_steering:
        w2r[i] = 1
    if not r.follow_all_instructions_no_steering and not r.follow_all_instructions_steering:
        w2w[i] = 1

analysis_df['w2r'] = w2r
analysis_df['w2w'] = w2w
analysis_df['r2w'] = r2w
analysis_df['r2r'] = r2r
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
fig.update_traces(textinfo='value+label')
fig.update_layout(title='Distribution of Accuracy Change upon Steering')
# remove legend
fig.update_layout(showlegend=False)
# remove margins
fig.update_layout(margin=dict(l=0, r=0, t=50, b=20))
# Add custom colors
fig.update_traces(marker=dict(colors=custom_colors), textinfo='percent+label')
# %%
# %%
# print some examples of the transitions
filtered_df = analysis_df[analysis_df['r2r'] == 1]
# filter for json instructions

for i, r in filtered_df.iterrows():
    print(f'Prompt: {r.prompt}')
    print(f'Instruction: {r.instruction_id_list}')
    print(f'Kwargs: {r.kwargs}')
    #print(f'Instruction (steering): {r.instruction_id_list_steering}')
    print(f'Output: {r.response}\n---------------------')
    print(f'Output (steering): {r.response_steering}')
    print('==========================================')
# %%
# check 
c = 0
for i, r in df_instr_plus_steering.iterrows():
    keyword = r.kwargs[0]['forbidden_words'][0]
    # check whether the keyword is in the response using regex
    if re.search(rf'\b{keyword.lower()}\b', r.response.lower()) and r.follow_all_instructions:
        print('Keyword found!!!')
        print(f'Keyword: {keyword} | Response: {r.response}')
        c += 1
print(f'Number of keywords found: {c}')
# %%
# =============================================================================
# Setting 1-2: make plot with all models
# =============================================================================

# load the dataframes
model_names = ['phi-3', 'gemma-2-2b-it']
weights = {'phi-3': -150, 'gemma-2-2b-it': -200}
dfs = {}
for model_name in model_names:
    
    folder = 'keywords/out'
    constrain_type = 'forbidden'
    steering= 'add_vector'
    layer = 24
    weight = weights[model_name]
    n_examples = 20

    file = f'{folder}/{model_name}/{constrain_type}/{steering}_{layer}_n_examples{n_examples}_{weight}/out.jsonl'
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    df_steering = pd.DataFrame(results)

    steering = 'no_instr'
    file = f'{folder}/{model_name}/{constrain_type}/{steering}/out.jsonl'
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    df_no_steering = pd.DataFrame(results)

    steering = 'standard'
    file = f'{folder}/{model_name}/{constrain_type}/{steering}/out.jsonl'
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    df_standard = pd.DataFrame(results)

    steering = 'instr_plus_add_vector'
    file = f'{folder}/{model_name}/{constrain_type}/{steering}_{layer}_n_examples{n_examples}_{weight}/out.jsonl'
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    df_instr_plus_steering = pd.DataFrame(results)

    print(f'No steering: {df_no_steering.follow_all_instructions.mean()}')
    print(f'Steering: {df_steering.follow_all_instructions.mean()}')
    print(f'Standard: {df_standard.follow_all_instructions.mean()}')
    print(f'Instr + steering: {df_instr_plus_steering.follow_all_instructions.mean()}')
    
    df_no_steering = correct_loose_score(df_no_steering)
    df_steering = correct_loose_score(df_steering)
    df_standard = correct_loose_score(df_standard)
    df_instr_plus_steering = correct_loose_score(df_instr_plus_steering)

    print(f'No steering: {df_no_steering.follow_all_instructions.mean()}')
    print(f'Steering: {df_steering.follow_all_instructions.mean()}')
    print(f'Standard: {df_standard.follow_all_instructions.mean()}')
    print(f'Instr + steering: {df_instr_plus_steering.follow_all_instructions.mean()}')

    dfs[model_name] = {
    'results_df': df_no_steering,
    'results_df_steering': df_steering,
    'results_df_standard': df_standard,
    'results_df_instr_plus_steering': df_instr_plus_steering
    }
# %%
# make bar plot with all models of results_df.follow_all_instructions.mean() and results_df_steering.follow_all_instructions.mean()
df = pd.DataFrame({
    'Model': model_names,
    'Standard Inference': [dfs[model_name]['results_df'].follow_all_instructions.mean() for model_name in model_names],
    'Steering': [dfs[model_name]['results_df_steering'].follow_all_instructions.mean() for model_name in model_names],
    'w/ Instr.': [dfs[model_name]['results_df_standard'].follow_all_instructions.mean() for model_name in model_names],
    'w/ Instr. + Steering': [dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.mean() for model_name in model_names]
})

# Specify a list of colors for each 'Setting'
index = 0
colors = [px.colors.qualitative.Plotly[index], px.colors.qualitative.Plotly[index]]

# plot 'w/o Instr.' and 'w/o Instr. + Steering' in one plot
fig = go.Figure()
for i, setting in enumerate(['Standard Inference', 'Steering']):
    fig.add_trace(go.Bar(
        x=df['Model'].apply(lambda x: x.replace('-', ' ').title()),
        y=df[setting],
        name=setting,
        marker_color=colors[i],
        opacity=1 if i == 1 else 0.5
    ))

# set title
fig.update_layout(title_text='Keyword Exclusion: *without* Input Instruction')
# resize plot
fig.update_layout(width=500, height=350)


# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

# set min y to 0.5
fig.update_layout(yaxis=dict(range=[0.3, 1]))

# store plot as pdf
fig.write_image('plots/keyword_exclusion_no_instruction.pdf')

fig.show()

 # %%
# %%
# =============================================================================
# Setting 3-4: make plot with all models
# =============================================================================

# make bar plot with all models of results_df.follow_all_instructions.mean() and results_df_steering.follow_all_instructions.mean()
df = pd.DataFrame({
    'Model': model_names,
    'Standard Inference': [dfs[model_name]['results_df_standard'].follow_all_instructions.mean() for model_name in model_names],
    'Steering': [dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.mean() for model_name in model_names]
})

# Specify a list of colors for each 'Setting'
index = 3
colors = [px.colors.qualitative.Plotly[index], px.colors.qualitative.Plotly[index]]

# plot 'w/o Instr.' and 'w/o Instr. + Steering' in one plot
fig = go.Figure()
for i, setting in enumerate(['Standard Inference', 'Steering']):
    fig.add_trace(go.Bar(
        x=df['Model'].apply(lambda x: x.replace('-', ' ').title()),
        y=df[setting],
        name=setting,
        marker_color=colors[i],
        opacity=1 if i == 1 else 0.5
    ))

# set title
fig.update_layout(title_text='IFEval Format & Structure: *with* Input Instruction')
# resize plot
fig.update_layout(width=500, height=350)

# set min y to 0.5
fig.update_layout(yaxis=dict(range=[0.5, 1]))

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

# store plot as pdf
fig.write_image('plots/keyword_exclusion_with_instruction.pdf')
fig.show()
# %%

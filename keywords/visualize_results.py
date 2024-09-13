# %%
import os
import json
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import re

if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on a sandbox machine')

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
model_name = 'gemma-2-2b-it'
constrain_type = 'forbidden'
steering= 'add_vector'
layer = 24
weight = -200
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
weights = {'forbidden' : {'phi-3': -150, 'gemma-2-2b-it': -200}, 'existence': {'phi-3': 120, 'gemma-2-2b-it': -200}}
constraints = ['forbidden', 'existence']
all_dfs = {}
for model_name in model_names:
    for constraint_type in constraints:
        
        folder = 'keywords/out'
        steering= 'add_vector'
        layer = 24
        weight = weights[constraint_type][model_name]
        n_examples = 20

        # todo remove this when we have the results
        if constraint_type == 'existence' and model_name == 'gemma-2-2b-it':
            all_dfs[constraint_type][model_name] = all_dfs[constraint_type]['phi-3']
            continue

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

        if 'forbidden_words' in df_no_steering.kwargs[0]:
            df_no_steering = correct_loose_score(df_no_steering)
            df_steering = correct_loose_score(df_steering)
            df_standard = correct_loose_score(df_standard)
            df_instr_plus_steering = correct_loose_score(df_instr_plus_steering)

        print(f'Model: {model_name} | Constraint: {constraint_type}')
        print(f'No steering: {df_no_steering.follow_all_instructions.mean()}')
        print(f'Steering: {df_steering.follow_all_instructions.mean()}')
        print(f'Standard: {df_standard.follow_all_instructions.mean()}')
        print(f'Instr + steering: {df_instr_plus_steering.follow_all_instructions.mean()}')

        if constraint_type not in all_dfs:
            all_dfs[constraint_type] = {}

        all_dfs[constraint_type][model_name] = {
        'results_df': df_no_steering,
        'results_df_steering': df_steering,
        'results_df_standard': df_standard,
        'results_df_instr_plus_steering': df_instr_plus_steering
        }

        # carry out mcnemar test
        from statsmodels.stats.contingency_tables import mcnemar

        # Construct the contingency table
        base_accuracies = df_no_steering.follow_all_instructions.astype(int).values
        steering_accuracies = df_steering.follow_all_instructions.astype(int).values
        table = [[0, 0], [0, 0]]
        for i in range(len(base_accuracies)):
            table[base_accuracies[i]][steering_accuracies[i]] += 1
        
        # Perform McNemar's test
        result = mcnemar(table, exact=True, correction=True)
        print(f"NO Instr - P-value: {result.pvalue}")

        standard_accuracies = df_standard.follow_all_instructions.astype(int).values
        instr_plus_steering_accuracies = df_instr_plus_steering.follow_all_instructions.astype(int).values
        table = [[0, 0], [0, 0]]
        for i in range(len(standard_accuracies)):
            table[standard_accuracies[i]][instr_plus_steering_accuracies[i]] += 1

        # Perform McNemar's test
        result = mcnemar(table, exact=True, correction=True)


# %%
from plotly.subplots import make_subplots

# Calculate means and 95% confidence intervals for df_forbidden
dfs = all_dfs['forbidden']
df_forbidden = pd.DataFrame({
    'Model': model_names,
    'Std. Inference': [dfs[model_name]['results_df'].follow_all_instructions.mean() for model_name in model_names],
    'Steering': [dfs[model_name]['results_df_steering'].follow_all_instructions.mean() for model_name in model_names],
    'w/ Instr.': [dfs[model_name]['results_df_standard'].follow_all_instructions.mean() for model_name in model_names],
    'w/ Instr. + Steering': [dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.mean() for model_name in model_names],
    'Std. Inference Error': [1.96 * dfs[model_name]['results_df'].follow_all_instructions.std() / (len(dfs[model_name]['results_df']) ** 0.5) for model_name in model_names],
    'Steering Error': [1.96 * dfs[model_name]['results_df_steering'].follow_all_instructions.std() / (len(dfs[model_name]['results_df_steering']) ** 0.5) for model_name in model_names]
})

# Calculate means and 95% confidence intervals for df_existence
dfs = all_dfs['existence']
df_existence = pd.DataFrame({
    'Model': model_names,
    'Std. Inference': [dfs[model_name]['results_df'].follow_all_instructions.mean() for model_name in model_names],
    'Steering': [dfs[model_name]['results_df_steering'].follow_all_instructions.mean() for model_name in model_names],
    'w/ Instr.': [0 for model_name in model_names],
    'w/ Instr. + Steering': [0 for model_name in model_names],
    'Std. Inference Error': [1.96 * dfs[model_name]['results_df'].follow_all_instructions.std() / (len(dfs[model_name]['results_df']) ** 0.5) for model_name in model_names],
    'Steering Error': [1.96 * dfs[model_name]['results_df_steering'].follow_all_instructions.std() / (len(dfs[model_name]['results_df_steering']) ** 0.5) for model_name in model_names]
})

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Exclusion', 'Inclusion'))

# Specify a list of colors for each 'Setting'
index = 4
color = px.colors.qualitative.Plotly[index]

settings = ['Std. Inference', 'Steering']
# settings = ['w/ Instr.', 'w/ Instr. + Steering']

# Add traces for df_forbidden
for i, setting in enumerate(settings):
    fig.add_trace(go.Bar(
        x=df_forbidden['Model'].apply(lambda x: x.replace('-', ' ').title()),
        y=df_forbidden[setting],
        name=setting,
        marker_color=color,
        opacity=1 if i == 1 else 0.8,
        marker_pattern_shape="/" if i == 1 else "",
        showlegend=False
    ), row=1, col=1)

index = 9
color = px.colors.qualitative.Plotly[index]

# Add traces for df_existence
for i, setting in enumerate(settings):
    fig.add_trace(go.Bar(
        x=df_existence['Model'].apply(lambda x: x.replace('-', ' ').title()),
        y=df_existence[setting],
        name=setting,
        marker_color=color,
        opacity=1 if i == 1 else 0.8,
        marker_pattern_shape="/" if i == 1 else "",
        showlegend=False
    ), row=1, col=2)


# Add custom legend item for 'Std. Inference'
fig.add_trace(go.Bar(
    x=[None], y=[None],
    marker=dict(color='white', line=dict(color='black', width=1)),
    showlegend=True,
    name='Std. Inference',
    offset=-10,
    # width=0.01  # Set tiny width
), row=1, col=1)

# Add custom legend item for 'Steering' with pattern
fig.add_trace(go.Bar(
    x=[None], y=[None],
    marker=dict(color='white', pattern=dict(
            shape="/",
            fillmode="replace",
            size=10,
            solidity=0.2,
            fgcolor='black'  # Pattern color
        ), line=dict(color='black', width=1)),
    showlegend=True,
    name='Steering',
    offset=-10,
    # width=0.01  # Set tiny width
), row=1, col=2)

# Update layout
fig.update_layout(width=350, height=250)
fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

# Set y-axis range
if 'Steering' in settings:
    fig.update_yaxes(range=[0.5, 0.9], row=1, col=1)
    fig.update_yaxes(range=[0, 0.4], row=1, col=2)
else:
    fig.update_yaxes(range=[0.6, 1], row=1, col=1)
    fig.update_yaxes(range=[0.6, 1], row=1, col=2)

# incline x-axis labels
# fig.update_xaxes(tickangle=30)

# move the legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-0.3,
    xanchor='right',
    x=0.8
))

# store plot as pdf
if 'Steering' in settings:
    fig.update_layout(title_text='(a) Accuracy <b>w/o</b> Text Instructions')
    fig.write_image('plots_for_paper/keywords/no_instruction.pdf')
else:
    fig.update_layout(title_text='(b) Accuracy <b>With</b> Text Instructions')
    fig.write_image('plots_for_paper/keywords/with_instruction.pdf')
# fig.write_image('plots/keyword_exclusion_without_instruction.pdf')

fig.show()

 # %%
# %%
# =============================================================================
# Setting 3-4: make plot with all models
# =============================================================================
# Calculate means and 95% confidence intervals
df = pd.DataFrame({
    'Model': model_names,
    'Std. Inference': [dfs[model_name]['results_df_standard'].follow_all_instructions.mean() for model_name in model_names],
    'Steering': [dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.mean() for model_name in model_names],
    'Std. Inference Error': [1.96 * dfs[model_name]['results_df_standard'].follow_all_instructions.std() / (len(dfs[model_name]['results_df_standard']) ** 0.5) for model_name in model_names],
    'Steering Error': [1.96 * dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.std() / (len(dfs[model_name]['results_df_instr_plus_steering']) ** 0.5) for model_name in model_names]
})

# Specify a list of colors for each 'Setting'
index = 3
colors = [px.colors.qualitative.Plotly[index], px.colors.qualitative.Plotly[index]]

# plot 'Std. Inference' and 'Steering' in one plot
fig = go.Figure()
for i, setting in enumerate(['Std. Inference', 'Steering']):
    fig.add_trace(go.Bar(
        x=df['Model'].apply(lambda x: x.replace('-', ' ').title()),
        y=df[setting],
        name=setting,
        marker_color=colors[i],
        opacity=1 if i == 1 else 0.5,
        error_y=dict(
            type='data',
            array=df[f'{setting} Error'],
            visible=True
        )
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
# fig.write_image('plots/keyword_exclusion_with_instruction.pdf')
fig.show()
# %%
# =============================================================================
# test statistical significance
# =============================================================================

from statsmodels.stats.contingency_tables import mcnemar

model_name = 'phi-3'


# Construct the contingency table
# Let's assume n01 = 7, n10 = 3 (example values)
n00 = analysis_df['w2w'].sum()
n01 = analysis_df['w2r'].sum()
n10 = analysis_df['r2w'].sum()
n11 = analysis_df['r2r'].sum()
table = [[n00, n01], [n10, n11]]
print(table)


# Perform McNemar's test
result = mcnemar(table, exact=True, correction=True)  # exact=False for large samples

# Output the test statistic and p-value
print(f"Test statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")
# %%
from scipy.stats import wilcoxon

model1_accuracies = dfs[model_name]['results_df_standard'].follow_all_instructions.astype(int).values
model2_accuracies = dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.astype(int).values

stat, p_value = wilcoxon(model1_accuracies, model2_accuracies)
print(f"Wilcoxon statistic: {stat}, P-value: {p_value}")

model1_accuracies = dfs[model_name]['results_df'].follow_all_instructions.astype(int).values
model2_accuracies = dfs[model_name]['results_df_steering'].follow_all_instructions.astype(int).values

stat, p_value = wilcoxon(model1_accuracies, model2_accuracies)
print(f"Wilcoxon statistic: {stat}, P-value: {p_value}")
# %%

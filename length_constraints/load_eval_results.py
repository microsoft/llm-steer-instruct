# %%
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
import plotly
import plotly.figure_factory as ff
import numpy as np

os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/length_constraints')


# %%
# load 
output_path = './out'
model_name = 'phi-3'
n_sent_max = 5
n_examples = 40
include_instructions = True
steering = 'add_vector_conciseness'
steering = 'none'
constraint_type= 'at_most'
source_layer_idx = 12
apply_to_all_layers = False
dry_run = False


folder = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/'
if steering != 'none' and not include_instructions:
    folder += f'/{steering}_{source_layer_idx}'
    if apply_to_all_layers:
        folder += '_all_layers'
elif steering != 'none' and include_instructions:
    folder += f'/{constraint_type}_instr_plus_{steering}_{source_layer_idx}'
    if apply_to_all_layers:
        folder += '_all_layers'
elif steering == 'none' and include_instructions:
    folder += f'/no_steering_{constraint_type}'
else:
    folder += '/no_steering_no_instruction'
out_path = f'{folder}/out'
out_path += ('_test' if dry_run else '')
out_path +=  '.jsonl'

with open(out_path) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]

results_df = pd.DataFrame(results)
    
# %%
for i, row in results_df.tail(10).iterrows():
    print(f'Prompt: {row["prompt"]}')
    print(f'Generated: {row["response"]}')
    print()
# %%
lenght_of_outputs = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
lenght_of_outputs_char = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
lenght_of_outputs_sent = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
for i, row in results_df.iterrows():
    lenght_of_outputs[row['length_constraint']].append(len(row['response'].split()))
    lenght_of_outputs_char[row['length_constraint']].append(len(row['response']))
    lenght_of_outputs_sent[row['length_constraint']].append(len(nltk.sent_tokenize(row['response'])))
    

# %%
# make histogram, one for each length constraint
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs.items():
    fig.add_trace(go.Histogram(x=lengths, name=f'Length constraint: {length_constraint}'))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()

# %%
# make boxplot, one for each length constraint
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs.items():
    fig.add_trace(go.Box(y=lengths, name=f'Length constraint: {length_constraint}'))
# set title
fig.update_layout(title_text='Length of generated responses with different constraints')
# remove legend
fig.update_layout(showlegend=False)
fig.show()

# %%
# make boxplot, one for each length constraint
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs_sent.items():
    fig.add_trace(go.Box(y=lengths, name=f'Length constraint: {length_constraint}'))
# set title
fig.update_layout(title_text='Length of generated responses with different constraints')
# remove legend
fig.update_layout(showlegend=False)
fig.show()

# %%
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs_sent.items():
    fig.add_trace(go.Histogram(x=lengths, name=f'Length constraint: {length_constraint}', nbinsx=50))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()

# %%
# =============================================================================
# compare steering vs no steering
# =============================================================================

constraint_type= 'at_most'
# load data without instructions with and without steering
folder_no_steering = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/no_steering_{constraint_type}'
out_path = f'{folder_no_steering}/out'
out_path +=  '.jsonl'

with open(out_path) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]

results_df_no_steering = pd.DataFrame(results)

steering_type = 'conciseness'

n_examples = 40
folder_steering = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/{constraint_type}_instr_plus_add_vector_{steering_type}_{source_layer_idx}'
out_path = f'{folder_steering}/out'
out_path +=  '.jsonl'

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
    lenght_correct_no_steering.append(row['length_constraint']+1 >= len(nltk.sent_tokenize(row['response'])))

lenght_correct_steering = []
lenght_of_outputs_char_steering = []
lenght_of_outputs_sent_steering = []

for i, row in results_df_steering.iterrows():
    lenght_of_outputs_sent_steering.append(len(nltk.sent_tokenize(row['response'])))
    lenght_of_outputs_char_steering.append(len(row['response']))
    lenght_correct_steering.append(row['length_constraint']+1 >= len(nltk.sent_tokenize(row['response'])))

results_df_no_steering['length'] = lenght_of_outputs_sent_no_steering
results_df_no_steering['correct'] = lenght_correct_no_steering
results_df_steering['length'] = lenght_of_outputs_sent_steering
results_df_steering['correct'] = lenght_correct_steering
print(f' Correct no steering: {results_df_no_steering.correct.mean()}')
print(f' Correct steering: {results_df_steering.correct.mean()}')

# %%
# print some examples with the same uid
uids = results_df_no_steering['uid'].unique()
uid = uids[7]

# print the rows with the same uid
for i, r in results_df_no_steering[results_df_no_steering['uid'] == uid].iterrows():
    print(f'Prompt: {r["prompt"]}')
    print(f'Length constraint: {r["length_constraint"]}')
    print(f'Length of response in words: {len(r["response"].split())}')
    print(f'Length of response  in sentences: {len(nltk.sent_tokenize(r["response"]))}')
    print(f'Response: {r["response"]}')
    print('-------------------')
    # print response from steering
    print(f'Length of response in words: {len(results_df_steering[results_df_steering["uid"] == uid]["response"].values[0].split())}')
    print(f'Length of response in sentences: {len(nltk.sent_tokenize(results_df_steering[results_df_steering["uid"] == uid]["response"].values[0]))}')
    print(f'Response: {results_df_steering[results_df_steering["uid"] == uid]["response"].values[0]}')
    print('\n==================================\n')
# %%

# make boxplots for the length of the responses with and without steering
fig = go.Figure()
for length_constraint in results_df_no_steering['length_constraint'].unique():
    lengths = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['length']
    fig.add_trace(go.Violin(y=lengths, name=f'No steering {length_constraint}'))
    lengths = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['length']
    fig.add_trace(go.Violin(y=lengths, name=f'Steering {length_constraint}'))
# set title
fig.update_layout(title_text='Length of generated responses with and without steering')
# %%
# make bar chart with accuracy per length constraint, one column for no steering and one for steering

# Define custom colors
no_steering_color = plotly.colors.qualitative.Plotly[5]  # Custom color for No Steering
steering_color = plotly.colors.qualitative.Plotly[0]  # Custom color for Steering

# Make bar chart with accuracy per length constraint, one column for no steering and one for steering
fig = go.Figure()
first = True
for length_constraint in results_df_no_steering['length_constraint'].unique():
    # No steering
    lengths_no_steering = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['correct']
    acc_no_steering = lengths_no_steering.mean()
    fig.add_trace(go.Bar(
        x=[f'# of Sentences = {length_constraint+1}'], 
        y=[acc_no_steering], 
        name='No steering', 
        marker_color=no_steering_color,
        width=0.3,
        showlegend=first,
    ))
    

    fig.add_trace(go.Bar(
        x=[f'# of Sentences = {length_constraint+1}'], 
        y=[0], 
        name='yyy', 
        marker_color=steering_color,
        width=0.5,
        showlegend=False,
    ))

    # Steering
    lengths_steering = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['correct']
    acc_steering = lengths_steering.mean()
    fig.add_trace(go.Bar(
        x=[f'# of Sentences = {length_constraint+1}'], 
        y=[acc_steering],
        name='Steering', 
        marker_color=steering_color,
        width=0.2,
        showlegend=first,
    ))

    first = False

# Group the pairs of columns by length constraint
fig.update_layout(barmode='group')

# Add title
fig.update_layout(title_text='Accuracy Per Length Constraint: Steering vs No Steering')

# remove padding
fig.update_layout(margin=dict(l=50, r=0, t=50, b=30))

# modefiy ymin
fig.update_layout(yaxis=dict(range=[0.4, 1]))

# reshape figure
fig.update_layout(width=600, height=350)

# save theplot as pdf
# fig.write_image('../plots/accuracy_per_length_constraintno_steering.pdf')

fig.show()

# %%
# =============================================================================
# TODO make plot for paper
# =============================================================================


# Define custom colors
no_steering_color = plotly.colors.qualitative.Plotly[5]  # Custom color for No Steering
steering_color = plotly.colors.qualitative.Plotly[0]  # Custom color for Steering

# Make bar chart with accuracy per length constraint, one column for no steering and one for steering
fig = go.Figure()
first = True
for length_constraint in results_df_no_steering['length_constraint'].unique():
    # No steering
    lengths_no_steering = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['correct']
    acc_no_steering = lengths_no_steering.mean()
    fig.add_trace(go.Bar(
        x=[f'{length_constraint+1}'], 
        y=[acc_no_steering], 
        name='Std. Inference', 
        marker_color=no_steering_color,
        width=0.47,
        showlegend=first,
    ))
    

    fig.add_trace(go.Bar(
        x=[f'{length_constraint+1}'], 
        y=[0], 
        name='yyy', 
        marker_color=steering_color,
        width=0.5,
        showlegend=False,
    ))

    # Steering
    lengths_steering = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['correct']
    acc_steering = lengths_steering.mean()
    fig.add_trace(go.Bar(
        x=[f'{length_constraint+1}'], 
        y=[acc_steering],
        name='Steering', 
        marker_color=steering_color,
        width=0.25,
        showlegend=first,
    ))

    # carry out mcnemar test
    # first, count the number of correct and incorrect predictions
    no_steering_correct = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['correct'].sum()
    no_steering_incorrect = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint].shape[0] - no_steering_correct
    steering_correct = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['correct'].sum()
    steering_incorrect = results_df_steering[results_df_steering['length_constraint'] == length_constraint].shape[0] - steering_correct

    # carry out the mcnemar test
    from statsmodels.stats.contingency_tables import mcnemar
    table = [[no_steering_correct, no_steering_incorrect], [steering_correct, steering_incorrect]]
    result = mcnemar(table, exact=False, correction=True)
    print(f'Length constraint: {length_constraint+1} | p-value: {result.pvalue}')

    first = False

# Group the pairs of columns by length constraint
fig.update_layout(barmode='group')

# incline x-axis labels
# fig.update_layout(xaxis_tickangle=45)

# Add title
fig.update_layout(title_text='(b) Accuracy Per Length Constraint')

# move legend to the bottom
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.55,
    xanchor="right",
    x=0.9
))

# add label for x axis
fig.update_layout(xaxis_title='# of Sentences')


# change title font size
fig.update_layout(title_font_size=16)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# modefiy ymin
fig.update_layout(yaxis=dict(range=[0.5, 1]))

# reshape figure
fig.update_layout(width=300, height=250)


# save theplot as pdf
# fig.write_image('../plots_for_paper/length/accuracy_per_length_constraint.pdf')

fig.show()

# %%

# %%
# %%
length_constraint = 2
# make bar plot of length of responses with and without steering
fig = go.Figure()
lengths1 = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['length']
fig.add_trace(go.Histogram(x=lengths1, name=f'No steering {length_constraint}', nbinsx=50))
lengths2 = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['length']
fig.add_trace(go.Histogram(x=lengths2, name=f'Steering {length_constraint}', nbinsx=50))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.update_layout(title_text=f'Length of generated responses with and without steering for length constraint {length_constraint+1}')
fig.show()

# %%
data1 = lengths1  
data2 = lengths2

# Creating the KDE plot
fig = ff.create_distplot([data1, data2], 
                         group_labels=['No steering 2', 'Steering 2'], 
                         show_hist=False, 
                         show_rug=False)

# Updating layout for better visualization
fig.update_layout(title="KDE Plot of Generated Responses with and without Steering",
                  xaxis_title="Length",
                  yaxis_title="Density",
                  legend_title="Condition")

fig.show()

# %%

length_constraint = 2

# Sample data: replace these with your actual data arrays
lengths1 = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['length']
lengths2 = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['length']

color1 = plotly.colors.qualitative.Plotly[5]
color2 = plotly.colors.qualitative.Plotly[0]

# Creating the KDE plot
kde_fig = ff.create_distplot([lengths1, lengths2], 
                             group_labels=['No steering', 'Steering'], 
                             show_hist=False, 
                             show_rug=False,
                                colors=[color1, color2])
# Make lines thicker
for trace in kde_fig.data:
    trace.update(line=dict(width=5))  


# Creating the histogram traces
hist_trace1 = go.Histogram(x=lengths1, name='No steering', nbinsx=50, opacity=0.3, marker_color=color1, histnorm='probability', showlegend=False)
hist_trace2 = go.Histogram(x=lengths2, name='Steering', nbinsx=50, opacity=0.3, marker_color=color2, histnorm='probability', showlegend=False)

# Extracting KDE traces
kde_trace1 = kde_fig.data[0]
kde_trace2 = kde_fig.data[1]

# Creating the combined figure
fig = go.Figure(data=[hist_trace1, hist_trace2, kde_trace1, kde_trace2])

# Updating layout for better visualization
fig.update_layout(title=f'Length of generated responses for length constraint {length_constraint+1}'.title(),
                  xaxis_title='Length',
                  yaxis_title='Density',
                  barmode='overlay',
                  legend_title='Setting')

# add vertical line for the length constraint
fig.add_shape(
    dict(
        type="line",
        x0=length_constraint+1.5,
        y0=0,
        x1=length_constraint+1.5,
        y1=0.63,
        line=dict(
            color="black",
            width=3,
            dash="dash",
        ),
    )
)
# Add horizontal label to the vertical line
fig.update_layout(
    annotations=[
        dict(
            x=length_constraint + 1.6,
            y=0.55,
            xref="x",
            yref="y",
            text=f'Length constraint: {length_constraint + 1}',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            textangle=0,
        )
    ]
)

# reshape figure
fig.update_layout(width=600, height=350)

# remove padding
fig.update_layout(margin=dict(l=50, r=0, t=50, b=30))
# store the plot as pdf
# fig.write_image(f'../plots/length_distribution_constraint_{length_constraint}.pdf')

fig.show()

# %%
# =============================================================================
# TODO plot for paper
# =============================================================================



length_constraint = 4

# Sample data: replace these with your actual data arrays
lengths1 = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['length']
lengths2 = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['length']

color1 = plotly.colors.qualitative.Plotly[5]
color2 = plotly.colors.qualitative.Plotly[0]

# Creating the KDE plot
kde_fig = ff.create_distplot([lengths1, lengths2], 
                             group_labels=['Std. Inference', 'Steering'], 
                             show_hist=False, 
                             show_rug=False,
                                colors=[color1, color2])
# Make lines thicker
for trace in kde_fig.data:
    trace.update(line=dict(width=4))  


# Creating the histogram traces
hist_trace1 = go.Histogram(x=lengths1, name='Std. Inference', nbinsx=50, opacity=0.5, marker_color=color1, histnorm='probability', showlegend=False)
hist_trace2 = go.Histogram(x=lengths2, name='Steering', nbinsx=50, opacity=0.6, marker_color=color2, histnorm='probability', showlegend=False)

# Extracting KDE traces
kde_trace1 = kde_fig.data[0]
kde_trace2 = kde_fig.data[1]

# Creating the combined figure
fig = go.Figure(data=[hist_trace1, hist_trace2, kde_trace1, kde_trace2])

# Updating layout for better visualization
fig.update_layout(title=f'(c) Length: Pre- vs. Post-Steering',
                  xaxis_title='Length',
                  yaxis_title='Density',
                  barmode='overlay',
                  legend_title='')

# add vertical line for the length constraint
fig.add_shape(
    dict(
        type="line",
        x0=length_constraint+1.5,
        y0=0,
        x1=length_constraint+1.5,
        y1=0.63,
        line=dict(
            color="black",
            width=2,
            dash="dash",
        ),
    )
)
# Add horizontal label to the vertical line
fig.update_layout(
    annotations=[
        dict(
            x=length_constraint + 1.6,
            y=0.55,
            xref="x",
            yref="y",
            text=f'Length constraint: {length_constraint + 1}',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            textangle=0,
        )
    ]
)

# move legend to the bottom
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-.5,
    xanchor="right",
    x=0.89
))

# reshape figure
fig.update_layout(width=300, height=250)

# change title font size
fig.update_layout(title_font_size=16)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
# store the plot as pdf
fig.write_image(f'../plots_for_paper/length/length_distribution_constraint_{length_constraint}.pdf')

fig.show()


# %%

# %%
# =============================================================================
# error analysis
# =============================================================================

correct2wrong = [0 for _ in range(len(results_df_no_steering))]
wrong2correct = [0 for _ in range(len(results_df_no_steering))]
wrong2wrong = [0 for _ in range(len(results_df_no_steering))]
correct2correct = [0 for _ in range(len(results_df_no_steering))]

for i, row in results_df_no_steering.iterrows():
    if row['correct'] and not results_df_steering.iloc[i]['correct']:
        correct2wrong[i] = 1
    elif not row['correct'] and results_df_steering.iloc[i]['correct']:
        wrong2correct[i] = 1
    elif not row['correct'] and not results_df_steering.iloc[i]['correct']:
        wrong2wrong[i] = 1
    else:
        correct2correct[i] = 1

results_df_steering['c2w'] = correct2wrong
results_df_steering['w2c'] = wrong2correct
results_df_steering['w2w'] = wrong2wrong
results_df_steering['c2c'] = correct2correct
# %%

# Define custom colors
custom_colors = plotly.colors.qualitative.Set1[2], plotly.colors.qualitative.Set2[7], plotly.colors.qualitative.Set1[0], plotly.colors.qualitative.Set2[7]

w2c_fraction = results_df_steering['w2c'].sum() / results_df_steering.shape[0]
c2w_fraction = results_df_steering['c2w'].sum() / results_df_steering.shape[0]
w2w_fraction = results_df_steering['w2w'].sum() / results_df_steering.shape[0]
c2c_fraction = results_df_steering['c2c'].sum() / results_df_steering.shape[0]

# make pie chart of the distribution of the different transitions
fig = px.pie(values=[w2c_fraction, w2w_fraction, c2w_fraction, c2c_fraction], names=['Wrong -> Correct', 'Wrong -> Wrong', 'Correct -> Wrong', 'Correct -> Correct'])
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

# print some examples of the transitions
filtered_df = results_df_steering[results_df_steering['w2c'] == 1]

for i, r in filtered_df.iterrows():
    if 'Superman' not in results_df_no_steering.iloc[i-1].prompt:
        continue
    print(f'Prompt: {results_df_no_steering.iloc[i-1].prompt}')
    print(f'Length constraint: {r.length_constraint+1}')
    response_no_steering = results_df_no_steering.iloc[i]['response']
    print(f'Length of response in words: {len(response_no_steering.split())}')
    print(f'Length of response  in sentences: {len(nltk.sent_tokenize(response_no_steering))}')
    print(f'Output: {response_no_steering}\n---------------------')
    print(f'Output (steering): {r.response}')
    print('==========================================')


# %%
# =============================================================================
# OLD: steering results
# =============================================================================

# load 
output_path = './out'
model_name = 'phi-3'
n_sent_max = 10
n_examples = 50
include_instructions = False
steering = 'add_vector'
source_layer_idx = 16
apply_to_all_layers = False
dry_run = False


folder = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/'
if steering != 'none' and not include_instructions:
    folder += f'/{steering}_{source_layer_idx}'
    if apply_to_all_layers:
        folder += '_all_layers'
elif steering != 'none' and include_instructions:
    folder += f'/instr_plus_{steering}_{source_layer_idx}'
    if apply_to_all_layers:
        folder += '_all_layers'
elif steering == 'none' and include_instructions:
    folder += '/no_steering'
else:
    folder += '/no_steering_no_instruction'
out_path = f'{folder}/out'
out_path += ('_test' if dry_run else '')
out_path +=  '.jsonl'

with open(out_path) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]

results_df = pd.DataFrame(results)
    
# %%
for i, row in results_df.tail(10).iterrows():
    print(f'Prompt: {row["prompt"]}')
    print(f'Generated: {row["response"]}')
    print()
# %%
lenght_of_outputs = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
lenght_of_outputs_char = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
lenght_of_outputs_sent = {length_constraint: [] for length_constraint in results_df['length_constraint'].unique()}
for i, row in results_df.iterrows():
    lenght_of_outputs[row['length_constraint']].append(len(row['response'].split()))
    lenght_of_outputs_char[row['length_constraint']].append(len(row['response']))
    lenght_of_outputs_sent[row['length_constraint']].append(len(nltk.sent_tokenize(row['response'])))

steering_weights = {1: 75, 2: 60, 3: 50, 4: 40, 5: 30, 6: 20, 7: 10, 8: 5, 9: 2, 10: 1}

# %%
# make histogram, one for each length constraint
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs.items():
    fig.add_trace(go.Histogram(x=lengths, name=f'Length constraint: {length_constraint}'))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()

# %%
# make boxplot, one for each length constraint
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs.items():
    fig.add_trace(go.Box(y=lengths, name=f'Length constraint: {length_constraint}'))
# set title
fig.update_layout(title_text='Length of generated responses with different constraints')
fig.show()

# %%
# make boxplot, one for each length constraint
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs_sent.items():
    fig.add_trace(go.Box(y=lengths, name=f'Steering weight: {steering_weights[length_constraint]}'))
# set title
fig.update_layout(title_text='Length of generated responses with different steering weights')
# remove legend
fig.update_layout(showlegend=False)
fig.show()

# %%
fig = go.Figure()
for length_constraint, lengths in lenght_of_outputs_sent.items():
    fig.add_trace(go.Histogram(x=lengths, name=f'Length constraint: {length_constraint}', nbinsx=50))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()
# %%
# =============================================================================
# load data without instructions with and without steering
# =============================================================================

folder_no_steering = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/no_steering_no_instruction'
out_path = f'{folder_no_steering}/out'
out_path +=  '.jsonl'

with open(out_path) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]

results_df_no_steering = pd.DataFrame(results)

folder_steering = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/adjust_rs_length_specific_{source_layer_idx}'

out_path = f'{folder_steering}/out'
out_path +=  '.jsonl'

with open(out_path) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]

results_df_steering = pd.DataFrame(results)
# %%
lenght_of_outputs_no_steering = {length_constraint: [] for length_constraint in results_df_no_steering['length_constraint'].unique()}
lenght_of_outputs_char_no_steering = {length_constraint: [] for length_constraint in results_df_no_steering['length_constraint'].unique()}
lenght_of_outputs_sent_no_steering = {length_constraint: [] for length_constraint in results_df_no_steering['length_constraint'].unique()}
for i, row in results_df_no_steering.iterrows():
    lenght_of_outputs_sent_no_steering[row['length_constraint']].append(len(nltk.sent_tokenize(row['response'])))

lenght_of_outputs_steering = {length_constraint: [] for length_constraint in results_df_steering['length_constraint'].unique()}
lenght_of_outputs_char_steering = {length_constraint: [] for length_constraint in results_df_steering['length_constraint'].unique()}
lenght_of_outputs_sent_steering = {length_constraint: [] for length_constraint in results_df_steering['length_constraint'].unique()}
for i, row in results_df_steering.iterrows():
    lenght_of_outputs_sent_steering[row['length_constraint']].append(len(nltk.sent_tokenize(row['response'])))

# %%
per_length_acc_no_steering = {}
per_length_acc_steering = {}
for length_constraint in results_df_no_steering['length_constraint'].unique():
    lengths = lenght_of_outputs_sent_no_steering[length_constraint]
    correct_lengths = [l for l in lengths if l == length_constraint]
    per_length_acc_no_steering[length_constraint] = len(correct_lengths) / len(lengths)

    lengths = lenght_of_outputs_sent_steering[length_constraint]
    correct_lengths = [l for l in lengths if l == length_constraint]
    per_length_acc_steering[length_constraint] = len(correct_lengths) / len(lengths)
# %%
# make bar chart with accuracy per length constraint, one column for no steering and one for steering
fig = go.Figure()
fig.add_trace(go.Bar(x=list(per_length_acc_no_steering.keys()), y=list(per_length_acc_no_steering.values()), name='No steering'))
fig.add_trace(go.Bar(x=list(per_length_acc_steering.keys()), y=list(per_length_acc_steering.values()), name='Steering'))
fig.update_layout(barmode='group')
# add title
fig.update_layout(title_text='Accuracy per length constraint')

fig.show()

# %%
# =============================================================================
# visualize some results
# =============================================================================

uids = results_df['uid'].unique()

uid = uids[6]

# print the rows with the same uid
for i, r in results_df[results_df['uid'] == uid].iterrows():
    print(f'Prompt: {r["prompt"]}')
    print(f'Length constraint: {r["length_constraint"]}')
    print(f'Length of response in words: {len(r["response"].split())}')
    print(f'Length of response in sentences: {len(nltk.sent_tokenize(r["response"]))}')
    print(f'Response: {r["response"]}')
    print('-------------------')

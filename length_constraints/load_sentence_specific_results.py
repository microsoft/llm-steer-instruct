# %%
import pandas as pd
import json
import plotly.graph_objects as go
import nltk
import plotly
import plotly.figure_factory as ff
from statsmodels.stats.contingency_tables import mcnemar

# %%
constraint_type= 'at_most'
n_sent_max = 5
n_examples = 40
output_path = f'./out'
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
print(f' Correct no steering: {results_df_no_steering.correct.mean()}')
print(f' Correct steering: {results_df_steering.correct.mean()}')


# %%
# =============================================================================
# reproduce the plot from the paper
# =============================================================================

# Define colors
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
        width=0.1,
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
        marker_pattern_shape="/",
        width=0.25,
        showlegend=first,
    ))

    # carry out mcnemar test
    # first, count the number of correct and incorrect predictions
    table = [[0, 0], [0, 0]]
    for i, row in results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint].iterrows():
        if row['correct'] and not results_df_steering.iloc[i]['correct']:
            table[0][1] += 1
        elif not row['correct'] and results_df_steering.iloc[i]['correct']:
            table[1][0] += 1
        elif not row['correct'] and not results_df_steering.iloc[i]['correct']:
            table[1][1] += 1
        else:
            table[0][0] += 1

    # carry out the mcnemar test
    result = mcnemar(table, exact=False, correction=True)
    print(f'Length constraint: {length_constraint+1} | p-value: {result.pvalue}')

    first = False

# Group the pairs of columns by length constraint
fig.update_layout(barmode='group')

# Add title
fig.update_layout(title_text='Accuracy Per Length Constraint')

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
fig.update_layout(yaxis_title='Accuracy')

fig.show()

# %%
# =============================================================================
# reproduce the plot from the paper
# =============================================================================

length_constraint = 3

# Sample data: replace these with your actual data arrays
lengths1 = results_df_no_steering[results_df_no_steering['length_constraint'] == length_constraint]['length']
lengths2 = results_df_steering[results_df_steering['length_constraint'] == length_constraint]['length']

# remove outliers
lengths1 = lengths1[lengths1 < 20]
lengths2 = lengths2[lengths2 < 20]

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
fig.update_layout(title=f'Length: Pre- vs. Post-Steering',
                  xaxis_title='Length',
                  yaxis_title='Probability Density',
                  barmode='overlay',
                  legend_title='')

# add vertical line for the length constraint
fig.add_shape(
    dict(
        type="line",
        x0=length_constraint+1,
        y0=0,
        x1=length_constraint+1,
        y1=0.9,
        line=dict(
            color="black",
            width=0.5,
            dash="dash",
        ),
    )
)
# Add horizontal label to the vertical line
fig.update_layout(
    annotations=[
        dict(
            x=length_constraint + 1.6,
            y=0.8,
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

fig.show()


# %%
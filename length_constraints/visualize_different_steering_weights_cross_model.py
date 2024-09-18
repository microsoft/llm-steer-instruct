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
model_name = 'gemma-2-9b'
n_sent_max = 1
n_examples = 50
include_instructions = False
steering = 'add_vector_long-short'
# steering = 'none'
constraint_type= 'at_most'
source_layer_idx = 16
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
length_of_outputs = []
length_of_outputs_char = []
length_of_outputs_sent = []
for i, row in results_df.iterrows():
    length_of_outputs.append(len(row['response'].split()))
    length_of_outputs_char.append(len(row['response']))
    length_of_outputs_sent.append(len(nltk.sent_tokenize(row['response'])))

results_df['length_of_outputs'] = length_of_outputs
results_df['length_of_outputs_char'] = length_of_outputs_char
results_df['length_of_outputs_sent'] = length_of_outputs_sent    
# %%
# make dist plots
length_constraint = 0
filtered_df = results_df[results_df.length_constraint == length_constraint]

data = [filtered_df[filtered_df.steering_weight == i].length_of_outputs for i in filtered_df.steering_weight.unique()]
labels = filtered_df.steering_weight.unique().astype(str)
colors = plotly.colors.qualitative.Plotly

# filter data: remove outliers larger than 150
# for i, d in enumerate(data):
    # data[i] = d[d < 150]

# Creating the KDE plot
kde_fig = ff.create_distplot(data, 
                             group_labels=labels, 
                             show_hist=False, 
                             show_rug=False,
                                colors=colors,
                             )
# Make lines thicker
for trace in kde_fig.data:
    trace.update(line=dict(width=4))  

hist_traces = []
kde_traces = []
for i, d in enumerate(data):
    # Creating the histogram traces
    hist_trace1 = go.Histogram(x=d, name='No steering', nbinsx=100, opacity=0.3, marker_color=colors[i], histnorm='probability', showlegend=False)
    hist_traces.append(hist_trace1)
    kde_traces.append(kde_fig.data[i])

# # Extracting KDE traces
# kde_trace1 = kde_fig.data[0]
# kde_trace2 = kde_fig.data[1]

# Creating the combined figure
fig = go.Figure(data=hist_traces + kde_traces)


# Updating layout for better visualization
fig.update_layout(title=f'Length of generated responses for length constraint {length_constraint+1}'.title(),
                  xaxis_title='Length',
                  yaxis_title='Density',
                  barmode='overlay',
                  legend_title='Setting')

# reshape figure
fig.update_layout(width=600, height=350)

# remove padding
fig.update_layout(margin=dict(l=50, r=0, t=50, b=30))
# store the plot as pdf
# fig.write_image(f'../plots/length_distribution_constraint_{length_constraint}.pdf')

fig.show()

# %%
length_constraint = 0
max_length = 460
# Creating a histogram with a boxplot as marginal_x
plot_df = filtered_df[filtered_df.length_of_outputs < max_length]

plot_df = plot_df[plot_df.length_constraint == length_constraint]
plot_df = plot_df[plot_df.steering_weight != 60]

data = [plot_df[plot_df.steering_weight == i].length_of_outputs for i in plot_df.steering_weight.unique()]
labels = plot_df.steering_weight.unique().astype(str)
colors = plotly.colors.qualitative.Plotly

# filter data: remove outliers 
for i, d in enumerate(data):
    # compute 75th percentile
    q75, q25 = np.percentile(d, 75), np.percentile(d, 25)
    iqr = q75 - q25
    data[i] = d #[d < q75 + 1.5 * iqr]

# Creating the KDE plot
kde_fig = ff.create_distplot(data, 
                             group_labels=labels, 
                             show_hist=False, 
                             show_rug=False,
                                colors=colors,
                             )

# Make lines thicker
for trace in kde_fig.data:
    trace.update(line=dict(width=4))  


fig = px.histogram(plot_df, 
                   x='length_of_outputs', 
                   color='steering_weight', 
                   marginal='box', 
                   title=f'Length of outputs for constraint {length_constraint}', 
                   labels={'length_of_outputs': 'Length of outputs', 'steering_weight': 'Steering weight'}, 
                   nbins=100, 
                   barmode='overlay', 
                   histnorm='probability density')

# Extracting traces from the px.histogram figure
for kde_trace in kde_fig.data:
    kde_trace.update(showlegend=False)  # Disable legend for KDE traces
    fig.add_trace(kde_trace)

# Updating layout for better visualization
fig.update_layout(title=f'Length of generated responses with different Steering Weights'.title(),
                  xaxis_title='Length (# of words)',
                  yaxis_title='Density',
                  barmode='overlay',
                  legend_title='Weight')

# reshape figure
fig.update_layout(width=600, height=350)

# remove padding
fig.update_layout(margin=dict(l=50, r=0, t=50, b=30))

# set x max to max_length
# fig.update_xaxes(range=[0, max_length])

# store the plot as pdf
# fig.write_image(f'../plots/length_distribution_{model_name}_short.pdf')

fig.show()

# %%
# =============================================================================
# TODO make plot for paper 
# =============================================================================
length_constraint = 0
max_length = 200
# Creating a histogram with a boxplot as marginal_x
plot_df = filtered_df[filtered_df.length_of_outputs < max_length]

plot_df = plot_df[plot_df.length_constraint == length_constraint]
plot_df = plot_df[plot_df.steering_weight != 60]

data = [plot_df[plot_df.steering_weight == i].length_of_outputs for i in plot_df.steering_weight.unique()]
labels = plot_df.steering_weight.unique().astype(str)
colors = plotly.colors.qualitative.Plotly


# Creating the KDE plot
kde_fig = ff.create_distplot(data, 
                             group_labels=labels, 
                             show_hist=False, 
                             show_rug=False,
                                colors=colors,
                             )

# Make lines thicker
for trace in kde_fig.data:
    trace.update(line=dict(width=4))  


fig = px.histogram(plot_df, 
                   x='length_of_outputs', 
                   color='steering_weight', 
                   marginal='box', 
                   title=f'Length of outputs for constraint {length_constraint}', 
                   labels={'length_of_outputs': 'Length of outputs', 'steering_weight': 'Steering weight'}, 
                   nbins=100, 
                   barmode='overlay', 
                   histnorm='probability density')

# Extracting traces from the px.histogram figure
for kde_trace in kde_fig.data:
    kde_trace.update(showlegend=False)  # Disable legend for KDE traces
    fig.add_trace(kde_trace)

# Updating layout for better visualization
fig.update_layout(title=f'(c) Cross-model Steering: Length',
                  xaxis_title='Length (# of words)',
                  yaxis_title='Probability Density',
                  barmode='overlay',
                  legend_title='Weight')

# reshape figure
fig.update_layout(width=300, height=250)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# change title font size
fig.update_layout(title_font_size=15)

# move the legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-0.8,
    xanchor='right',
    x=0.9
))

# set x max to max_length
# fig.update_xaxes(range=[0, max_length])

# store the plot as pdf
fig.write_image(f'../plots_for_paper/cross-model/length_distribution_{model_name}_short.png', scale=5)

fig.show()

# %%
# make a histogram of the length of the outputs for each steering weight
fig = px.histogram(filtered_df, x='length_of_outputs_sent', color='steering_weight', marginal='box', title=f'Length of outputs for constraint {length_constraint}', labels={'length_of_outputs': 'Length of outputs', 'steering_weight': 'Steering weight'}, nbins=100, barmode='overlay')

fig.show()

 # %%
# print some outputs
uids = results_df.uid.unique()

length_constraint = 0
# filtered_df = results_df[results_df.length_constraint == length_constraint]
# filtered_df = results_df[results_df.length_of_outputs > 400]
# filtered_df = filtered_df[filtered_df.steering_weight == 40]

uid = uids[5]
filtered_df = results_df[results_df.uid == uid]

for i, row in filtered_df.iterrows():
    print(f'Prompt: {row["prompt"]}')
    print(f'Steering weight: {row['steering_weight']}\nLength: {row['length_of_outputs_sent']}\nGenerated: {row["response"]}')
    print('-----------------------')

# %%

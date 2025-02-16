# %%
import os
import sys
import json
import pandas as pd
import plotly.express as px
import nltk
import plotly
import plotly.figure_factory as ff

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)


# %%
# load  data
output_path = f'{script_dir}/out'
model_name = 'phi-3'
n_sent_max = 2
n_examples = 50
include_instructions = False
steering = 'add_vector_conciseness'
constraint_type= 'at_most'
source_layer_idx = 12
dry_run = False


folder = f'{output_path}/{model_name}/1-{n_sent_max}sentences_{n_examples}examples/'
if steering != 'none' and not include_instructions:
    folder += f'/{steering}_{source_layer_idx}'
elif steering != 'none' and include_instructions:
    folder += f'/{constraint_type}_instr_plus_{steering}_{source_layer_idx}'
elif steering == 'none' and include_instructions:
    folder += f'/no_steering_{constraint_type}'
else:
    folder += '/no_steering_no_instruction'
out_path = f'{folder}/out.jsonl'

with open(out_path) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]

results_df = pd.DataFrame(results)
    
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
# =============================================================================
# reproduce the plot from the paper 
# =============================================================================
length_constraint = 0
max_length = 460

length_constraint = 0
filtered_df = results_df[results_df.length_constraint == length_constraint]

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
fig.update_layout(title=f'Length vs. Steering Weights',
                  xaxis_title='Length (# of words)',
                  yaxis_title='Probability Density',
                  barmode='overlay',
                  legend_title='Weight')

# move the legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-0.75,
    xanchor='right',
    x=0.8,
    # change the font size
))

fig.show()

# %%
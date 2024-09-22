# %%
import os
os.chdir('/Users/alestolfo/workspace/llm-steer-instruct')
# %%
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly
from scipy.stats import ttest_ind, ttest_rel
import numpy as np
# %%
# Load the results
path = f'gpt-4o_evaluation/out/eval_report_length_phi/transformed_data.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]

results_df = pd.DataFrame(results)

len(results_df)
# %%
# compute per-example quality score
for i, r in results_df.iterrows():
    are_valid = r.model_answers['is_answer_valid']
    answers = r.model_answers['answers']
    score = 0
    for is_valid, answer in zip(are_valid, answers):
        if is_valid:
            score += answer
    score /= sum(are_valid)
    results_df.at[i, 'quality_score'] = score

# %%
weights = [0,1,2,3,4]
for k in results_df.key:
    print(k)
    # check that the length of the df filtered by the key is 5
    assert len(results_df[results_df.key == k]) == 5
    # set weights
    results_df.loc[results_df.key == k, 'steering_weight'] = weights

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
# load 
output_path = './length_constraints/out'
model_name = 'phi-3'
n_sent_max = 2
n_examples = 50
include_instructions = False
steering = 'add_vector_long-short'
# steering = 'none'
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

check_df = pd.DataFrame(results)


# %%
for weight in results_df.steering_weight.unique():
    print(f'Weight: {weight}')
    df_weight = results_df[results_df.steering_weight == weight]

    avg_quality_score = df_weight.quality_score.mean()
    print(f'Avg quality score: {avg_quality_score}')

    # print average length of outputs
    avg_length_of_outputs = df_weight.length_of_outputs.mean()
    print(f'Avg length of outputs: {avg_length_of_outputs}')

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
# compare outputs 
# =============================================================================
instr = [i for i in all_instructions if 'capital' in i][0]

# instr_df = joined_df[joined_df.instruction_id_list.apply(lambda x: x[0]) == instr]
instr_df = joined_df[joined_df.IFEvalQualMetric_score >= joined_df.IFEvalQualMetric_score_no_steering]

for i, r in instr_df.iterrows():
    print(f'Prompt: {r.prompt_without_instruction}')
    print(f'IFEvalQualMetric_score: {r.IFEvalQualMetric_score_no_steering}')
    print(f'Output (no steering): {r.response_no_steering}')
    print('--------------------------------')
    print(f'IFEvalQualMetric_score: {r.IFEvalQualMetric_score}')
    print(f'Output (STEERING): {r.response}')
    print('================================')

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
steering = 'eval_reports_nosteering'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]
df_instr_no_steering = pd.DataFrame(results)  

# Load the results
steering = 'Phi_noinstr_nosteering_Eval'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]
df_no_instr_no_steering = pd.DataFrame(results)  

steering = 'eval_reports_wsteering'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]
df_instr_steering = pd.DataFrame(results)

steering = 'Phi_noinstr_steering_Eval'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]
df_no_instr_steering = pd.DataFrame(results)

setting = 2
if setting == 4:
    # join the two dataframes on "key" column
    joined_df = pd.merge(df_instr_steering, df_instr_no_steering[['key', 'IFEvalQualMetric_score', 'response', 'model_answers', 'model_output']], on='key', suffixes=('', '_no_steering'))
if setting == 2:
    # join the two dataframes on "key" column
    joined_df = pd.merge(df_no_instr_steering, df_no_instr_no_steering[['key', 'IFEvalQualMetric_score', 'response', 'model_answers', 'model_output']], on='key', suffixes=('', '_no_steering'))
if setting == 3:
    # join the two dataframes on "key" column
    joined_df = pd.merge(df_instr_steering, df_instr_no_steering[['key', 'IFEvalQualMetric_score', 'response', 'model_answers', 'model_output']], on='key', suffixes=('', '_no_steering'))
if setting == 1:
    # join the two dataframes on "key" column
    joined_df = pd.merge(df_instr_no_steering, df_no_instr_no_steering[['key', 'IFEvalQualMetric_score', 'response', 'model_answers', 'model_output']], on='key', suffixes=('', '_no_steering'))
    # keep only the rows with keys that are in df_instr_steering
    joined_df = joined_df[joined_df.key.isin(df_instr_steering.key)]




# %%
# instr_with_no_steer_phi = instrs_no_optimal_layer + ['detectable_format:multiple_sections']
# filter df_no_steering to exclude instructions that were not steered
# df_steering = df_steering[df_steering.instruction_id_list_for_eval.apply(lambda x: x[0] not in instr_with_no_steer_phi)]
# %%
len(joined_df)
# %%

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
# %%
# =============================================================================
# Correlate accuracy with IFEvalQualMetric_score
# =============================================================================

# Load the results
model_name = 'phi-3'
single_instr = 'single_instr/all_base_x_all_instr'

# mode = 'standard'
subset = ''
eval_type = 'loose'

if setting == 4:
    mode = 'instr_plus_adjust_rs_-1'
elif setting == 2 or setting == 3:
    mode = 'adjust_rs_-1'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_instr_plus_steering = pd.DataFrame(results)

path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/out.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_instr_plus_steering_no_score = pd.DataFrame(results)

all_instructions = results_df_instr_plus_steering.instruction_id_list.apply(lambda x: x[0]).unique()
if setting == 4 or setting == 3:
    mode = 'standard'
elif setting == 2:
    mode = 'no_instr'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_standard = pd.DataFrame(results)

# filter results_df_standard to exclude instructions not in all_instructions
results_df_standard = results_df_standard[results_df_standard.instruction_id_list.apply(lambda x: x[0] in all_instructions)]
results_df_standard.reset_index(drop=True, inplace=True)

print(f'Length of results_df_instr_plus_steering: {len(results_df_instr_plus_steering)}')
print(f'Length of results_df_standard: {len(results_df_standard)}')

# join the two dataframes on the index
print(f'Length of joined_df: {len(joined_df)}')
joined_df = pd.merge(joined_df, results_df_instr_plus_steering[['follow_all_instructions']], left_index=True, right_index=True) 
print(f'Length of joined_df: {len(joined_df)}')
joined_df = pd.merge(joined_df, results_df_standard[['follow_all_instructions']], left_index=True, right_index=True, suffixes=('', '_standard'))
print(f'Length of joined_df: {len(joined_df)}')
if 'steering_layer' in results_df_instr_plus_steering_no_score.columns:
    joined_df = pd.merge(joined_df, results_df_instr_plus_steering_no_score[['steering_layer']], left_index=True, right_index=True)
    print(f'Length of joined_df: {len(joined_df)}')
    # replace the nans with -1 in the steering_layer column
    joined_df.steering_layer.fillna(-1, inplace=True)
joined_df['accuracy_delta'] = joined_df.follow_all_instructions.astype(int) - joined_df.follow_all_instructions_standard.astype(int)
joined_df['IFEvalQualMetric_score_delta'] = joined_df.IFEvalQualMetric_score - joined_df.IFEvalQualMetric_score_no_steering


print(f'Changes in accuracy+: {joined_df.accuracy_delta.apply(lambda x: x == 1 ).sum()}')
print(f'Changes in accuracy-: {joined_df.accuracy_delta.apply(lambda x: x == -1 ).sum()}')
print(f'No changes in accuracy: {joined_df.accuracy_delta.apply(lambda x: x == 0 ).sum()}')

print(f'Avg follow_all_instructions: {joined_df.follow_all_instructions.mean()}')
print(f'Avg follow_all_instructions_standard: {joined_df.follow_all_instructions_standard.mean()}')
# %%
# make histogram of values for steering_layer column
fig = px.histogram(joined_df, x='steering_layer', barmode='group', nbins=20)
fig.show()
# count the number of -1 values in the steering_layer column
joined_df.steering_layer.value_counts()
print(joined_df.steering_layer.value_counts())
# %%
# compute binary variable that is 1 when IFEvalQualMetric_score_delta > 0.2 and -1 when IFEvalQualMetric_score_delta < -0.2 and 0 otherwise
joined_df['IFEvalQualMetric_score_delta_bin'] = joined_df.IFEvalQualMetric_score_delta #.apply(lambda x: x if (x > 0.21 or x < -0.21) else 0)


# %%
# filter df to exclude the rows where steering_layer == -1
if setting != 1:
    filtered_df = joined_df[joined_df.steering_layer != -1]
else:
    filtered_df = joined_df
print(len(filtered_df))
# exclude 'language' instructions
filtered_df = filtered_df[filtered_df.instruction_id_list.apply(lambda x: x[0]).apply(lambda x: 'language' not in x)]
print(len(filtered_df))
# filtered_df = filtered_df[filtered_df.instruction_id_list.apply(lambda x: 'english_cap' in x[0])] 
# filtered_df = filtered_df[filtered_df.accuracy_delta == 1]
# make histograms of the IFEvalQualMetric_score column, one for follow_all_instructions == True and one for follow_all_instructions == False
fig = px.histogram(filtered_df, x='IFEvalQualMetric_score_delta_bin', barmode='group', nbins=20)
# add title
if setting == 4:
    fig.update_layout(title_text='Output Quality Score <b>w/</b> Instr.')
elif setting == 2:
    fig.update_layout(title_text='Output Quality Score <b>w/o</b> Instr.')
elif setting == 1:
    fig.update_layout(title_text='Output Quality Score <b>w/o</b> Steering')
# add axis labels
fig.update_xaxes(title_text='Delta in Quality Score')
fig.update_yaxes(title_text='Count')

# resize the figure
fig.update_layout(
    width=300,
    height=250,
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=10, t=50, b=0)
)

line_color = 'black' #plotly.colors.qualitative.Plotly[1]

# add vertical line at the mean
fig.add_shape(
    dict(
        type='line',
        x0=filtered_df.IFEvalQualMetric_score_delta_bin.mean(),
        y0=0,
        x1=filtered_df.IFEvalQualMetric_score_delta_bin.mean(),
        y1=filtered_df.IFEvalQualMetric_score_delta_bin.value_counts().max()+10,
        line=dict(color=line_color, width=2, dash='dash')
    )
)

# add horizontal text at the mean
fig.add_annotation(
    x=filtered_df.IFEvalQualMetric_score_delta_bin.mean()-0.30,
    y=filtered_df.IFEvalQualMetric_score_delta_bin.value_counts().max()+5,
    text=f'Mean: {filtered_df.IFEvalQualMetric_score_delta_bin.mean():.2f}',
    showarrow=False,
    arrowhead=1,
    arrowcolor='red',
    arrowwidth=2,
    arrowsize=1,
    ax=-60,
    ay=+40,
    font=dict(color=line_color, size=12)
)

# store the plot as pdf
# if setting == 4:
#     fig.write_image('./plots_for_paper/quality_score/change_phi3_instr.pdf')
# elif setting == 2:
#     fig.write_image('./plots_for_paper/quality_score/change_phi3_no_instr.pdf')
# elif setting == 1:
#     fig.write_image('./plots_for_paper/quality_score/change_phi3_both_no_steering.pdf')

fig.show()
# %%
# =============================================================================
# Per-instruction scores
# =============================================================================

# compute the avg IFEvalQualMetric_score per instruction
all_instructions = filtered_df.instruction_id_list.apply(lambda x: x[0]).unique()

avg_scores_no_steering = { instr : filtered_df[filtered_df.instruction_id_list.apply(lambda x: x[0]) == instr].IFEvalQualMetric_score_no_steering.mean() for instr in all_instructions }
avg_scores_steering = { instr : filtered_df[filtered_df.instruction_id_list.apply(lambda x: x[0]) == instr].IFEvalQualMetric_score.mean() for instr in all_instructions }

# sort the dictionaries alphabetically
avg_scores_no_steering = dict(sorted(avg_scores_no_steering.items()))
avg_scores_steering = dict(sorted(avg_scores_steering.items()))

# %%
color1 = plotly.colors.qualitative.Plotly[0]
color2 = plotly.colors.qualitative.Plotly[2]

# make bar plot of the avg scores
fig = go.Figure()
x_labels = [k.split(':')[1].replace('_', ' ').title() for k in avg_scores_no_steering.keys()] 
fig.add_trace(go.Bar(x=x_labels, y=list(avg_scores_no_steering.values()), name='No steering', marker_color=color1))
fig.add_trace(go.Bar(x=x_labels, y=list(avg_scores_steering.values()), name='Steering', marker_color=color2))
fig.update_layout(barmode='group', xaxis_tickangle=-45)

# add axis labels
fig.update_xaxes(title_text='Instruction')
fig.update_yaxes(title_text='Avg Qual. Score')

# reshape the figure
fig.update_layout(
    width=500,
    height=400,
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=0)
)

# incline the x-axis labels
fig.update_xaxes(tickangle=45)

# add title
if w_instruction:
    fig.update_layout(title_text='Per-instruction Output Quality Score <b>w/</b> Instr.')
else:
    fig.update_layout(title_text='Per-instruction Output Quality Score <b>w/o</b> Instr.')

fig.show()

# store the plot as pdf
if w_instruction:
    fig.write_image('./plots_for_paper/quality_score/per_instr_phi3_instr.pdf')
else:
    fig.write_image('./plots_for_paper/quality_score/per_instr_phi3_no_instr.pdf')

# %%
quality_scores_on_acc_increase = filtered_df[filtered_df.accuracy_delta == 1 ].IFEvalQualMetric_score_delta
print(f'Avg IFEvalQualMetric_score_delta for accuracy increase: {quality_scores_on_acc_increase.mean()}')

# make t-test to check if the difference in quality scores is significant
quality_scores_on_acc_decrease = filtered_df[filtered_df.accuracy_delta == -1 ].IFEvalQualMetric_score_delta
ttest_ind(quality_scores_on_acc_increase, np.zeros(len(quality_scores_on_acc_increase)))
print(f' T-test results: {ttest_rel(filtered_df.IFEvalQualMetric_score_delta, np.zeros(len(filtered_df)))}')

# carry out paired t-test between the columns IFEvalQualMetric_score and IFEvalQualMetric_score_no_steering
ttest_rel(filtered_df.IFEvalQualMetric_score, filtered_df.IFEvalQualMetric_score_no_steering)
print(f' T-test results: {ttest_rel(filtered_df.IFEvalQualMetric_score, filtered_df.IFEvalQualMetric_score_no_steering)}')

# %%
print(filtered_df.IFEvalQualMetric_score.mean())
print(filtered_df.IFEvalQualMetric_score_no_steering.mean())
# %%
# =============================================================================
# Compare the outputs for instructions with no steering
# =============================================================================

# compute the avg IFEvalQualMetric_score per instruction
all_instructions = filtered_df.instruction_id_list.apply(lambda x: x[0]).unique()

avg_scores_no_steering = { instr : filtered_df[filtered_df.instruction_id_list.apply(lambda x: x[0]) == instr].IFEvalQualMetric_score_no_steering.mean() for instr in all_instructions }
avg_scores_steering = { instr : filtered_df[filtered_df.instruction_id_list.apply(lambda x: x[0]) == instr].IFEvalQualMetric_score.mean() for instr in all_instructions }

# sort the dictionaries alphabetically
avg_scores_no_steering = dict(sorted(avg_scores_no_steering.items()))
avg_scores_steering = dict(sorted(avg_scores_steering.items()))

color1 = plotly.colors.qualitative.Plotly[0]
color2 = plotly.colors.qualitative.Plotly[2]

# make bar plot of the avg scores
fig = go.Figure()
x_labels = [k.split(':')[1].replace('_', ' ').title() for k in avg_scores_no_steering.keys()] 
fig.add_trace(go.Bar(x=x_labels, y=list(avg_scores_no_steering.values()), name='No steering', marker_color=color1))
fig.add_trace(go.Bar(x=x_labels, y=list(avg_scores_steering.values()), name='Steering', marker_color=color2))
fig.update_layout(barmode='group', xaxis_tickangle=-45)

# add axis labels
fig.update_xaxes(title_text='Instruction')
fig.update_yaxes(title_text='Avg Output Quality Score')

# add title
fig.update_layout(title_text='Avg Output Quality Score per Instruction')

fig.show()

# %%
# =============================================================================
# get bar chart for overall quality score change
# =============================================================================


# compute the avg IFEvalQualMetric_score per instruction
all_instructions = filtered_df.instruction_id_list.apply(lambda x: x[0]).unique()

avg_scores_no_steering = { instr : filtered_df[filtered_df.instruction_id_list.apply(lambda x: x[0]) == instr].IFEvalQualMetric_score_no_steering.mean() for instr in all_instructions }
avg_scores_steering = { instr : filtered_df[filtered_df.instruction_id_list.apply(lambda x: x[0]) == instr].IFEvalQualMetric_score.mean() for instr in all_instructions }

# sort the dictionaries alphabetically
avg_scores_no_steering = dict(sorted(avg_scores_no_steering.items()))
avg_scores_steering = dict(sorted(avg_scores_steering.items()))

color1 = plotly.colors.qualitative.Plotly[0]
color2 = plotly.colors.qualitative.Plotly[2]

# make bar plot of the avg scores
fig = go.Figure()
x_labels = ['No steering', 'Steering']
fig.add_trace(go.Bar(x=x_labels, y=[filtered_df.IFEvalQualMetric_score_no_steering.mean(), filtered_df.IFEvalQualMetric_score.mean()], marker_color=color1))
fig.update_layout(barmode='group', xaxis_tickangle=-45)

# add axis labels
fig.update_xaxes(title_text='Instruction')
fig.update_yaxes(title_text='Avg Output Quality Score')

# add title
fig.update_layout(title_text='Avg Output Quality Score per Instruction')

# resize the figure
fig.update_layout(
    width=500,
    height=300,
)

fig.show()
# %%


# %%
# filter df to exclude the rows where steering_layer != -1
filtered_df = joined_df[joined_df.steering_layer == -1]
# make histograms of the IFEvalQualMetric_score column, one for follow_all_instructions == True and one for follow_all_instructions == False
fig = px.histogram(filtered_df, x='IFEvalQualMetric_score_delta', color='accuracy_delta', barmode='group')

# add title 
fig.update_layout(title_text='Distribution of IFEvalQualMetric_score_delta for instructions with no steering'.title())

# add axis labels
fig.update_xaxes(title_text='IFEvalQualMetric_score_delta')
fig.update_yaxes(title_text='Count')

fig.show()

# %%
#example_df = joined_df[joined_df.instruction_id_list.apply(lambda x: 'multiple' in x[0])] 
example_df = joined_df[joined_df.IFEvalQualMetric_score_delta <= -0.5]

for i, r in example_df.iterrows():
    print(f'Prompt: {r.original_prompt_no_instr}')
    print(f'response (STEERING): {r.model_output_no_steering}')
    print(f'answers: {r.model_answers_no_steering}')
    print(f'IFEvalQualMetric_score: {r.IFEvalQualMetric_score_no_steering}')
    print(f'Output (no steering): {r.response_no_steering}')
    print('--------------------------------')
    print(f'IFEvalQualMetric_score (STEERING): {r.IFEvalQualMetric_score}')
    print(f'response (STEERING): {r.model_output}')
    print(f'answers (STEERING): {r.model_answers}')
    print(f'Output (STEERING): {r.response}')
    print('================================')
# %%
df_for_csv = joined_df[['original_prompt_no_instr', 'instruction_id_list', 'prompt_no_instr_evalq', 'model_output_no_steering', 'model_answers_no_steering', 'IFEvalQualMetric_score_no_steering', 'response_no_steering', 'IFEvalQualMetric_score', 'model_output', 'model_answers', 'response', 'IFEvalQualMetric_score_delta', 'accuracy_delta', 'follow_all_instructions', 'follow_all_instructions_standard']]

# rename original_prompt_no_instr to prompt_w_instr
df_for_csv.rename(columns={'original_prompt_no_instr': 'prompt_w_instr'}, inplace=True)

# rename prompt_no_instr_evalq to prompt_no_instr
df_for_csv.rename(columns={'prompt_no_instr_evalq': 'prompt_no_instr'}, inplace=True)

# %%
df_for_csv.to_csv('quality_score.csv')
# store also as jsonl
df_for_csv.to_json('quality_score.jsonl', orient='records', lines=True)
# %%
# compute correlation between IFEvalQualMetric_score_delta and accuracy_delta
joined_df[['IFEvalQualMetric_score_delta', 'accuracy_delta']].corr()
# %%
# compute t-test to check if the difference in quality scores is significant when accuracy increases
quality_scores_on_acc_increase = joined_df[joined_df.accuracy_delta == 1 ].IFEvalQualMetric_score_delta
ttest_ind(quality_scores_on_acc_increase, np.zeros(len(quality_scores_on_acc_increase)))

# %%
# =============================================================================
# todo compare no instr w/ steering and instr w/o steering
# =============================================================================

# Load the results
steering = 'Phi_noinstr_steering_Eval'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]

df_steering_no_instr = pd.DataFrame(results)  

steering = 'eval_reports_nosteering'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]
df_no_steering_with_instr = pd.DataFrame(results)

# load df for no steering without instr
steering = 'Phi_noinstr_nosteering_Eval'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]
df_no_steering_no_instr = pd.DataFrame(results)

# %%
# join the two dataframes on "key" column
joined_steering_df = pd.merge(df_steering_no_instr, df_no_steering_with_instr[['key', 'IFEvalQualMetric_score', 'response', 'model_answers', 'model_output']], on='key', suffixes=('', '_no_steering_w_instr'))
joined_steering_df = pd.merge(joined_steering_df, df_no_steering_no_instr[['key', 'IFEvalQualMetric_score', 'response', 'model_answers', 'model_output']], on='key', suffixes=('', '_no_steering_no_instr'))

# %%
mode = 'adjust_rs_-1'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/out.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_instr_plus_steering_no_score = pd.DataFrame(results)

all_instructions = results_df_instr_plus_steering.instruction_id_list.apply(lambda x: x[0]).unique()
# %%
joined_steering_df = pd.merge(joined_steering_df, results_df_instr_plus_steering_no_score[['steering_layer']], left_index=True, right_index=True)

# %%
# print the two IFEvalQualMetric_score means
print(f'IFEvalQualMetric_score mean (w/o steering, no instr): {joined_steering_df.IFEvalQualMetric_score_no_steering_no_instr.mean()}')
print(f'IFEvalQualMetric_score mean (w/ steering, no instr): {joined_steering_df.IFEvalQualMetric_score.mean()}')
print(f'IFEvalQualMetric_score mean (w/o steering, w/ instr): {joined_steering_df.IFEvalQualMetric_score_no_steering_w_instr.mean()}')


only_steering_df = joined_steering_df[joined_steering_df.steering_layer != -1]
# print the two IFEvalQualMetric_score means
print('After filtering out the rows where steering_layer == -1')
print(f'IFEvalQualMetric_score mean (w/o steering, no instr): {only_steering_df.IFEvalQualMetric_score_no_steering_no_instr.mean()}')
print(f'IFEvalQualMetric_score mean (w/ steering, no instr): {only_steering_df.IFEvalQualMetric_score.mean()}')
print(f'IFEvalQualMetric_score mean (w/o steering, w/ instr): {only_steering_df.IFEvalQualMetric_score_no_steering_w_instr.mean()}')
# %%

# Calculate means and standard errors
means = [
    only_steering_df.IFEvalQualMetric_score_no_steering_no_instr.mean(),
    only_steering_df.IFEvalQualMetric_score.mean(),
    only_steering_df.IFEvalQualMetric_score_no_steering_w_instr.mean()
]

errors = [
    only_steering_df.IFEvalQualMetric_score_no_steering_no_instr.sem(),
    only_steering_df.IFEvalQualMetric_score.sem(),
    only_steering_df.IFEvalQualMetric_score_no_steering_w_instr.sem()
]

# Make bar plot of the avg scores with error bars
fig = go.Figure()
x_labels = ['No steering, no instr', 'Steering, no instr', 'No steering, w/ instr']
fig.add_trace(go.Bar(
    x=x_labels,
    y=means,
    error_y=dict(type='data', array=errors),
    marker_color=color1
))

# Update layout
fig.update_layout(barmode='group', xaxis_tickangle=-45)

# Add axis labels
fig.update_xaxes(title_text='Setting')
fig.update_yaxes(title_text='Avg Output Quality Score')

# Add title
fig.update_layout(title_text='Avg Output Quality Score per Setting')

# resize the figure
fig.update_layout(
    width=350,
    height=300,
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.show()

# %%
# compute t-test to check if the difference in quality scores is significant
ttest_rel(only_steering_df.IFEvalQualMetric_score, only_steering_df.IFEvalQualMetric_score_no_steering_w_instr, alternative='less')

# %%

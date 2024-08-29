# %%
import os
os.chdir('/home/t-astolfo/t-astolfo')
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
steering = 'Phi_noinstr_nosteering_Eval'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]

df_no_steering = pd.DataFrame(results)  

steering = 'eval_reports_wsteering'
steering = 'Phi_noinstr_steering_Eval'
path = f'gpt-4o_evaluation/out/{steering}/eval_report/metric_results.jsonl'
with open(path, 'r') as f:
    results = [json.loads(line) for line in f]

df_steering = pd.DataFrame(results)

# %%
# instr_with_no_steer_phi = instrs_no_optimal_layer + ['detectable_format:multiple_sections']
# filter df_no_steering to exclude instructions that were not steered
# df_steering = df_steering[df_steering.instruction_id_list_for_eval.apply(lambda x: x[0] not in instr_with_no_steer_phi)]
# %%
len(df_steering)
# %%
# join the two dataframes on "key" column
joined_df = pd.merge(df_steering, df_no_steering[['key', 'IFEvalQualMetric_score', 'response', 'model_answers']], on='key', suffixes=('', '_no_steering'))
# %%
print(joined_df.IFEvalQualMetric_score.mean())
print(joined_df.IFEvalQualMetric_score_no_steering.mean())
# %%
# compute the avg IFEvalQualMetric_score per instruction
all_instructions = joined_df.instruction_id_list.apply(lambda x: x[0]).unique()

avg_scores_no_steering = { instr : joined_df[joined_df.instruction_id_list.apply(lambda x: x[0]) == instr].IFEvalQualMetric_score_no_steering.mean() for instr in all_instructions }
avg_scores_steering = { instr : joined_df[joined_df.instruction_id_list.apply(lambda x: x[0]) == instr].IFEvalQualMetric_score.mean() for instr in all_instructions }

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
fig.update_yaxes(title_text='Avg Output Quality Score')

# add title
fig.update_layout(title_text='Avg Output Quality Score per Instruction')

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

mode = 'instr_plus_adjust_rs_-1'
mode = 'adjust_rs_-1'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_instr_plus_steering = pd.DataFrame(results)

path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/out_w_steering_layer.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_instr_plus_steering_no_score = pd.DataFrame(results)

all_instructions = results_df_instr_plus_steering.instruction_id_list.apply(lambda x: x[0]).unique()

mode = 'standard'
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

# %%
# join the two dataframes on the index
joined_df = pd.merge(df_steering, df_no_steering[['key', 'IFEvalQualMetric_score', 'response', 'model_answers']], on='key', suffixes=('', '_no_steering'))
print(f'Length of joined_df: {len(joined_df)}')
joined_df = pd.merge(joined_df, results_df_instr_plus_steering[['follow_all_instructions']], left_index=True, right_index=True) 
print(f'Length of joined_df: {len(joined_df)}')
joined_df = pd.merge(joined_df, results_df_standard[['follow_all_instructions']], left_index=True, right_index=True, suffixes=('', '_standard'))
print(f'Length of joined_df: {len(joined_df)}')
joined_df = pd.merge(joined_df, results_df_instr_plus_steering_no_score[['steering_layer']], left_index=True, right_index=True)
print(f'Length of joined_df: {len(joined_df)}')
joined_df['accuracy_delta'] = joined_df.follow_all_instructions.astype(int) - joined_df.follow_all_instructions_standard.astype(int)
joined_df['IFEvalQualMetric_score_delta'] = joined_df.IFEvalQualMetric_score - joined_df.IFEvalQualMetric_score_no_steering
# replace the nans with -1 in the steering_layer column
joined_df.steering_layer.fillna(-1, inplace=True)

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
# filter df to exclude the rows where steering_layer == -1
filtered_df = joined_df[joined_df.steering_layer != -1]
# filtered_df = filtered_df[filtered_df.instruction_id_list.apply(lambda x: 'english_cap' in x[0])] 
# filtered_df = filtered_df[filtered_df.accuracy_delta == 1]
# make histograms of the IFEvalQualMetric_score column, one for follow_all_instructions == True and one for follow_all_instructions == False
fig = px.histogram(filtered_df, x='IFEvalQualMetric_score_delta', barmode='group', nbins=20)
# add title
fig.update_layout(title_text='Output Quality Score Change upon steering'.title())
# add axis labels
fig.update_xaxes(title_text='Delta in Quality Score')
fig.update_yaxes(title_text='Count')

# resize the figure
fig.update_layout(
    width=500,
    height=300,
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
        x0=filtered_df.IFEvalQualMetric_score_delta.mean(),
        y0=0,
        x1=filtered_df.IFEvalQualMetric_score_delta.mean(),
        y1=filtered_df.IFEvalQualMetric_score_delta.value_counts().max()+10,
        line=dict(color=line_color, width=2, dash='dash')
    )
)

# add horizontal text at the mean
fig.add_annotation(
    x=filtered_df.IFEvalQualMetric_score_delta.mean()-0.22,
    y=filtered_df.IFEvalQualMetric_score_delta.value_counts().max()+5,
    text=f'Mean: {filtered_df.IFEvalQualMetric_score_delta.mean():.2f}',
    showarrow=False,
    arrowhead=1,
    arrowcolor='red',
    arrowwidth=2,
    arrowsize=1,
    ax=-50,
    ay=+40,
    font=dict(color=line_color, size=12)
)

# store the plot as pdf
# fig.write_image('./plots/quality_score_change_phi3_no_instr.pdf')

fig.show()
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
filtered_df = joined_df[joined_df.instruction_id_list.apply(lambda x: 'multiple' in x[0])] 

for i, r in filtered_df.iterrows():
    print(f'Prompt: {r.original_prompt_no_instr}')
    print(f'model_questions: {r.model_questions['questions']}')
    print(f'answers: {r.model_answers_no_steering}')
    print(f'IFEvalQualMetric_score: {r.IFEvalQualMetric_score_no_steering}')
    print(f'Output (no steering): {r.response_no_steering}')
    print('--------------------------------')
    print(f'IFEvalQualMetric_score: {r.IFEvalQualMetric_score}')
    print(f'response: {r.model_output}')
    print(f'answers: {r.model_answers}')
    print(f'Output (STEERING): {r.response}')
    print('================================')
# %%

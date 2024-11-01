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
from collections import Counter
# %%
import os
import json
import pandas as pd

folder = 'gpt-4o_evaluation/29-09-2024_gpt4o_eval/29-09-2024_gpt4o_eval/keyword_inclusion'
# folder = 'gpt-4o_evaluation/29-09-2024_gpt4o_eval/29-09-2024_gpt4o_eval/keyword_exclusion'
# folder = 'gpt-4o_evaluation/29-09-2024_gpt4o_eval/29-09-2024_gpt4o_eval/length/1-5sentences_100examples'
setting_dfs = {}
settings = []

for i, setting_folder in enumerate(os.listdir(folder)):
    if setting_folder == '.DS_Store':
        continue
    path = os.path.join(folder, setting_folder, 'outputs', 'answer_post_processing_output', 'transformed_data.jsonl')
    with open(path, 'r') as f:
        results = [json.loads(line) for line in f]
    data_df = pd.DataFrame(results)
    setting_dfs[setting_folder] = data_df
    settings.append(setting_folder)

for i, sett in enumerate(settings):
    print(f'Setting {i}: {sett}')

# %%
qual_score_deltas_dict = {}
qual_score_sett1_dict = {}
qual_score_sett2_dict = {}

pairs_of_setting_incl = [(1,0), (3,4), (1,3)]
pairs_of_setting_excl = [(5,2), (3,6), (5,3)]
# pairs_of_setting = [(1,0), (1,2), (2,3)]

if 'inclusion' in folder:
    pairs_of_setting = pairs_of_setting_incl
elif 'exclusion' in folder:
    pairs_of_setting = pairs_of_setting_excl
else:
    raise ValueError('Folder name must contain either "inclusion" or "exclusion"')

for pair in pairs_of_setting:
    print(f'sett1 : {settings[pair[0]]}, sett2: {settings[pair[1]]}')


# %%
n_runs = 1
steering_settings = list(setting_dfs.keys())
qual_score_deltas = np.zeros((len(pairs_of_setting), n_runs, len(setting_dfs[steering_settings[0]])))
qual_score_sett1 = np.zeros((len(pairs_of_setting), n_runs, len(setting_dfs[steering_settings[0]])))
qual_score_sett2 = np.zeros((len(pairs_of_setting), n_runs, len(setting_dfs[steering_settings[0]])))
joined_dfs = {}


for pair_idx, pair in enumerate(pairs_of_setting):
    setting1 = steering_settings[pair[0]]
    setting2 = steering_settings[pair[1]]
    run_idx =0 

    df1 = setting_dfs[setting1]
    df2 = setting_dfs[setting2]
    columns_to_merge = ['key', 'response', 'model_answers', 'model_output', 'uid']
    print(f'Length of df1: {len(df1)}')
    print(f'Length of df2: {len(df2)}')

    joined_df = pd.merge(df1, df2[columns_to_merge], on='uid', suffixes=('', '_steering'))
    print(f'Length of joined_df: {len(joined_df)}')

    qual_scores = []
    qual_scores_steering = []
    for i, r in joined_df.iterrows():
        if r['model_answers'].__len__() == 0 or r['model_answers_steering'].__len__() == 0:
            # print('Empty')
            # append nan
            qual_scores.append(np.nan)
            qual_scores_steering.append(np.nan)
            continue
        if sum(r['model_answers']['is_answer_valid']) == 0 or sum(r['model_answers_steering']['is_answer_valid']) == 0:
            # print('No valid answers')
            # append nan
            qual_scores.append(np.nan)
            qual_scores_steering.append(np.nan)
            continue
        are_valid = r['model_answers']['is_answer_valid']
        answers = r['model_answers']['answers']
        partial_score = 0
        for idx, is_valid in enumerate(are_valid):
            if is_valid:
                partial_score += answers[idx]
        qual_scores.append(partial_score/sum(are_valid))

        are_valid = r['model_answers_steering']['is_answer_valid']
        answers = r['model_answers_steering']['answers']
        partial_score = 0
        for idx, is_valid in enumerate(are_valid):
            if is_valid:
                partial_score += answers[idx]
        qual_scores_steering.append(partial_score/sum(are_valid))

    joined_df['qual_score'] = qual_scores
    joined_df['qual_score_steering'] = qual_scores_steering
    qual_score_delta = joined_df['qual_score'] - joined_df['qual_score_steering']
    joined_df['qual_score_delta'] = qual_score_delta
    qual_score_deltas[pair_idx, run_idx] = qual_score_delta
    joined_dfs[pair_idx, run_idx] = joined_df
    qual_score_sett1[pair_idx, run_idx] = joined_df['qual_score']
    qual_score_sett2[pair_idx, run_idx] = joined_df['qual_score_steering']

    print(f'Setting pair: {setting1} vs {setting2}')
    print(f'Average quality score: {joined_df["qual_score"].mean()}')
    print(f'Average quality score steering: {joined_df["qual_score_steering"].mean()}')

# %%

# Make bar chart of the quality score deltas for each model for each pair of settings
fig = go.Figure()

setting_names = ['Steering<br><b>w/o</b> Instr.', 'Steering<br><b>w/</b> Instr.', 'No Steering<br> w/ vs. w/o Instr.']
pretty_model_names = {'phi': 'Phi-3', 'gemma-2-2b-it': 'Gemma 2B IT', 'mistral-7b-instruct': 'Mistral 7B I.', 'gemma-2-9b-it': 'Gemma 9B IT'}

show_legend = False

overall_mean_scores = []
overall_ses = []

for setting_pair_idx, pair in enumerate(pairs_of_setting):
    length_constr_df = joined_dfs[setting_pair_idx, 0]
    scores = length_constr_df['qual_score_delta'].values

    # drop nan values
    scores = scores[~np.isnan(scores)]

    print(f'Setting pair: {pair}')
    print(f'length of scores: {len(scores)}')

    mean_score = -np.mean(scores)
    std_dev = np.std(scores, ddof=1)
    se = std_dev / np.sqrt(scores.shape[0])

    overall_mean_scores.append(mean_score)
    overall_ses.append(se)

# Make bar plot of the quality score deltas with error bars
fig.add_trace(go.Bar(
    x=setting_names,
    y=overall_mean_scores,
    error_y=dict(type='data', array=overall_ses, width=0),
    # name=f'{length_constraint}',
    marker_color=px.colors.qualitative.Plotly[0],
    showlegend=show_legend
))

if 'inclusion' in folder:
    title='(b) Word Inclusion'
elif 'exclusion' in folder:
    title='(c) Word Exclusion'

fig.update_layout(
    title=title,
    yaxis_title='Avg. Qual. score Delta'.title(),
    barmode='group'
)

# resize the figure
fig.update_layout(
    autosize=False,
    width=290,
    height=250,
)

# decrease fond size of the legend
fig.update_layout(
    legend=dict(
        font=dict(size=11.8)
    )
)

# move legend to the bottom
fig.update_layout(
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.45,
        xanchor='right',
        x=0.9
    )
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=0),
)

# add vertical line that separates the two pairs of settings
fig.add_shape(
    dict(
        type='line',
        x0=1.5,
        x1=1.5,
        y0=0.05,
        y1=-0.18,
        line=dict(
            color='black',
            dash='dash',
            width=1
        )
    )
)

fig.show()

# store plot as pdf
if 'inclusion' in folder:
    plotly.io.write_image(fig, './plots_for_paper/quality_score/word_inclusion.pdf')
elif 'exclusion' in folder:
    plotly.io.write_image(fig, './plots_for_paper/quality_score/word_exclusion.pdf')

# %%
# =============================================================================
# histogram
# =============================================================================

# Make histogram of the quality score deltas 
fig = go.Figure()

setting_names = ['Steering<br><b>w/o</b> Instr.', 'Steering<br><b>w/</b> Instr.', 'No Steering<br> w/ vs. w/o Instr.']

setting_idx = 1

length_constr_df = joined_dfs[setting_idx, 0]
scores = length_constr_df['qual_score_delta'].values

# drop nan values
scores = -scores[~np.isnan(scores)]
if setting_idx == 0:
    title = '(a) Steering w/o Instr.'
    color = px.colors.qualitative.Plotly[2]
elif setting_idx == 1:
    title = '(b) Steering w/ Instr.'
    color = px.colors.qualitative.Plotly[0]
elif setting_idx == 2:
    title = '(c) No Steering w/ vs. w/o Instr.'
    color = px.colors.qualitative.Plotly[1]

# make histogram
fig.add_trace(go.Histogram(
    x=scores,
    histnorm='percent',
    name=setting_names[setting_idx],
    marker_color=color,
    showlegend=False
))

fig.update_layout(
    title=title,
    xaxis_title='Quality score Delta'.title(),
    yaxis_title='Frequency (%)'.title(),
    barmode='overlay',
)

# resize the figure
fig.update_layout(
    autosize=False,
    width=300,
    height=250,
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),
)



# set x axis range
fig.update_layout(
    xaxis=dict(range=[-0.8, 0.8])
)

line_color='black'
# add vertical line at the mean
fig.add_shape(
    dict(
        type='line',
        x0=scores.mean(),
        y0=0,
        x1=scores.mean(),
        y1=60,
        line=dict(color=line_color, width=2, dash='dash')
    )
)

# set y axis range
fig.update_layout(
    yaxis=dict(range=[0, 60])
)

# set y ticks
fig.update_layout(
    yaxis=dict(tickvals=[0, 10, 20, 30, 40, 50, 60])
)

# add horizontal text at the mean
fig.add_annotation(
    x=scores.mean()-0.30,
    y=55,
    text=f'Mean: {scores.mean():.2f}',
    showarrow=False,
    arrowhead=1,
    arrowcolor='red',
    arrowwidth=2,
    arrowsize=1,
    ax=-60,
    ay=+40,
    font=dict(color=line_color, size=12)
)

fig.show()

# store plot as pdf
if setting_idx == 0:
    plotly.io.write_image(fig, './plots_for_paper/quality_score/histogram_steering_no_instr.pdf')
elif setting_idx == 1:
    plotly.io.write_image(fig, './plots_for_paper/quality_score/histogram_steering_instr.pdf')
elif setting_idx == 2:
    plotly.io.write_image(fig, './plots_for_paper/quality_score/histogram_no_steering_instr.pdf')


# %%
for pair in [(0,1), (1,2), (0,2)]:
    a = joined_dfs[pair[0], 0]['qual_score_delta'].values
    b = joined_dfs[pair[1], 0]['qual_score_delta'].values

    res = ttest_rel(a, b)
    print(f'Setting pair: {pair}')
    print(f'p-value: {res.pvalue}')

# %%
# Make histogram of the quality score deltas 
fig = go.Figure()

setting_names = ['Steering<br><b>w/o</b> Instr.', 'Steering<br><b>w/</b> Instr.', 'No Steering<br> w/ vs. w/o Instr.']

# Define the settings to plot
settings_to_plot = [1, 2]
colors = [px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1]]

for idx, setting_idx in enumerate(settings_to_plot):
    length_constr_df = joined_dfs[setting_idx, 0]
    scores = length_constr_df['qual_score_delta'].values

    # Drop NaN values
    scores = -scores[~np.isnan(scores)]

    # Add histogram trace
    fig.add_trace(go.Histogram(
        x=scores,
        histnorm='percent',
        name=setting_names[setting_idx],
        marker_color=colors[idx],
        opacity=0.6,  # Set opacity for transparency
        showlegend=True
    ))

    # Add vertical line at the mean
    fig.add_shape(
        dict(
            type='line',
            x0=scores.mean(),
            y0=0,
            x1=scores.mean(),
            y1=Counter(scores)[0.0] + 10,
            line=dict(color=colors[idx], width=2, dash='dash')
        )
    )

    # Add horizontal text at the mean
    fig.add_annotation(
        x=scores.mean() - 0.30,
        y=Counter(scores)[0.0] + 5,
        text=f'Mean: {scores.mean():.2f}',
        showarrow=False,
        font=dict(color=colors[idx], size=12)
    )

fig.update_layout(
    title='Changes in Response Quality Score',
    xaxis_title='Quality score Delta'.title(),
    yaxis_title='Frequency'.title(),
    barmode='overlay'
)

# Resize the figure
fig.update_layout(
    autosize=False,
    width=350,
    height=300,
)

# Remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),
)

# Move legend to the bottom
fig.update_layout(
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.45,
        xanchor='right',
        x=0.1,
    )
)

# Set x-axis range
fig.update_layout(
    xaxis=dict(range=[-0.8, 0.8])
)

fig.show()


# %%
# print some outputs
setting_pair_idx = 1
run_idx = 0
joined_df = joined_dfs[setting_pair_idx, run_idx]

# sort df by quality score delta
sorted_df = joined_df.sort_values(by='qual_score_delta', ascending=False)

for i, r in sorted_df.head(10).iterrows():
    print(f'Instruction: {r["instruction_id_list"]}')
    print(f'Quality score: {r["qual_score"]}')
    print(f'Quality score steering: {r["qual_score_steering"]}')
    print(f'Quality score delta: {r["qual_score_delta"]}')
    print(f'Question: {r["original_prompt_subq"]}')
    print(f'Response: {r["response"]}')
    print(f'------------')
    print(f'Response steering: {r["response_steering"]}')
    # print(f'Answer: {r["model_answers"]}')
    # print(f'Answer steering: {r["model_answers_steering"]}')
    print('===========================')

# %%

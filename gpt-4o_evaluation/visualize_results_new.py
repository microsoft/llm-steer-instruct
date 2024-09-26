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
import os
import json
import pandas as pd

folder = 'gpt-4o_evaluation/gpt_4o_evals-2'
model_name = 'phi'
model_name = 'gemma-2-2b-it'
# model_name = 'mistral-7b-instruct'
model_names = ['phi', 'gemma-2-2b-it', 'mistral-7b-instruct', 'gemma-2-9b-it']

pairs_of_setting = [(0,1), (2,3), (0,2)]
joined_dfs = {model_name: {} for model_name in model_names}
qual_score_deltas_dict = {}
qual_score_sett1_dict = {}
qual_score_sett2_dict = {}


for model_name in model_names:
    if model_name == 'gemma-2-9b-it':
        std_setting = 'standard_no_hf'
    else:
        std_setting = 'standard'

    # Define the steering settings
    steering_settings = ['no_instr', 'adjust_rs_-1', std_setting, 'instr_plus_adjust_rs_-1']

    if model_name == 'phi':
        sep = '_'
        steering_settings = ['no_instr_no_steering', 'no_instr_steering', 'instr_no_steering', 'instr_steering']
    else:
        sep = '-'

    runs = {k: [] for k in steering_settings}

    # Load the results
    for steering in steering_settings:
        path = f'{folder}/{model_name}{sep}{steering}_gpt4o/'
        for f in os.listdir(path):
            new_path = os.path.join(path, f, 'answer_post_processing_output', 'transformed_data.jsonl')
            with open(new_path, 'r') as file:
                results = [json.loads(line) for line in file]
            df = pd.DataFrame(results)
            runs[steering].append(df)

    instr_to_drop = ['language:response_language']
    # if model_name == 'phi':
        # instr_to_drop.append('detectable_format:multiple_sections')
    instr_to_drop_for_wo_steering = ['detectable_format:multiple_sections', 'detectable_format:title']

    len_df = len(runs[steering_settings[1]][0]) - len(runs[steering_settings[1]][0][runs[steering_settings[1]][0]['instruction_id_list'].apply(lambda x: any([instr in x for instr in instr_to_drop]))])

    qual_score_deltas = np.zeros((len(pairs_of_setting), len(runs[steering_settings[0]]), len_df))
    qual_score_sett1 = np.zeros((len(pairs_of_setting), len(runs[steering_settings[0]]), len_df))
    qual_score_sett2 = np.zeros((len(pairs_of_setting), len(runs[steering_settings[0]]), len_df))

    for pair_idx, pair in enumerate(pairs_of_setting):
        setting1 = steering_settings[pair[0]]
        setting2 = steering_settings[pair[1]]
        print(f'Pair: {setting1} vs {setting2}')
        for run_idx in range(len(runs[steering_settings[0]])):
            df1 = runs[setting1][run_idx]
            df2 = runs[setting2][run_idx]
            columns_to_merge = ['key', 'response', 'model_answers', 'model_output']
            if pair_idx != 2:
                if 'steering_layer' not in df2.columns or True:
                    print(f'Loading steering layer for {setting2}')
                    # load results with steering layer
                    if pair_idx == 0:
                        instr = 'no_instr'
                    else:
                        instr = 'instr'
                    model_name_rep = model_name + ('-3' if model_name == 'phi' else '')
                    path = f'ifeval_experiments/representations/{model_name_rep}/single_instr_all_base_x_all_instr/pre_computed_ivs_best_layer_validation_{instr}.h5'
                    pre_computed_ivs = pd.read_hdf(path)
                    steering_layer_dict = { instr: l for instr, l in zip(pre_computed_ivs['instruction'], pre_computed_ivs['max_diff_layer_idx'])}
                    df2['steering_layer'] = df2['instruction_id_list_for_eval'].apply(lambda x: steering_layer_dict.get(x[0], -1))
                columns_to_merge.append('steering_layer')

            joined_df = pd.merge(df1, df2[columns_to_merge], on='key', suffixes=('', '_steering'))
            if len(joined_df) > 163:
                # remove rows with key value not present in the runs[steering_settings[0]][0] dataframe
                keys = runs[steering_settings[1]][0]['key'].values
                joined_df = joined_df[joined_df['key'].isin(keys)]
                print(f' Length of joined_df after filtering: {len(joined_df)}')

            # drop rows with instr_to_drop
            joined_df = joined_df[joined_df['instruction_id_list'].apply(lambda x: all([instr not in x for instr in instr_to_drop]))]
            print(f' Length of joined_df after dropping instr: {len(joined_df)}')

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
                if 'steering_layer' in r and r['steering_layer'] == -1:
                    # print('Steering layer is -1')
                    # append nan
                    qual_scores.append(np.nan)
                    qual_scores_steering.append(np.nan)
                    continue
                if pair_idx == 0 and r['instruction_id_list'][0] in instr_to_drop_for_wo_steering:
                    print(f'Instruction to drop: {r["instruction_id_list"]}')
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
            joined_dfs[model_name][pair_idx, run_idx] = joined_df
            qual_score_sett1[pair_idx, run_idx] = joined_df['qual_score']
            qual_score_sett2[pair_idx, run_idx] = joined_df['qual_score_steering']
        
        qual_score_deltas_dict[model_name] = qual_score_deltas
        qual_score_sett1_dict[model_name] = qual_score_sett1
        qual_score_sett2_dict[model_name] = qual_score_sett2

        print(f'Model: {model_name}')
        print(f'Setting pair: {setting1} vs {setting2}')
        print(f'Average quality score: {joined_df["qual_score"].mean()}')
        print(f'Average quality score steering: {joined_df["qual_score_steering"].mean()}')

# %%
# Define a color mapping for each model
color_mapping = { model_name: px.colors.qualitative.Plotly[4+i] for i, model_name in enumerate(model_names) }

# Make bar chart of the quality score deltas for each model for each pair of settings
fig = go.Figure()

setting_names = ['Steering<br><b>w/o</b> Instr.', 'Steering<br><b>w/</b> Instr.', 'No Steering<br> w/ vs. w/o Instr.']
pretty_model_names = {'phi': 'Phi-3', 'gemma-2-2b-it': 'Gemma 2B IT', 'mistral-7b-instruct': 'Mistral 7B I.', 'gemma-2-9b-it': 'Gemma 9B IT'}

show_legend = True

for model_name in model_names:
    overall_mean_scores = []
    overall_ses = []

    for setting_pair_idx, pair in enumerate(pairs_of_setting):
        scores = qual_score_deltas_dict[model_name][setting_pair_idx]

        # Drop rows with NaNs
        scores = scores[:, ~np.isnan(scores).any(axis=0)]

        # Transpose the matrix
        scores = scores.T

        # Step 1: Compute the average score for each example across the 3 runs
        mean_scores = np.mean(scores, axis=1)

        # Step 2: Compute the standard deviation for each example
        std_devs = np.std(scores, axis=1, ddof=1)

        # Step 3: Compute the standard error for each example
        standard_errors = std_devs / np.sqrt(3)

        # Option 1: Compute the standard error of the overall mean score
        overall_mean_score = -np.mean(mean_scores)
        overall_std_dev = np.std(mean_scores, ddof=1)
        overall_se = overall_std_dev / np.sqrt(scores.shape[0])

        overall_mean_scores.append(overall_mean_score)
        overall_ses.append(overall_se)

    # Make bar plot of the quality score deltas with error bars
    fig.add_trace(go.Bar(
        x=setting_names,
        y=overall_mean_scores,
        error_y=dict(type='data', array=overall_ses, width=0),
        name=pretty_model_names[model_name],
        marker_color=color_mapping[model_name],
        showlegend=show_legend
    ))

fig.update_layout(
    title='Changes in Response Quality Score',
    yaxis_title='Avg. Quality score Delta'.title(),
    barmode='group'
)

# resize the figure
fig.update_layout(
    autosize=False,
    width=350,
    height=300,
)

# decrease fond size of the legend
fig.update_layout(
    legend=dict(
        font=dict(size=11.8)
    )
)

# # incline the x-axis labels
# fig.update_layout(
#     xaxis=dict(tickangle=30),
# )

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
    margin=dict(l=0, r=0, t=30, b=0),
)

# add vertical line that separates the two pairs of settings
fig.add_shape(
    dict(
        type='line',
        x0=1.5,
        x1=1.5,
        y0=0.03,
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
# plotly.io.write_image(fig, './plots_for_paper/quality_score_deltas.pdf')

# %%
# =============================================================================
# Same plot but the x-axis labels are the models and the colors are the settings
# =============================================================================

# Define a color mapping for each setting pair
color_mapping = {}
color_mapping[0] = px.colors.qualitative.Plotly[0]
color_mapping[1] = px.colors.qualitative.Plotly[3]
color_mapping[2] = px.colors.qualitative.T10[9]

# Make bar chart of the quality score deltas for each setting pair for each model
fig = go.Figure()

setting_names = ['Steering <b>w/o</b> Instr.', 'Steering <b>w/</b> Instr.', 'No Steering w/ vs. w/o Instr.']
pretty_model_names = {'phi': 'Phi-3', 'gemma-2-2b-it': 'Gemma<br>2B IT', 'mistral-7b-instruct': 'Mistral<br>7B I.', 'gemma-2-9b-it': 'Gemma<br>9B IT'}

show_legend = True

for setting_pair_idx, pair in enumerate(pairs_of_setting):
    overall_mean_scores = []
    overall_ses = []

    for model_name in model_names:
        scores = qual_score_deltas_dict[model_name][setting_pair_idx]

        # Drop rows with NaNs
        scores = scores[:, ~np.isnan(scores).any(axis=0)]

        # Transpose the matrix
        scores = scores.T

        # Step 1: Compute the average score for each example across the 3 runs
        mean_scores = np.mean(scores, axis=1)

        # Step 2: Compute the standard deviation for each example
        std_devs = np.std(scores, axis=1, ddof=1)

        # Step 3: Compute the standard error for each example
        standard_errors = std_devs / np.sqrt(3)

        # Option 1: Compute the standard error of the overall mean score
        overall_mean_score = -np.mean(mean_scores)
        overall_std_dev = np.std(mean_scores, ddof=1)
        overall_se = overall_std_dev / np.sqrt(scores.shape[0])

        overall_mean_scores.append(overall_mean_score)
        overall_ses.append(overall_se)

    # Make bar plot of the quality score deltas with error bars
    fig.add_trace(go.Bar(
        x=[pretty_model_names[model_name] for model_name in model_names],
        y=overall_mean_scores,
        error_y=dict(type='data', array=overall_ses, width=0),
        name=setting_names[setting_pair_idx],
        marker_color=color_mapping[setting_pair_idx],
        showlegend=show_legend
    ))

fig.update_layout(
    title='Changes in Response Quality Score',
    yaxis_title='Avg. Quality Score Delta'.title(),
    barmode='group'
)

# Resize the figure
fig.update_layout(
    autosize=False,
    width=350,
    height=300,
)

# Decrease font size of the legend
fig.update_layout(
    legend=dict(
        font=dict(size=12)
    )
)

# Move legend to the bottom
fig.update_layout(
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.6,
        xanchor='right',
        x=0.85
    )
)

# Remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),
)

fig.show()

# Store plot as pdf (optional)
# plotly.io.write_image(fig, './plots_for_paper/quality_score_deltas.pdf')



 # %%
model_name = 'mistral-7b-instruct'
qual_score_delta_means_mean = {}
qual_score_delta_means_sem = {}

for pair_idx, pair in enumerate(pairs_of_setting):
    scores = qual_score_deltas_dict[model_name][pair_idx]

    print('Shape of scores:', scores.shape)
    # drop rows with nans
    scores = scores[:, ~np.isnan(scores).any(axis=0)]
    print('Shape of scores after dropping nans:', scores.shape)

    # transpose the matrix
    scores = scores.T

    # Step 1: Compute the average score for each example across the 3 runs
    mean_scores = np.mean(scores, axis=1)

    # Step 2: Compute the standard deviation for each example
    std_devs = np.std(scores, axis=1, ddof=1)

    # Step 3: Compute the standard error for each example
    standard_errors = std_devs / np.sqrt(3)

    # Option 1: Compute the standard error of the overall mean score
    overall_mean_score = np.mean(mean_scores)
    overall_std_dev = np.std(mean_scores, ddof=1)
    overall_se = overall_std_dev / np.sqrt(scores.shape[0])

    qual_score_delta_means_mean[pair] = overall_mean_score
    qual_score_delta_means_sem[pair] = overall_se

print(qual_score_delta_means_mean)
print(qual_score_delta_means_sem)

# %%
# make  bar plot of the quality score deltas with error bars
fig = go.Figure()
for pair in pairs_of_setting:
    fig.add_trace(go.Bar(
        x=[f'{steering_settings[pair[0]]} vs {steering_settings[pair[1]]}'],
        y=[qual_score_delta_means_mean[pair]],
        error_y=dict(type='data', array=[qual_score_delta_means_sem[pair]]),
        name='Quality score delta',
    ))

fig.update_layout(
    title=f'Quality score deltas - {model_name}',
    yaxis_title='Quality score delta',
    barmode='group'
)

fig.show()
# %%
# =============================================================================
# Per-instruction quality score deltas
# =============================================================================
setting_pair_idx = 1
run_idx = 0
model_name = 'gemma-2-2b-it'
joined_df = joined_dfs[model_name][setting_pair_idx, run_idx]

# compute the average quality score delta for each instruction
instr_to_drop = 'language:response_language'
filtered_df = joined_df[joined_df['instruction_id_list'].apply(lambda x: instr_to_drop not in x)]

instr_ids = filtered_df['instruction_id_list'].apply(lambda x: x[0].split(':')[-1])
filtered_df['instruction_id'] = instr_ids

# make bar chart of the quality score deltas per instruction
grouped_df = filtered_df['qual_score_delta'].groupby(filtered_df['instruction_id']).mean().reset_index()
grouped_df_sem = filtered_df['qual_score_delta'].groupby(filtered_df['instruction_id']).sem().reset_index()
fig = go.Figure()
fig.add_trace(go.Bar(
    x=grouped_df['instruction_id'],
    y=grouped_df['qual_score_delta'],
    error_y=dict(type='data', array=grouped_df_sem['qual_score_delta']),
    name='Quality score delta'
))

fig.update_layout(
    title='Quality score deltas per instruction',
    yaxis_title='Quality score delta ',
    barmode='group'
)

fig.show()


# %%
# print some outputs
setting_pair_idx = 1
run_idx = 0
joined_df = joined_dfs[model_name][setting_pair_idx, run_idx]

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


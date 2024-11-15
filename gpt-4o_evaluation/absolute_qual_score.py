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
model_names = ['phi', 'gemma-2-2b-it', 'mistral-7b-instruct', 'gemma-2-9b-it']

results_rows = []

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

    len_df = len(runs[steering_settings[1]][0]) - len(runs[steering_settings[1]][0][runs[steering_settings[1]][0]['instruction_id_list'].apply(lambda x: any([instr in x for instr in instr_to_drop]))])

    for sett_idx, setting in enumerate(steering_settings):
        print(f'Sett: {setting}')
        for run_idx in range(len(runs[steering_settings[0]])):
            df = runs[setting][run_idx]
            columns_to_merge = ['key', 'response', 'model_answers', 'model_output']
            # if sett_idx in [1,3]:
            #     if 'steering_layer' not in df.columns or True:
            #         print(f'Loading steering layer for {setting}')
            #         # load results with steering layer
            #         if sett_idx == 1:
            #             instr = 'no_instr'
            #         else:
            #             instr = 'instr'
            #         model_name_rep = model_name + ('-3' if model_name == 'phi' else '')
            #         path = f'ifeval_experiments/representations/{model_name_rep}/single_instr_all_base_x_all_instr/pre_computed_ivs_best_layer_validation_{instr}.h5'
            #         pre_computed_ivs = pd.read_hdf(path)
            #         steering_layer_dict = { instr: l for instr, l in zip(pre_computed_ivs['instruction'], pre_computed_ivs['max_diff_layer_idx'])}
            #         df['steering_layer'] = df['instruction_id_list_for_eval'].apply(lambda x: steering_layer_dict.get(x[0], -1))
            #     columns_to_merge.append('steering_layer')

            # joined_df = pd.merge(df1, df2[columns_to_merge], on='key', suffixes=('', '_steering'))
            if len(df) > 163:
                # remove rows with key value not present in the runs[steering_settings[0]][0] dataframe
                keys = runs[steering_settings[1]][0]['key'].values
                df = df[df['key'].isin(keys)]
                print(f' Length of joined_df after filtering: {len(df)}')

            # drop rows with instr_to_drop
            # df = df[df['instruction_id_list'].apply(lambda x: all([instr not in x for instr in instr_to_drop]))]
            # print(f' Length of joined_df after dropping instr: {len(df)}')

            qual_scores = []
            for i, r in df.iterrows():
                if r['model_answers'].__len__() == 0:
                    # print('Empty')
                    # append nan
                    qual_scores.append(np.nan)
                    continue
                if sum(r['model_answers']['is_answer_valid']) == 0:
                    # print('No valid answers')
                    # append nan
                    qual_scores.append(np.nan)
                    continue
                if r.instruction_id_list[0] in instr_to_drop:
                    qual_scores.append(np.nan)
                    continue
                are_valid = r['model_answers']['is_answer_valid']
                answers = r['model_answers']['answers']
                partial_score = 0
                for idx, is_valid in enumerate(are_valid):
                    if is_valid:
                        partial_score += answers[idx]

                qual_score = partial_score/sum(are_valid)

                if model_name == 'phi' and sett_idx == 1 and r.instruction_id_list[0] in ['detectable_content:multiple_sections', 'detectable_format:json_format', 'detectable_format:number_bullet_lists']:
                    # set qual score as the same as in results_rows[-1]
                    print('i: ', i)
                    print(f"len of results_rows[-1]['qual_scores'] {len(results_rows[-1]['qual_scores'])}")
                    qual_score = results_rows[-1]['qual_scores'][i]

                qual_scores.append(qual_score)

            df['qual_score'] = qual_scores

            results_rows.append({'model_name': model_name, 'setting': setting, 'run_idx': run_idx, 'qual_scores': qual_scores, 'setting_idx': sett_idx})

        print(f'Model: {model_name}')

# %%
aggregate_results = []
for model_name in model_names:
    # get the results for the model
    model_results = [r for r in results_rows if r['model_name'] == model_name]

    # compute mean and sem across runs for each setting
    for setting_idx in range(4):
        setting_results = [r for r in model_results if r['setting_idx'] == setting_idx]
        scores = np.array([r['qual_scores'] for r in setting_results])

        print('Shape of scores:', scores.shape)

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
        overall_mean_score = np.mean(mean_scores)
        overall_std_dev = np.std(mean_scores, ddof=1)
        overall_se = overall_std_dev / np.sqrt(scores.shape[0])

        print('shape of mean_scores:', mean_scores.shape)
        print('shape of standard_errors:', standard_errors.shape)
        

        aggregate_results.append({'model_name': model_name, 'setting_idx': setting_idx, 'overall_mean_score': overall_mean_score, 'overall_se': overall_se})



# %%
# Convert aggregate_results to a DataFrame
df = pd.DataFrame(aggregate_results)

# Define a color mapping for each setting_idx
color_mapping = { i: px.colors.qualitative.Plotly[i] for i in range(4)}

# Create a bar chart
fig = go.Figure()

for setting_idx in range(4):
    setting_df = df[df['setting_idx'] == setting_idx]
    fig.add_trace(go.Bar(
        x=setting_df['model_name'],
        y=setting_df['overall_mean_score'],
        error_y=dict(type='data', array=setting_df['overall_se'], width=0),
        name=f'Setting {setting_idx}',
        marker_color=color_mapping[setting_idx]
    ))

fig.update_layout(
    title='Quality Score Deltas for Different Models and Settings',
    xaxis_title='Model Name',
    yaxis_title='Quality Score Delta',
    barmode='group'
)

fig.show()
# %%

# %%
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.contingency_tables import mcnemar

# %%
# load the dataframes
model_names = ['phi-3', 'gemma-2-2b-it', 'mistral-7b-instruct', 'gemma-2-9b-it']
single_instr = 'single_instr/all_base_x_all_instr'
eval_type = 'loose'

dfs = {}
for model_name in model_names:
    mode = 'no_instr'
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df = pd.DataFrame(results)

    mode = 'adjust_rs_-1_perplexity'
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_steering = pd.DataFrame(results)

    mode = 'standard' if model_name != 'gemma-2-9b-it' else 'standard_no_hf'
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_standard = pd.DataFrame(results)

    mode = 'instr_plus_adjust_rs_-1_perplexity' 
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_instr_plus_steering = pd.DataFrame(results)

    # load ./data/input_data_single_instr_no_instr.jsonl
    with open('./data/input_data_single_instr_no_instr.jsonl') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    data_df = pd.DataFrame(data)

    # rename prompt to prompt_no_instr
    data_df = data_df.rename(columns={'prompt': 'prompt_no_instr'})
    # rename original prompt to prompt
    data_df = data_df.rename(columns={'original_prompt': 'prompt'})
    # drop instruction_id_list
    data_df = data_df.drop(columns=['instruction_id_list'])

    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    results_df = results_df[results_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df.reset_index(drop=True, inplace=True)
    results_df_steering = results_df_steering[results_df_steering.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df_steering.reset_index(drop=True, inplace=True)
    results_df_standard = results_df_standard[results_df_standard.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df_standard.reset_index(drop=True, inplace=True)
    results_df_instr_plus_steering = results_df_instr_plus_steering[results_df_instr_plus_steering.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df_instr_plus_steering.reset_index(drop=True, inplace=True)

    # join data_df with results_df on prompt
    results_df = results_df.merge(data_df, on='prompt')
    results_df_steering = results_df_steering.merge(data_df, on='prompt')
    results_df_standard = results_df_standard.merge(data_df, on='prompt')
    results_df_instr_plus_steering = results_df_instr_plus_steering.merge(data_df, on='prompt')

    # print average follow_all_instructions for detectable_format:title
    print(f'{model_name} - No Instr: {results_df[results_df.instruction_id_list.apply(lambda x: "detectable_format:title" in x)].follow_all_instructions.mean()}')
    print(f'{model_name} - Steering: {results_df_steering[results_df_steering.instruction_id_list.apply(lambda x: "detectable_format:title" in x)].follow_all_instructions.mean()}')
    print(f'{model_name} - Standard: {results_df_standard[results_df_standard.instruction_id_list.apply(lambda x: "detectable_format:title" in x)].follow_all_instructions.mean()}')
    print(f'{model_name} - Instr + Steering: {results_df_instr_plus_steering[results_df_instr_plus_steering.instruction_id_list.apply(lambda x: "detectable_format:title" in x)].follow_all_instructions.mean()}')


    dfs[model_name] = {
        'results_df': results_df,
        'results_df_steering': results_df_steering,
        'results_df_standard': results_df_standard,
        'results_df_instr_plus_steering': results_df_instr_plus_steering
    }

    # compute mcnemar's test
    model1_accuracies = dfs[model_name]['results_df_standard'].follow_all_instructions.astype(int).values
    model2_accuracies = dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.astype(int).values

    table = [[0, 0], [0, 0]]

    for i in range(len(model1_accuracies)):
        table[int(model1_accuracies[i])][int(model2_accuracies[i])] += 1

    result = mcnemar(table, exact=False, correction=True)
    print(f"{model_name} - Instr - McNemar test: {result.pvalue}")

    model1_accuracies = dfs[model_name]['results_df'].follow_all_instructions.astype(int).values
    model2_accuracies = dfs[model_name]['results_df_steering'].follow_all_instructions.astype(int).values

    table = [[0, 0], [0, 0]]

    for i in range(len(model1_accuracies)):
        table[int(model1_accuracies[i])][int(model2_accuracies[i])] += 1

    result = mcnemar(table, exact=False, correction=True)
    print(f"{model_name} - No Instr - McNemar test: {result.pvalue}")


# %%
# =============================================================================
# Setting 1-2: make plot with all models
# =============================================================================


show_error_bars = False

# Calculate means and 95% confidence intervals
df = pd.DataFrame({
    'Model': model_names,
    'Std. Inference': [dfs[model_name]['results_df'].follow_all_instructions.mean() for model_name in model_names],
    'Steering': [dfs[model_name]['results_df_steering'].follow_all_instructions.mean() for model_name in model_names],
    'w/ Instr.': [dfs[model_name]['results_df_standard'].follow_all_instructions.mean() for model_name in model_names],
    'w/ Instr. + Steering': [dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.mean() for model_name in model_names],
    'Std. Inference Error': [1.96 * dfs[model_name]['results_df'].follow_all_instructions.std() / (len(dfs[model_name]['results_df']) ** 0.5) for model_name in model_names],
    'Steering Error': [1.96 * dfs[model_name]['results_df_steering'].follow_all_instructions.std() / (len(dfs[model_name]['results_df_steering']) ** 0.5) for model_name in model_names]
})

# Specify a list of colors for each 'Setting'
index = 0
color = px.colors.qualitative.Plotly[index]

model_labels_dict = {
    'phi-3': 'Phi-3',
    'gemma-2-2b-it': 'Gemma 2<br>2B IT',
    'mistral-7b-instruct': 'Mistral<br>7B Instr.',
    'gemma-2-9b-it': 'Gemma 2<br>9B IT'
}
model_labels = [model_labels_dict[model_name] for model_name in model_names]

# plot 'Std. Inference' and 'Steering' in one plot
fig = go.Figure()
for i, setting in enumerate(['Std. Inference', 'Steering']):
    fig.add_trace(go.Bar(
        x=model_labels,
        y=df[setting],
        name=setting,
        marker_color=color,
        opacity=1 if i == 1 else 0.8,
        marker_pattern_shape='/' if i == 1 else '',
        error_y=dict(
            type='data',
            array=df[f'{setting} Error'],
            visible=show_error_bars,
            width=0,
        )
    ))

# set title
fig.update_layout(title_text='(a) Accuracy <b>w/o</b> Text Instructions')
# change title font size
fig.update_layout(title_font_size=16)

# resize plot
fig.update_layout(width=300, height=250)

fig.update_layout(yaxis=dict(range=[0, 0.4]))

# add y axis label
fig.update_layout(yaxis_title='Accuracy')

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# move legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-0.4,
    xanchor='right',
    x=0.9
))

fig.show()
# %%
# =============================================================================
# Setting 3-4: make plot with all models
# =============================================================================

# Calculate means and 95% confidence intervals
df = pd.DataFrame({
    'Model': model_names,
    'Std. Inference': [dfs[model_name]['results_df_standard'].follow_all_instructions.mean() for model_name in model_names],
    'Steering': [dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.mean() for model_name in model_names],
    'Std. Inference Error': [1 * dfs[model_name]['results_df_standard'].follow_all_instructions.std() / (len(dfs[model_name]['results_df_standard']) ** 0.5) for model_name in model_names],
    'Steering Error': [1 * dfs[model_name]['results_df_instr_plus_steering'].follow_all_instructions.std() / (len(dfs[model_name]['results_df_instr_plus_steering']) ** 0.5) for model_name in model_names]
})

# Specify a list of colors for each 'Setting'
index = 3
color = px.colors.qualitative.Plotly[index]

model_labels_dict = {
    'phi-3': 'Phi-3',
    'gemma-2-2b-it': 'Gemma 2<br>2B IT',
    'mistral-7b-instruct': 'Mistral<br>7B Instr.',
    'gemma-2-9b-it': 'Gemma 2<br>9B IT'
}
model_labels = [model_labels_dict[model_name] for model_name in model_names]

# plot 'Std. Inference' and 'Steering' in one plot
fig = go.Figure()
for i, setting in enumerate(['Std. Inference', 'Steering']):
    fig.add_trace(go.Bar(
        x=model_labels,
        y=df[setting],
        name=setting,
        marker_color=color,
        opacity=1 if i == 1 else 0.8,
        marker_pattern_shape='/' if i == 1 else '',
        error_y=dict(
            type='data',
            array=df[f'{setting} Error'],
            visible=show_error_bars,
            width=0,
        )
    ))

# set title
fig.update_layout(title_text='(b) Accuracy <b>with</b> Text Instr.')
# change title font size
fig.update_layout(title_font_size=16)

# resize plot
fig.update_layout(width=300, height=250)

# set min y to 0.5
fig.update_layout(yaxis=dict(range=[0.5, 0.90]))

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# add y axis label
fig.update_layout(yaxis_title='Accuracy')

# move legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-0.4,
    xanchor='right',
    x=0.9
))

fig.show()
# %%
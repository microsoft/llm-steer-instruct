# %%
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from eval.evaluation_main import test_instruction_following_loose
import plotly

# %%
def truncate_responses(df):
    new_rows = []
    for i, row in df.iterrows():
        if 'Q:' in row['response']:
            row['response'] = row['response'].split('Q:')[0]

        # get row from results_df_w_kwargs with the same prompt
        row_w_kwargs = results_df_w_kwargs[results_df_w_kwargs.prompt == row.prompt].iloc[0]

        # compute accuracy
        prompt_to_response = {}
        prompt_to_response[row['prompt']] = row['response']
        output = test_instruction_following_loose(row_w_kwargs, prompt_to_response)
        row['old_follow_all_instructions'] = row['follow_all_instructions']
        row['follow_all_instructions'] = output.follow_all_instructions
        new_rows.append(row)

    return pd.DataFrame(new_rows)

# %%
overall_accuracies = {}
model_names = ['gemma-2-2b', 'gemma-2-9b']
for model_name in model_names:
    single_instr = 'single_instr/all_base_x_all_instr'
    mode = 'no_instr'
    # mode = 'standard'
    subset = ''

    eval_type = 'loose'

    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df = pd.DataFrame(results)

    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/out.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_w_kwargs = pd.DataFrame(results)

    mode = 'adjust_rs_-1_perplexity'
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_steering = pd.DataFrame(results)

    mode = 'adjust_rs_-1_perplexity_cross_model'
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_steering_cross = pd.DataFrame(results)

    mode = 'standard'
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_standard = pd.DataFrame(results)

    mode = 'instr_plus_adjust_rs_-1_perplexity'
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_instr_plus_steering = pd.DataFrame(results)

    mode = 'instr_plus_adjust_rs_-1_perplexity_cross_model'
    path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
    with open(path_to_results) as f:
        results = f.readlines()
        results = [json.loads(r) for r in results]
    results_df_instr_plus_steering_cross = pd.DataFrame(results)

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

    # join data_df with results_df on prompt
    results_df = results_df.merge(data_df, on='prompt')
    results_df_steering = results_df_steering.merge(data_df, on='prompt')
    results_df_steering_cross = results_df_steering_cross.merge(data_df, on='prompt')
    results_df_standard = results_df_standard.merge(data_df, on='prompt')
    results_df_instr_plus_steering = results_df_instr_plus_steering.merge(data_df, on='prompt')
    results_df_instr_plus_steering_cross = results_df_instr_plus_steering_cross.merge(data_df, on='prompt')

    nonparametric_only = True
    if nonparametric_only:
        # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
        filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
        results_df = results_df[results_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
        results_df.reset_index(drop=True, inplace=True)
        results_df_steering = results_df_steering[results_df_steering.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
        results_df_steering.reset_index(drop=True, inplace=True)
        results_df_steering_cross = results_df_steering_cross[results_df_steering_cross.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
        results_df_steering_cross.reset_index(drop=True, inplace=True)
        results_df_standard = results_df_standard[results_df_standard.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
        results_df_standard.reset_index(drop=True, inplace=True)
        results_df_instr_plus_steering_cross = results_df_instr_plus_steering_cross[results_df_instr_plus_steering_cross.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
        results_df_instr_plus_steering_cross.reset_index(drop=True, inplace=True)
        results_df_instr_plus_steering = results_df_instr_plus_steering[results_df_instr_plus_steering.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
        results_df_instr_plus_steering.reset_index(drop=True, inplace=True)
    
    # correct ifeval accuracy scores by truncating the responses at the first occurence of 'Q:'
    results_df = truncate_responses(results_df)
    results_df_steering = truncate_responses(results_df_steering)
    results_df_steering_cross = truncate_responses(results_df_steering_cross)
    results_df_standard = truncate_responses(results_df_standard)
    results_df_instr_plus_steering = truncate_responses(results_df_instr_plus_steering)
    results_df_instr_plus_steering_cross = truncate_responses(results_df_instr_plus_steering_cross)

    # Store overall accuracy
    overall_accuracy = [
        {'setting': '<b>w/o</b> Instr.', 'steering': 'Std. Inference', 'accuracy': results_df.follow_all_instructions.mean(),},
        {'setting': '<b>w/o</b> Instr.', 'steering': 'Same-model Steering', 'accuracy': results_df_steering.follow_all_instructions.mean()},
        {'setting': '<b>w/o</b> Instr.', 'steering': 'Cross-model Steering', 'accuracy': results_df_steering_cross.follow_all_instructions.mean()},
        {'setting': '<b>w/</b> Instr.', 'steering': 'Std. Inference', 'accuracy': results_df_standard.follow_all_instructions.mean()},
        {'setting': '<b>w/</b> Instr.', 'steering': 'Same-model Steering', 'accuracy': results_df_instr_plus_steering.follow_all_instructions.mean()},
        {'setting': '<b>w/</b> Instr.', 'steering': 'Cross-model Steering', 'accuracy': results_df_instr_plus_steering_cross.follow_all_instructions.mean()}
    ]

    overall_accuracies[model_name] = overall_accuracy

# %%
setting = '<b>w/o</b> Instr.'
setting = '<b>w/</b> Instr.'

# Create a DataFrame for overall accuracy
df_overall_2b = pd.DataFrame(overall_accuracies['gemma-2-2b'])
df_overall_2b = df_overall_2b[df_overall_2b['setting'] == setting]

df_overall_9b = pd.DataFrame(overall_accuracies['gemma-2-9b'])
df_overall_9b = df_overall_9b[df_overall_9b['setting'] == setting]

# merge the two dataframes adding the model name as a column
df_overall_2b['model'] = 'Gemma 2 2B'
df_overall_9b['model'] = 'Gemma 2 9B'

df_overall = pd.concat([df_overall_2b, df_overall_9b])

# Define color mapping for the two groups
colors = [plotly.colors.qualitative.Plotly[8], plotly.colors.qualitative.Plotly[3]]

# Create a bar plot for overall accuracy
fig = go.Figure()

for steering in df_overall['steering'].unique():
    df_temp = df_overall[df_overall['steering'] == steering]
    if steering == 'Std. Inference':
        pattern_shape = ""
        opacity = 0.7
    elif steering == 'Same-model Steering':
        pattern_shape = "/"
        opacity = 0.85
    elif steering == 'Cross-model Steering':
        pattern_shape = "x"
        opacity = 1
    print(steering)
    fig.add_trace(go.Bar(
        x=df_temp['model'],
        y=df_temp['accuracy'],
        name=steering,
        marker=dict(color=colors, opacity=opacity, pattern=dict(
            shape=pattern_shape,
            fillmode="overlay",
            size=10,
            solidity=0.2 if steering == 'Same-model Steering' else 0.2,
            fgcolor='black'  # Pattern color
        )),
        showlegend=False,
    ))


# Add custom legend item for 'Std. Inference'
fig.add_trace(go.Bar(
    x=[None], y=[None],
    marker=dict(color='white', line=dict(color='black', width=1)),
    showlegend=True,
    name='Std. Inference',
    offset=-10,
))

# Add custom legend item for 'Steering' with pattern
fig.add_trace(go.Bar(
    x=[None], y=[None],
    marker=dict(color='white', pattern=dict(
            shape="/",
            fillmode="replace",
            size=10,
            solidity=0.2,
            fgcolor='black'  # Pattern color
        ), line=dict(color='black', width=1)),
    showlegend=True,
    name='Same-model Steering',
    offset=-10,
))

# Add custom legend item for 'Cross-model Steering' with pattern
fig.add_trace(go.Bar(
    x=[None], y=[None],
    marker=dict(color='white', pattern=dict(
            shape="x",
            fillmode="replace",
            size=10,
            solidity=0.2,
            fgcolor='black'  # Pattern color
        ), line=dict(color='black', width=1)),
    showlegend=True,
    name='Cross-model Steering',
    offset=-10,
))

# set title
if setting == '<b>w/o</b> Instr.':
    title = f'(a) Cross-model Steering: {setting}' 
else:
    title = f'(b) Cross-model Steering: {setting}'
fig.update_layout(title_text=title, title_font_size=15)


# move the legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-0.7,
    xanchor='right',
    x=0.8
))

fig.update_layout(width=300, height=250)
# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

fig.show()

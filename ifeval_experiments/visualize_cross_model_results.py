# %%
import os
import sys
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/ifeval_experiments')
    print('We\'re on the local machine')
elif 'cluster' in os.getcwd():
    os.chdir('/cluster/project/sachan/alessandro/llm-steer-instruct')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct/ifeval_experiments')
    print('We\'re on a sandbox machine')

import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from eval.evaluation_main import test_instruction_following_loose
import plotly
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
        
    # print the max length of the responses
    print(f'Max length of responses: {max([len(r.split()) for r in results_df.response])}')
    print(f'Max length of responses steering: {max([len(r.split()) for r in results_df_steering.response])}')
    print(f'Max length of responses steering cross: {max([len(r.split()) for r in results_df_steering_cross.response])}')
    print(f'Max length of responses standard: {max([len(r.split()) for r in results_df_standard.response])}')
    print(f'Max length of responses instr_plus_steering cross: {max([len(r.split()) for r in results_df_instr_plus_steering_cross.response])}')
    print(f'Max length of responses instr_plus_steering: {max([len(r.split()) for r in results_df_instr_plus_steering.response])}')

    # correct if accuracy scores by truncating the responses at the first occurence of 'Q:'
    def truncate_responses(df):
        new_rows = []
        for i, row in df.iterrows():
            if 'Q:' in row['response']:
                row['response'] = row['response'].split('Q:')[0]


            # compute accuracy
            prompt_to_response = {}
            prompt_to_response[row['prompt']] = row['response']
            output = test_instruction_following_loose(row, prompt_to_response)
            row['old_follow_all_instructions'] = row['follow_all_instructions']
            row['follow_all_instructions'] = output.follow_all_instructions
            new_rows.append(row)

        return pd.DataFrame(new_rows)

    results_df = truncate_responses(results_df)
    results_df_steering = truncate_responses(results_df_steering)
    results_df_steering_cross = truncate_responses(results_df_steering_cross)
    results_df_standard = truncate_responses(results_df_standard)
    results_df_instr_plus_steering = truncate_responses(results_df_instr_plus_steering)
    results_df_instr_plus_steering_cross = truncate_responses(results_df_instr_plus_steering_cross)


    # print overall accuracy
    print(f'Overall accuracy: {results_df.follow_all_instructions.mean()}')
    print(f'Overall accuracy steering: {results_df_steering.follow_all_instructions.mean()}')
    print(f'Overall accuracy steering cross: {results_df_steering_cross.follow_all_instructions.mean()}')
    print(f'Overall accuracy standard: {results_df_standard.follow_all_instructions.mean()}')
    print(f'Overall accuracy instr_plus_steering cross: {results_df_instr_plus_steering_cross.follow_all_instructions.mean()}')
    print(f'Overall accuracy instr_plus_steering: {results_df_instr_plus_steering.follow_all_instructions.mean()}')
    # Calculate overall accuracy
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

# incline the x-axis labels
# fig.update_xaxes(tickangle=30)

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

store=True

if store:
    # save plot as pdf
    if setting == '<b>w/o</b> Instr.':
        fig.write_image(f'./plots_for_paper/cross-model/no_instr.pdf')
    else:
        fig.write_image(f'./plots_for_paper/cross-model/instr.pdf')

    # save as png
    if setting == '<b>w/o</b> Instr.':
        fig.write_image(f'./plots_for_paper/cross-model/no_instr.png', scale=5)
    else:
        fig.write_image(f'./plots_for_paper/cross-model/instr.png', scale=5)

fig.show()


# %%
# =============================================================================
# category-wise metrics
# =============================================================================
all_instruct = list(set([ item for l in results_df.instruction_id_list for item in l]))
all_categories = [i.split(':')[0] for i in all_instruct]
# %%
category_corr = {cat: 0 for cat in all_categories}
category_corr_steering = {cat: 0 for cat in all_categories}
category_corr_steering_cross = {cat: 0 for cat in all_categories}
category_corr_standard = {cat: 0 for cat in all_categories}
category_corr_instr_plus_steering = {cat: 0 for cat in all_categories}
category_corr_instr_plus_steering_cross = {cat: 0 for cat in all_categories}
category_count = {cat: 0 for cat in all_categories}

instr_corr = {instr: 0 for instr in all_instruct}
instr_corr_steering = {instr: 0 for instr in all_instruct}
instr_corr_steering_cross = {instr: 0 for instr in all_instruct}
instr_corr_standard = {instr: 0 for instr in all_instruct}
instr_corr_instr_plus_steering = {instr: 0 for instr in all_instruct}
instr_corr_instr_plus_steering_cross = {instr: 0 for instr in all_instruct}
instr_count = {instr: 0 for instr in all_instruct}


for i, row in results_df.iterrows():
    for instr, corr in zip(row.instruction_id_list, row.follow_instruction_list):
        category = instr.split(':')[0]
        category_count[category] += 1
        category_corr[category] += corr
        corr_steering = results_df_steering.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        corr_steering_cross = results_df_steering_cross.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        corr_standard = results_df_standard.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        corr_instr_plus_steering = results_df_instr_plus_steering.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        corr_instr_plus_steering_cross = results_df_instr_plus_steering_cross.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        category_corr_steering[category] += corr_steering
        category_corr_steering_cross[category] += corr_steering_cross
        category_corr_standard[category] += corr_standard
        category_corr_instr_plus_steering[category] += corr_instr_plus_steering
        category_corr_instr_plus_steering_cross[category] += corr_instr_plus_steering_cross

        instr_count[instr] += 1
        instr_corr[instr] += corr
        instr_corr_steering[instr] += corr_steering
        instr_corr_steering_cross[instr] += corr_steering_cross
        instr_corr_standard[instr] += corr_standard
        instr_corr_instr_plus_steering[instr] += corr_instr_plus_steering
        instr_corr_instr_plus_steering_cross[instr] += corr_instr_plus_steering_cross


category_acc = {cat: category_corr[cat] / category_count[cat] for cat in all_categories}
category_acc_steering = {cat: category_corr_steering[cat] / category_count[cat] for cat in all_categories}
category_acc_steering_cross = {cat: category_corr_steering_cross[cat] / category_count[cat] for cat in all_categories}
category_acc_standard = {cat: category_corr_standard[cat] / category_count[cat] for cat in all_categories}
category_acc_instr_plus_steering = {cat: category_corr_instr_plus_steering[cat] / category_count[cat] for cat in all_categories}
category_acc_instr_plus_steering_cross = {cat: category_corr_instr_plus_steering_cross[cat] / category_count[cat] for cat in all_categories}

instr_acc = {instr: instr_corr[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_steering = {instr: instr_corr_steering[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_steering_cross = {instr: instr_corr_steering_cross[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_standard = {instr: instr_corr_standard[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_instr_plus_steering = {instr: instr_corr_instr_plus_steering[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_instr_plus_steering_cross = {instr: instr_corr_instr_plus_steering_cross[instr] / instr_count[instr] for instr in all_instruct}

# make histogram of category accuracie
df = pd.DataFrame({'Category': list(category_acc.keys()), 'Accuracy': list(category_acc.values()), 'Setting': '<b>w/o</b> Instr.'})
df_steering = pd.DataFrame({'Category': list(category_acc_steering.keys()), 'Accuracy': list(category_acc_steering.values()), 'Setting': '<b>w/o</b> Instr. + Steering'})
df_steering_cross = pd.DataFrame({'Category': list(category_acc_steering_cross.keys()), 'Accuracy': list(category_acc_steering_cross.values()), 'Setting': '<b>w/o</b> Instr. + Cross Steering'})
df_standard = pd.DataFrame({'Category': list(category_acc_standard.keys()), 'Accuracy': list(category_acc_standard.values()), 'Setting': '<b>w/</b> Instr.'})
df_instr_plus_steering = pd.DataFrame({'Category': list(category_acc_instr_plus_steering.keys()), 'Accuracy': list(category_acc_instr_plus_steering.values()), 'Setting': '<b>w/</b> Instr. + Steering'})
df_instr_plus_steering_cross = pd.DataFrame({'Category': list(category_acc_instr_plus_steering_cross.keys()), 'Accuracy': list(category_acc_instr_plus_steering_cross.values()), 'Setting': '<b>w/</b> Instr. + Cross Steering'})
df = pd.concat([df, df_steering, df_steering_cross, df_standard, df_instr_plus_steering, df_instr_plus_steering_cross])
fig = px.bar(df, x='Category', y='Accuracy', color='Setting', barmode='group')
# set title
fig.update_layout(title_text=f'Accuracy of {model_name} on IFEval (single-instruction only)')
# remove legend
fig.update_layout(showlegend=True)
# resize plot
#fig.update_layout(width=1000, height=600)
fig.show()

# %%
# make histogram of instruction accuracies
df = pd.DataFrame({'instruction': list(instr_acc.keys()), 'Accuracy': list(instr_acc.values()), 'Setting': '<b>w/o</b> Instr.'})
df = df.sort_values(by='instruction', ascending=False)
df_steering = pd.DataFrame({'instruction': list(instr_acc_steering.keys()), 'Accuracy': list(instr_acc_steering.values()), 'Setting': '<b>w/o</b> Instr. + Steering'})
df_steering = df_steering.sort_values(by='instruction', ascending=False)
df_steering_cross = pd.DataFrame({'instruction': list(instr_acc_steering_cross.keys()), 'Accuracy': list(instr_acc_steering_cross.values()), 'Setting': '<b>w/o</b> Instr. + Cross Steering'})
df_steering = df_steering.sort_values(by='instruction', ascending=False)
df_standard = pd.DataFrame({'instruction': list(instr_acc_standard.keys()), 'Accuracy': list(instr_acc_standard.values()), 'Setting': '<b>w/</b> Instr.'})
df_standard = df_standard.sort_values(by='instruction', ascending=False)
df_instr_plus_steering = pd.DataFrame({'instruction': list(instr_acc_instr_plus_steering.keys()), 'Accuracy': list(instr_acc_instr_plus_steering.values()), 'Setting': '<b>w/</b> Instr. + Steering'})
df_instr_plus_steering = df_instr_plus_steering.sort_values(by='instruction', ascending=False)
df_instr_plus_steering_cross = pd.DataFrame({'instruction': list(instr_acc_instr_plus_steering_cross.keys()), 'Accuracy': list(instr_acc_instr_plus_steering_cross.values()), 'Setting': '<b>w/</b> Instr. + Cross Steering'})
df_instr_plus_steering = df_instr_plus_steering.sort_values(by='instruction', ascending=False)
df = pd.concat([df, df_steering, df_steering_cross, df_standard, df_instr_plus_steering, df_instr_plus_steering_cross])

fig = px.bar(df, x='instruction', y='Accuracy', color='Setting', barmode='group')
# set title
fig.update_layout(title_text=f'Accuracy of {model_name} on IFEval (single-instruction only)')
# tilt the x-axis labels
fig.update_xaxes(tickangle=45)
# remove legend
fig.update_layout(showlegend=True)
fig.show()

# %%
# =============================================================================
# old code
# =============================================================================


model_name = 'gemma-2-2b'
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

mode = 'adjust_rs_-1'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_steering = pd.DataFrame(results)

mode = 'adjust_rs_-1_cross_model'
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

mode = 'instr_plus_adjust_rs_-1'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_instr_plus_steering = pd.DataFrame(results)

mode = 'instr_plus_adjust_rs_-1_cross_model'
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
# %%
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
    
# %%
# print the max length of the responses
print(f'Max length of responses: {max([len(r.split()) for r in results_df.response])}')
print(f'Max length of responses steering: {max([len(r.split()) for r in results_df_steering.response])}')
print(f'Max length of responses steering cross: {max([len(r.split()) for r in results_df_steering_cross.response])}')
print(f'Max length of responses standard: {max([len(r.split()) for r in results_df_standard.response])}')
print(f'Max length of responses instr_plus_steering cross: {max([len(r.split()) for r in results_df_instr_plus_steering_cross.response])}')
print(f'Max length of responses instr_plus_steering: {max([len(r.split()) for r in results_df_instr_plus_steering.response])}')

# %%
# correct if accuracy scores by truncating the responses at the first occurence of 'Q:'
def truncate_responses(df):
    new_rows = []
    for i, row in df.iterrows():
        if 'Q:' in row['response']:
            row['response'] = row['response'].split('Q:')[0]


        # compute accuracy
        prompt_to_response = {}
        prompt_to_response[row['prompt']] = row['response']
        output = test_instruction_following_loose(row, prompt_to_response)
        row['old_follow_all_instructions'] = row['follow_all_instructions']
        row['follow_all_instructions'] = output.follow_all_instructions
        new_rows.append(row)

    return pd.DataFrame(new_rows)

results_df = truncate_responses(results_df)
results_df_steering = truncate_responses(results_df_steering)
results_df_steering_cross = truncate_responses(results_df_steering_cross)
results_df_standard = truncate_responses(results_df_standard)
results_df_instr_plus_steering_cross = truncate_responses(results_df_instr_plus_steering_cross)


# %%
# print overall accuracy
print(f'Overall accuracy: {results_df.follow_all_instructions.mean()}')
print(f'Overall accuracy steering: {results_df_steering.follow_all_instructions.mean()}')
print(f'Overall accuracy steering cross: {results_df_steering_cross.follow_all_instructions.mean()}')
print(f'Overall accuracy standard: {results_df_standard.follow_all_instructions.mean()}')
print(f'Overall accuracy instr_plus_steering cross: {results_df_instr_plus_steering_cross.follow_all_instructions.mean()}')
print(f'Overall accuracy instr_plus_steering: {results_df_instr_plus_steering.follow_all_instructions.mean()}')
# Calculate overall accuracy
overall_accuracy = [
    {'setting': '<b>w/o</b> Instr.', 'steering': 'Std. Inference', 'accuracy': results_df.follow_all_instructions.mean(),},
    {'setting': '<b>w/o</b> Instr.', 'steering': 'Same-model Steering', 'accuracy': results_df_steering.follow_all_instructions.mean()},
    {'setting': '<b>w/o</b> Instr.', 'steering': 'Cross-model Steering', 'accuracy': results_df_steering_cross.follow_all_instructions.mean()},
    {'setting': '<b>w/</b> Instr.', 'steering': 'Std. Inference', 'accuracy': results_df_standard.follow_all_instructions.mean()},
    {'setting': '<b>w/</b> Instr.', 'steering': 'Same-model Steering', 'accuracy': results_df_instr_plus_steering.follow_all_instructions.mean()},
    {'setting': '<b>w/</b> Instr.', 'steering': 'Cross-model Steering', 'accuracy': results_df_instr_plus_steering_cross.follow_all_instructions.mean()}
]
 # %%
# =============================================================================
# visualize some examples
# =============================================================================

# get examples for which instructions were followed
#filtered_df = results_df_steering[results_df_steering.follow_all_instructions == True]

# filter results_df to only detectable_format instructions
# filtered_df = results_df_instr_plus_steering[results_df_instr_plus_steering.instruction_id_list.apply(lambda x: 'detectable_format:json_format' in x)]
# filtered_df = results_df_standard[results_df_standard.instruction_id_list.apply(lambda x: 'detectable_format:json_format' in x)]
# filtered_df = results_df_steering[results_df_steering.instruction_id_list.apply(lambda x: 'detectable_format:json_format' in x)]

df = results_df_steering_cross
filtered_df = df[df.instruction_id_list.apply(lambda x: 'change_case:english_lowercase' in x)]

# %%
for i, row in filtered_df.iterrows():
    print(f'instruction_id_list: {row.instruction_id_list}')
    print(f'follow_instruction_list: {row.follow_instruction_list}')
    print(f'Prompt: {row.prompt}')
    print(f'Prompt no_instr: {row.prompt_no_instr}')
    print(f'Output: {row.response}')
    print('-----------------------------------\n')
# %%

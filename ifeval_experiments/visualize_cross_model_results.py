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
# %%

model_name = 'phi-3'
# model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_name = 'mistral-7b-instruct'
model_name = 'gemma-2-9b'
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

# path_to_results = f'./ifeval_experiments/out/{model_name}/eval_results_{eval_type}.jsonl'
# with open(path_to_results) as f:
#     results = f.readlines()
#     results = [json.loads(r) for r in results]
# results_df_steering = pd.DataFrame(results)

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
    
# %%
# print the max length of the responses
print(f'Max length of responses: {max([len(r.split()) for r in results_df.response])}')
print(f'Max length of responses steering: {max([len(r.split()) for r in results_df_steering.response])}')
print(f'Max length of responses steering cross: {max([len(r.split()) for r in results_df_steering_cross.response])}')
print(f'Max length of responses standard: {max([len(r.split()) for r in results_df_standard.response])}')
print(f'Max length of responses instr_plus_steering cross: {max([len(r.split()) for r in results_df_instr_plus_steering_cross.response])}')
# print(f'Max length of responses instr_plus_steering: {max([len(r.split()) for r in results_df_instr_plus_steering.response])}')

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
all_instruct = list(set([ item for l in results_df.instruction_id_list for item in l]))
all_categories = [i.split(':')[0] for i in all_instruct]
# %%
category_corr = {cat: 0 for cat in all_categories}
category_corr_steering = {cat: 0 for cat in all_categories}
category_corr_standard = {cat: 0 for cat in all_categories}
category_corr_instr_plus_steering = {cat: 0 for cat in all_categories}
category_count = {cat: 0 for cat in all_categories}

instr_corr = {instr: 0 for instr in all_instruct}
instr_corr_steering = {instr: 0 for instr in all_instruct}
instr_corr_standard = {instr: 0 for instr in all_instruct}
instr_corr_instr_plus_steering = {instr: 0 for instr in all_instruct}
instr_count = {instr: 0 for instr in all_instruct}


for i, row in results_df.iterrows():
    for instr, corr in zip(row.instruction_id_list, row.follow_instruction_list):
        category = instr.split(':')[0]
        category_count[category] += 1
        category_corr[category] += corr
        corr_steering = results_df_steering.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        corr_standard = results_df_standard.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        # corr_instr_plus_steering = results_df_instr_plus_steering.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        category_corr_steering[category] += corr_steering
        category_corr_standard[category] += corr_standard
        # category_corr_instr_plus_steering[category] += corr_instr_plus_steering

        instr_count[instr] += 1
        instr_corr[instr] += corr
        instr_corr_steering[instr] += corr_steering
        instr_corr_standard[instr] += corr_standard
        # instr_corr_instr_plus_steering[instr] += corr_instr_plus_steering


category_acc = {cat: category_corr[cat] / category_count[cat] for cat in all_categories}
category_acc_steering = {cat: category_corr_steering[cat] / category_count[cat] for cat in all_categories}
category_acc_standard = {cat: category_corr_standard[cat] / category_count[cat] for cat in all_categories}
category_acc_instr_plus_steering = {cat: category_corr_instr_plus_steering[cat] / category_count[cat] for cat in all_categories}

instr_acc = {instr: instr_corr[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_steering = {instr: instr_corr_steering[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_standard = {instr: instr_corr_standard[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_instr_plus_steering = {instr: instr_corr_instr_plus_steering[instr] / instr_count[instr] for instr in all_instruct}
# %%
# print overall accuracy
print(f'Overall accuracy: {results_df.follow_all_instructions.mean()}')
print(f'Overall accuracy steering: {results_df_steering.follow_all_instructions.mean()}')
print(f'Overall accuracy steering cross: {results_df_steering_cross.follow_all_instructions.mean()}')
print(f'Overall accuracy standard: {results_df_standard.follow_all_instructions.mean()}')
print(f'Overall accuracy instr_plus_steering cross: {results_df_instr_plus_steering_cross.follow_all_instructions.mean()}')
# print(f'Overall accuracy instr_plus_steering: {results_df_instr_plus_steering.follow_all_instructions.mean()}')

# amke histogram of overall accuracy
df = pd.DataFrame({
    'Setting': ['w/o Instr.', 'w/o Instr. + Steering', '+ Cross Steering', 'w/ Instr.', 'w/Instr. + Cross St.'], # 'w/ Instr. + Steering'],
    'Accuracy': [
        results_df.follow_all_instructions.mean(),
        results_df_steering.follow_all_instructions.mean(),
        results_df_steering_cross.follow_all_instructions.mean(),
        results_df_standard.follow_all_instructions.mean(),
        results_df_instr_plus_steering_cross.follow_all_instructions.mean()
        # results_df_instr_plus_steering.follow_all_instructions.mean()
    ]
})

# Specify a list of colors for each 'Setting'
colors = px.colors.qualitative.Plotly

fig = px.bar(df, x='Setting', y='Accuracy', color='Setting',
             color_discrete_sequence=colors)

# set title
if nonparametric_only:
    title = f'{model_name.replace('-', ' ').capitalize()} on IFEval (single nonpar. instructions)'
else:
    title = f'{model_name.replace('-', ' ').capitalize()} on IFEval (single instr. only)'
fig.update_layout(title_text=title)
# remove legend
fig.update_layout(showlegend=False)
fig.update_layout(width=420, height=300)
# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

# save plot as pdf
# fig.write_image(f'./ifeval_experiments/figures/{model_name}_overall_accuracy.pdf')

fig.show()

# %%
# make histogram of category accuracie
df = pd.DataFrame({'Category': list(category_acc.keys()), 'Accuracy': list(category_acc.values()), 'Setting': 'w/o Instr.'})
df_steering = pd.DataFrame({'Category': list(category_acc_steering.keys()), 'Accuracy': list(category_acc_steering.values()), 'Setting': 'w/o Instr. + Steering'})
df_steering_cross = pd.DataFrame({'Category': list(category_acc_steering_cross.keys()), 'Accuracy': list(category_acc_steering_cross.values()), 'Setting': 'w/o Instr. + Cross Steering'})
df_standard = pd.DataFrame({'Category': list(category_acc_standard.keys()), 'Accuracy': list(category_acc_standard.values()), 'Setting': 'w/ Instr.'})
df_instr_plus_steering = pd.DataFrame({'Category': list(category_acc_instr_plus_steering.keys()), 'Accuracy': list(category_acc_instr_plus_steering.values()), 'Setting': 'w/ Instr. + Steering'})
df = pd.concat([df, df_steering, df_standard, df_instr_plus_steering])
fig = px.bar(df, x='Category', y='Accuracy', color='Setting', barmode='group')
# set title
fig.update_layout(title_text=f'Accuracy of {model_name} on IFEval (single-instruction only)')
# remove legend
fig.update_layout(showlegend=False)
# resize plot
#fig.update_layout(width=1000, height=600)
fig.show()

# %%
# make histogram of instruction accuracies
df = pd.DataFrame({'instruction': list(instr_acc.keys()), 'Accuracy': list(instr_acc.values()), 'Setting': 'w/o Instr.'})
df = df.sort_values(by='instruction', ascending=False)
df_steering = pd.DataFrame({'instruction': list(instr_acc_steering.keys()), 'Accuracy': list(instr_acc_steering.values()), 'Setting': 'w/o Instr. + Steering'})
df_steering = df_steering.sort_values(by='instruction', ascending=False)
df_standard = pd.DataFrame({'instruction': list(instr_acc_standard.keys()), 'Accuracy': list(instr_acc_standard.values()), 'Setting': 'w/ Instr.'})
df_standard = df_standard.sort_values(by='instruction', ascending=False)
df_instr_plus_steering = pd.DataFrame({'instruction': list(instr_acc_instr_plus_steering.keys()), 'Accuracy': list(instr_acc_instr_plus_steering.values()), 'Setting': 'w/ Instr. + Steering'})
df_instr_plus_steering = df_instr_plus_steering.sort_values(by='instruction', ascending=False)
df = pd.concat([df, df_steering, df_standard, df_instr_plus_steering])

fig = px.bar(df, x='instruction', y='Accuracy', color='Setting', barmode='group')
# set title
fig.update_layout(title_text=f'Accuracy of {model_name} on IFEval (single-instruction only)')
# tilt the x-axis labels
fig.update_xaxes(tickangle=45)
# remove legend
fig.update_layout(showlegend=False)
fig.show()


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
filtered_df = results_df_steering[results_df_steering.instruction_id_list.apply(lambda x: 'detectable_format:json_format' in x)]

# %%
for i, row in filtered_df.iterrows():
    print(f'instruction_id_list: {row.instruction_id_list}')
    print(f'follow_instruction_list: {row.follow_instruction_list}')
    print(f'Prompt: {row.prompt}')
    print(f'Prompt no_instr: {row.prompt_no_instr}')
    print(f'Output: {row.response}')
    print('-----------------------------------\n')
# %%

# load the dataframe from ./representations/phi-3/detectable_format_number_highlighted_sections.h5
df = pd.read_hdf('./ifeval_experiments/representations/phi-3/single_instr_subset_0.7/detectable_format_number_highlighted_sections.h5')
# %%
# =============================================================================
# Results for gemma 2B
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

# path_to_results = f'./ifeval_experiments/out/{model_name}/eval_results_{eval_type}.jsonl'
# with open(path_to_results) as f:
#     results = f.readlines()
#     results = [json.loads(r) for r in results]
# results_df_steering = pd.DataFrame(results)

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

mode = 'instr_plus_adjust_rs_-1_cross_model'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_instr_plus_steering_cross = pd.DataFrame(results)


mode = 'instr_plus_adjust_rs_-1'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/{subset}eval_results_{eval_type}.jsonl'
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
# %%
# join data_df with results_df on prompt
results_df = results_df.merge(data_df, on='prompt')
# results_df_steering = results_df_steering.merge(data_df, on='prompt')
results_df_steering_cross = results_df_steering_cross.merge(data_df, on='prompt')
results_df_standard = results_df_standard.merge(data_df, on='prompt')
# results_df_instr_plus_steering_cross = results_df_instr_plus_steering_cross.merge(data_df, on='prompt')

nonparametric_only = True
if nonparametric_only:
    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    results_df = results_df[results_df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df.reset_index(drop=True, inplace=True)
    # results_df_steering = results_df_steering[results_df_steering.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    # results_df_steering.reset_index(drop=True, inplace=True)
    results_df_steering_cross = results_df_steering_cross[results_df_steering_cross.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df_steering_cross.reset_index(drop=True, inplace=True)
    results_df_standard = results_df_standard[results_df_standard.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    results_df_standard.reset_index(drop=True, inplace=True)
    # results_df_instr_plus_steering_cross = results_df_instr_plus_steering_cross[results_df_instr_plus_steering_cross.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]
    # results_df_instr_plus_steering_cross.reset_index(drop=True, inplace=True)
    
# %%
# print the max length of the responses
print(f'Max length of responses: {max([len(r.split()) for r in results_df.response])}')
print(f'Max length of responses steering: {max([len(r.split()) for r in results_df_steering.response])}')
print(f'Max length of responses steering cross: {max([len(r.split()) for r in results_df_steering_cross.response])}')
print(f'Max length of responses standard: {max([len(r.split()) for r in results_df_standard.response])}')
print(f'Max length of responses instr_plus_steering cross: {max([len(r.split()) for r in results_df_instr_plus_steering_cross.response])}')
# print(f'Max length of responses instr_plus_steering: {max([len(r.split()) for r in results_df_instr_plus_steering.response])}')

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
all_instruct = list(set([ item for l in results_df.instruction_id_list for item in l]))
all_categories = [i.split(':')[0] for i in all_instruct]
# %%
category_corr = {cat: 0 for cat in all_categories}
category_corr_steering = {cat: 0 for cat in all_categories}
category_corr_standard = {cat: 0 for cat in all_categories}
category_corr_instr_plus_steering = {cat: 0 for cat in all_categories}
category_count = {cat: 0 for cat in all_categories}

instr_corr = {instr: 0 for instr in all_instruct}
instr_corr_steering = {instr: 0 for instr in all_instruct}
instr_corr_standard = {instr: 0 for instr in all_instruct}
instr_corr_instr_plus_steering = {instr: 0 for instr in all_instruct}
instr_count = {instr: 0 for instr in all_instruct}


for i, row in results_df.iterrows():
    for instr, corr in zip(row.instruction_id_list, row.follow_instruction_list):
        category = instr.split(':')[0]
        category_count[category] += 1
        category_corr[category] += corr
        corr_steering = results_df_steering.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        corr_standard = results_df_standard.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        # corr_instr_plus_steering = results_df_instr_plus_steering.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        category_corr_steering[category] += corr_steering
        category_corr_standard[category] += corr_standard
        # category_corr_instr_plus_steering[category] += corr_instr_plus_steering

        instr_count[instr] += 1
        instr_corr[instr] += corr
        instr_corr_steering[instr] += corr_steering
        instr_corr_standard[instr] += corr_standard
        # instr_corr_instr_plus_steering[instr] += corr_instr_plus_steering


category_acc = {cat: category_corr[cat] / category_count[cat] for cat in all_categories}
category_acc_steering = {cat: category_corr_steering[cat] / category_count[cat] for cat in all_categories}
category_acc_standard = {cat: category_corr_standard[cat] / category_count[cat] for cat in all_categories}
category_acc_instr_plus_steering = {cat: category_corr_instr_plus_steering[cat] / category_count[cat] for cat in all_categories}

instr_acc = {instr: instr_corr[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_steering = {instr: instr_corr_steering[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_standard = {instr: instr_corr_standard[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_instr_plus_steering = {instr: instr_corr_instr_plus_steering[instr] / instr_count[instr] for instr in all_instruct}
# %%
# print overall accuracy
print(f'Overall accuracy: {results_df.follow_all_instructions.mean()}')
print(f'Overall accuracy steering: {results_df_steering.follow_all_instructions.mean()}')
print(f'Overall accuracy steering cross: {results_df_steering_cross.follow_all_instructions.mean()}')
print(f'Overall accuracy standard: {results_df_standard.follow_all_instructions.mean()}')
print(f'Overall accuracy instr_plus_steering cross: {results_df_instr_plus_steering_cross.follow_all_instructions.mean()}')
# print(f'Overall accuracy instr_plus_steering: {results_df_instr_plus_steering.follow_all_instructions.mean()}')

# amke histogram of overall accuracy
df = pd.DataFrame({
    'Setting': ['w/o Instr.', 'w/o Instr. + Steering', '+ Cross Steering', 'w/ Instr.', 'w/Instr. + Cross St.'], # 'w/ Instr. + Steering'],
    'Accuracy': [
        results_df.follow_all_instructions.mean(),
        results_df_steering.follow_all_instructions.mean(),
        results_df_steering_cross.follow_all_instructions.mean(),
        results_df_standard.follow_all_instructions.mean(),
        results_df_instr_plus_steering_cross.follow_all_instructions.mean()
        # results_df_instr_plus_steering.follow_all_instructions.mean()
    ]
})

# Specify a list of colors for each 'Setting'
colors = px.colors.qualitative.Plotly

fig = px.bar(df, x='Setting', y='Accuracy', color='Setting',
             color_discrete_sequence=colors)

# set title
if nonparametric_only:
    title = f'{model_name.replace('-', ' ').capitalize()} on IFEval (single nonpar. instructions)'
else:
    title = f'{model_name.replace('-', ' ').capitalize()} on IFEval (single instr. only)'
fig.update_layout(title_text=title)
# remove legend
fig.update_layout(showlegend=False)
fig.update_layout(width=420, height=300)
# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

# save plot as pdf
# fig.write_image(f'./ifeval_experiments/figures/{model_name}_overall_accuracy.pdf')

fig.show()
# %%

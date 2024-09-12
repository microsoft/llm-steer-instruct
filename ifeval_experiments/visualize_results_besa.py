# %%
import os
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
    print('We\'re on a Windows machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on a sandbox machine')

import pandas as pd
import json
import plotly.express as px
# %%

model_name = 'phi-3'
single_instr = 'all_instr'
mode = 'no_instr'
mode = 'standard'

eval_type = 'loose'

path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df = pd.DataFrame(results)


eval_type = 'strict'
path_to_results = f'./ifeval_experiments/out/{model_name}/{single_instr}/{mode}/eval_results_{eval_type}.jsonl'
with open(path_to_results) as f:
    results = f.readlines()
    results = [json.loads(r) for r in results]
results_df_strict = pd.DataFrame(results)


# %%
# join data_df with results_df on prompt
results_df = results_df.merge(data_df, on='prompt')
results_df_steering = results_df_steering.merge(data_df, on='prompt')
results_df_standard = results_df_standard.merge(data_df, on='prompt')
results_df_instr_plus_steering = results_df_instr_plus_steering.merge(data_df, on='prompt')


# %%
all_instruct = list(set([ item for l in results_df.instruction_id_list for item in l]))
all_categories = [i.split(':')[0] for i in all_instruct]
# %%
category_corr = {cat: 0 for cat in all_categories}
category_corr_strict = {cat: 0 for cat in all_categories}
category_count = {cat: 0 for cat in all_categories}

instr_corr = {instr: 0 for instr in all_instruct}
instr_corr_strict = {instr: 0 for instr in all_instruct}
instr_count = {instr: 0 for instr in all_instruct}

for i, row in results_df.iterrows():
    for instr, corr in zip(row.instruction_id_list, row.follow_instruction_list):
        category = instr.split(':')[0]
        category_count[category] += 1
        category_corr[category] += corr
        corr_strict = results_df_strict.iloc[i].follow_instruction_list[row.instruction_id_list.index(instr)]
        category_corr_strict[category] += corr_strict

        instr_count[instr] += 1
        instr_corr[instr] += corr
        instr_corr_strict[instr] += corr_strict

category_acc = {cat: category_corr[cat] / category_count[cat] for cat in all_categories}
category_acc_strict = {cat: category_corr_strict[cat] / category_count[cat] for cat in all_categories}

instr_acc = {instr: instr_corr[instr] / instr_count[instr] for instr in all_instruct}
instr_acc_strict = {instr: instr_corr_strict[instr] / instr_count[instr] for instr in all_instruct}
# %%
# print overall accuracy
print(f'Overall accuracy: {results_df.follow_all_instructions.mean()}')
print(f'Overall accuracy strict: {results_df_strict.follow_all_instructions.mean()}')

# amke histogram of overall accuracy
df = pd.DataFrame({
    'Evaluation Type': ['Strict', 'Loose'],
    'Accuracy': [
        results_df_strict.follow_all_instructions.mean(),
        results_df.follow_all_instructions.mean(),
    ]
})

# Specify a list of colors for each 'Evaluation Type'
colors = px.colors.qualitative.Plotly

fig = px.bar(df, x='Evaluation Type', y='Accuracy', color='Evaluation Type',
             color_discrete_sequence=colors)

# set title
fig.update_layout(title_text=f'Overall accuracy of {model_name} on IFEval')
fig.show()
# %%
# make histogram of category accuracie
df = pd.DataFrame({'Category': list(category_acc.keys()), 'Accuracy': list(category_acc.values()), 'Evaluation Type': 'Loose'})
df = df.sort_values('Category')
df_strict = pd.DataFrame({'Category': list(category_acc_strict.keys()), 'Accuracy': list(category_acc_strict.values()), 'Evaluation Type': 'Strict'})
df_strict = df_strict.sort_values('Category')
df = pd.concat([df_strict, df])
fig = px.bar(df, x='Category', y='Accuracy', color='Evaluation Type', barmode='group')
# set title
fig.update_layout(title_text=f'Evaluation of Phi-3-mini-4k-instruct on IFEval')
fig.show()
# save the plot as pdf
fig.write_image(f'./plots/{model_name}_category_accuracy.pdf')

# %%
# make histogram of instruction accuracies
df = pd.DataFrame({'Instruction': list(instr_acc.keys()), 'Accuracy': list(instr_acc.values()), 'Evaluation Type': 'Loose'})
df = df.sort_values('Instruction')
df_strict = pd.DataFrame({'Instruction': list(instr_acc_strict.keys()), 'Accuracy': list(instr_acc_strict.values()), 'Evaluation Type': 'Strict'})
df_strict = df_strict.sort_values('Instruction')
df = pd.concat([df_strict, df])
fig = px.bar(df, x='Instruction', y='Accuracy', color='Evaluation Type', barmode='group')
# set title
fig.update_layout(title_text=f'Evaluation of Phi-3-mini-4k-instruct on IFEval')
# tilt the x-axis labels
fig.update_xaxes(tickangle=45)
fig.show()



# %%
# =============================================================================
# visualize some examples
# =============================================================================

# get examples for which instructions were followed
#filtered_df = results_df_steering[results_df_steering.follow_all_instructions == True]

# filter results_df to only detectable_format instructions
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

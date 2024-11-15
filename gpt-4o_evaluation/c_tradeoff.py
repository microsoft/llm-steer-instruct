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
from transformers import AutoTokenizer
# %%
import os
import json
import pandas as pd

folder = 'gpt-4o_evaluation/30-09-2024_gpt-4o_eval/30-09-2024_gpt-4o_eval/keyword_exclusion'
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
avg_scores_instr = {}
avg_scores_no_instr = {}

avg_accuracy_instr = {}
avg_accuracy_no_instr = {}

avg_broken_outputs_instr = {}
avg_broken_outputs_no_instr = {}

with open('./hf_token.txt') as f:
    hf_token = f.read()
model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name_hf, token=hf_token)

with open('./hf_token.txt') as f:
    hf_token = f.read()
model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'

for i, sett in enumerate(settings):
    print(f'Setting {i}: {sett}')
    data_df = setting_dfs[sett]
    qual_scores = []
    for i, r in data_df.iterrows(): 
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
        are_valid = r['model_answers']['is_answer_valid']
        answers = r['model_answers']['answers']
        partial_score = 0
        for idx, is_valid in enumerate(are_valid):
            if is_valid:
                partial_score += answers[idx]
        qual_scores.append(partial_score/sum(are_valid))

    data_df['qual_score'] = qual_scores


    broken_outputs = []
    accuracy_with_quality_check = []
    for i, r in data_df.iterrows():
        # compute accuracy

        response  = r['response']
        tokens = tokenizer.tokenize(response)
        print(f'Length of tokens: {len(tokens)}')
        counter = Counter(tokens)
        #remove '▁the' '▁' from the counter
        if '▁the' in counter:
            del counter['▁the']
        if '▁' in counter:
            del counter['▁']
        if ',' in counter:
            del counter[',']
        if '.' in counter:
            del counter['.']
        # take the number of occurrences of the most common token
        most_common = counter.most_common(1)[0][1]
        # get most common token
        if most_common > 50:
            broken_outputs.append(1)
            if sett == 'standard' or sett == 'no_instr':
                print(f'Broken output: {response}')
                print(f'Most common token: {counter.most_common(1)}')
        else:
            broken_outputs.append(0)

    data_df['broken_output'] = broken_outputs   

    
    if sett == 'standard':
        weight = -1
        avg_scores_instr[weight] = data_df['qual_score'].mean()
        avg_accuracy_instr[weight] = data_df['follow_all_instructions'].mean()
        avg_broken_outputs_instr[weight] = data_df['broken_output'].mean()
    elif sett == 'no_instr':
        weight = -1
        avg_scores_no_instr[weight] = data_df['qual_score'].mean()
        avg_accuracy_no_instr[weight] = data_df['follow_all_instructions'].mean()
        avg_broken_outputs_no_instr[weight] = data_df['broken_output'].mean()
    elif 'instr' in sett:
        weight = sett.split('_')[-1]
        avg_scores_instr[weight] = data_df['qual_score'].mean()
        avg_accuracy_instr[weight] = data_df['follow_all_instructions'].mean()
        avg_broken_outputs_instr[weight] = data_df['broken_output'].mean()
    else:
        weight = sett.split('_')[-1]
        avg_scores_no_instr[weight] = data_df['qual_score'].mean()
        avg_accuracy_no_instr[weight] = data_df['follow_all_instructions'].mean()
        avg_broken_outputs_no_instr[weight] = data_df['broken_output'].mean()

    
    



# %%
print(avg_scores_no_instr)
print(avg_scores_instr)
# %%
# sort the dictionaries by key
from collections import OrderedDict

avg_scores_instr = OrderedDict(sorted(avg_scores_instr.items(), key=lambda item: -int(item[0])))
avg_scores_no_instr = OrderedDict(sorted(avg_scores_no_instr.items(), key=lambda item: -int(item[0])))

# make two line plots: one for instr and one for no_instr, weigth as x, avg qual score as y
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_scores_instr.keys()), y=list(avg_scores_instr.values()), mode='lines+markers', name='With instructions'))
fig.add_trace(go.Scatter(x=list(avg_scores_no_instr.keys()), y=list(avg_scores_no_instr.values()), mode='lines+markers', name='Without instructions'))
fig.update_layout(title='Average quality score per setting', xaxis_title='Weight', yaxis_title='Average quality score')
fig.show()
# %%
# sort the dictionaries by key and convert them into OrderedDict
avg_accuracy_instr = OrderedDict(sorted(avg_accuracy_instr.items(), key=lambda item: -int(item[0])))
avg_accuracy_no_instr = OrderedDict(sorted(avg_accuracy_no_instr.items(), key=lambda item: -int(item[0])))

# make two line plots: one for instr and one for no_instr, weigth as x, avg qual score as y
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_accuracy_instr.keys()), y=list(avg_accuracy_instr.values()), mode='lines+markers', name='With instructions'))
fig.add_trace(go.Scatter(x=list(avg_accuracy_no_instr.keys()), y=list(avg_accuracy_no_instr.values()), mode='lines+markers', name='Without instructions'))
fig.update_layout(title='Average accuracy per setting', xaxis_title='Weight', yaxis_title='Average accuracy')
fig.show()
# %%
# sort the dictionaries by key
avg_broken_outputs_instr = OrderedDict(sorted(avg_broken_outputs_instr.items(), key=lambda item: -int(item[0])))
avg_broken_outputs_no_instr = OrderedDict(sorted(avg_broken_outputs_no_instr.items(), key=lambda item: -int(item[0])))

# make two line plots: one for instr and one for no_instr, weigth as x, avg qual score as y
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_broken_outputs_instr.keys()), y=list(avg_broken_outputs_instr.values()), mode='lines+markers', name='With instructions'))
fig.add_trace(go.Scatter(x=list(avg_broken_outputs_no_instr.keys()), y=list(avg_broken_outputs_no_instr.values()), mode='lines+markers', name='Without instructions'))
fig.update_layout(title='Average broken outputs per setting', xaxis_title='Weight', yaxis_title='Average broken outputs')
fig.show()
# %%
# make scatter plot of broken outputs and quality scores
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_broken_outputs_instr.values()), y=list(avg_scores_instr.values()), mode='markers', name='With instructions'))
fig.add_trace(go.Scatter(x=list(avg_broken_outputs_no_instr.values()), y=list(avg_scores_no_instr.values()), mode='markers', name='Without instructions'))
fig.update_layout(title='Broken outputs vs quality scores', xaxis_title='Broken outputs', yaxis_title='Quality scores')
fig.show()
# %%

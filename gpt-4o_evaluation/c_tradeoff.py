# %%
import os
os.chdir('/Users/alestolfo/workspace/llm-steer-instruct')
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly
from scipy.stats import ttest_ind, ttest_rel
import numpy as np
from collections import Counter
from utils.generation_utils import compute_perplexity
from tqdm import tqdm
# %%
import os
import json
import pandas as pd

# folder = 'gpt-4o_evaluation/30-09-2024_gpt-4o_eval/30-09-2024_gpt-4o_eval/keyword_exclusion'
folder = 'gpt-4o_evaluation/16-11-2024_quality_check/c_sweep'
setting_dfs = {}
settings = []
paths = {}

for i, setting_folder in enumerate(os.listdir(folder)):
    if setting_folder == '.DS_Store':
        continue
    path = os.path.join(folder, setting_folder, 'output', 'answer_post_processing_output', 'transformed_data.jsonl')
    with open(path, 'r') as f:
        results = [json.loads(line) for line in f]
    data_df = pd.DataFrame(results)
    setting_dfs[setting_folder] = data_df
    settings.append(setting_folder)
    paths[setting_folder] = path

for i, sett in enumerate(settings):
    print(f'Setting {i}: {sett}')

# %%

device = 'mps'
perplexity_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
perplexity_model.to(device)
perplexity_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')


# %%


with open('./hf_token.txt') as f:
    hf_token = f.read()
model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name_hf, token=hf_token)

with open('./hf_token.txt') as f:
    hf_token = f.read()
model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'

total = len(settings) * len(setting_dfs[settings[0]])
p_bar = tqdm(total=total)

for i, sett in enumerate(settings):
    print(f'Setting {i}: {sett}')
    data_df = setting_dfs[sett]
    
    # check if perplexity is already computed
    if 'perplexity' in data_df.columns:
        print('Perplexity already computed')
        continue

    qual_scores = []
    perplexities = []
    broken_outputs = []
    accuracy_with_quality_check = []
    for i, r in data_df.iterrows(): 
        if r['model_answers'].__len__() == 0:
                # print('Empty')
                # append nan
                qual_scores.append(np.nan)
        elif sum(r['model_answers']['is_answer_valid']) == 0:
            # print('No valid answers')
            # append nan
            qual_scores.append(np.nan)
        else:
            are_valid = r['model_answers']['is_answer_valid']
            answers = r['model_answers']['answers']
            partial_score = 0
            for idx, is_valid in enumerate(are_valid):
                if is_valid:
                    partial_score += answers[idx]
            qual_scores.append(partial_score/sum(are_valid))

        # compute perplexity
        response = r['response']
        perplexity = compute_perplexity(response, device, perplexity_model, perplexity_tokenizer)
        perplexities.append(perplexity)
    
        # compute broken outputs
        tokens = tokenizer.tokenize(response)
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
        else:
            broken_outputs.append(0)
        
        p_bar.update(1)

    data_df['broken_output'] = broken_outputs

    data_df['qual_score'] = qual_scores
    data_df['perplexity'] = perplexities

    # store the new results_df as a jsonl file
    new_path = f'{paths[sett].replace("transformed_data.jsonl", "transformed_data_with_perplexity.jsonl")}'
    print(f'storing new file at {new_path}')
    data_df.to_json(new_path, orient='records', lines=True)

# %%
avg_scores_instr = {}
avg_scores_no_instr = {}

avg_accuracy_instr = {}
avg_accuracy_no_instr = {}

avg_broken_outputs_instr = {}
avg_broken_outputs_no_instr = {}

avg_low_perplexity_instr = {}
avg_low_perplexity_no_instr = {}

for i, sett in enumerate(settings):
    data_df = setting_dfs[sett]

    data_df['low_perplexity'] = data_df['perplexity'] < 2.5

    if sett == 'standard':
        weight = 0
        avg_scores_instr[weight] = data_df['qual_score'].mean()
        avg_accuracy_instr[weight] = data_df['follow_all_instructions'].mean()
        avg_broken_outputs_instr[weight] = data_df['broken_output'].mean()
        avg_low_perplexity_instr[weight] = data_df['low_perplexity'].mean()
    elif sett == 'no_instr':
        weight = 0
        avg_scores_no_instr[weight] = data_df['qual_score'].mean()
        avg_accuracy_no_instr[weight] = data_df['follow_all_instructions'].mean()
        avg_broken_outputs_no_instr[weight] = data_df['broken_output'].mean()
        avg_low_perplexity_no_instr[weight] = data_df['low_perplexity'].mean()
    elif 'instr' in sett:
        weight = sett.split('_')[-1]
        avg_scores_instr[weight] = data_df['qual_score'].mean()
        avg_accuracy_instr[weight] = data_df['follow_all_instructions'].mean()
        avg_broken_outputs_instr[weight] = data_df['broken_output'].mean()
        avg_low_perplexity_instr[weight] = data_df['low_perplexity'].mean()
    else:
        weight = sett.split('_')[-1]
        avg_scores_no_instr[weight] = data_df['qual_score'].mean()
        avg_accuracy_no_instr[weight] = data_df['follow_all_instructions'].mean()
        avg_broken_outputs_no_instr[weight] = data_df['broken_output'].mean()
        avg_low_perplexity_no_instr[weight] = data_df['low_perplexity'].mean()
    
    


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
avg_low_perplexity_instr = OrderedDict(sorted(avg_low_perplexity_instr.items(), key=lambda item: -int(item[0])))
avg_low_perplexity_no_instr = OrderedDict(sorted(avg_low_perplexity_no_instr.items(), key=lambda item: -int(item[0])))

# make two line plots: one for instr and one for no_instr, weigth as x, avg qual score as y
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_low_perplexity_instr.keys()), y=list(avg_low_perplexity_instr.values()), mode='lines+markers', name='With instructions'))
fig.add_trace(go.Scatter(x=list(avg_low_perplexity_no_instr.keys()), y=list(avg_low_perplexity_no_instr.values()), mode='lines+markers', name='Without instructions'))
fig.update_layout(title='Average low perplexity per setting', xaxis_title='Weight', yaxis_title='Average low perplexity')
fig.show()
# %%
# make scatter plot of low perplexity and quality scores
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_low_perplexity_instr.values()), y=list(avg_scores_instr.values()), mode='markers', name='With instructions'))
fig.add_trace(go.Scatter(x=list(avg_low_perplexity_no_instr.values()), y=list(avg_scores_no_instr.values()), mode='markers', name='Without instructions'))

fig.update_layout(title='Low perplexity vs quality scores', xaxis_title='Low perplexity', yaxis_title='Quality scores')
fig.show()

# compute correlation between low perplexity and quality scores
low_perplexity_instr = list(avg_low_perplexity_instr.values())
scores_instr = list(avg_scores_instr.values())
low_perplexity_no_instr = list(avg_low_perplexity_no_instr.values())
scores_no_instr = list(avg_scores_no_instr.values())

corr_instr = np.corrcoef(low_perplexity_instr, scores_instr)
corr_no_instr = np.corrcoef(low_perplexity_no_instr, scores_no_instr)

print(f'Correlation between low perplexity and quality scores with instructions: {corr_instr[0, 1]}')
print(f'Correlation between low perplexity and quality scores without instructions: {corr_no_instr[0, 1]}')

# %%
# make scatter plot of broken outputs and quality scores
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_broken_outputs_instr.values()), y=list(avg_scores_instr.values()), mode='markers', name='With instructions'))
fig.add_trace(go.Scatter(x=list(avg_broken_outputs_no_instr.values()), y=list(avg_scores_no_instr.values()), mode='markers', name='Without instructions'))
fig.update_layout(title='Broken outputs vs quality scores', xaxis_title='Broken outputs', yaxis_title='Quality scores')
fig.show()
# %%
# make two line plots: one for avg qual score and one for avg accuracy, weigth as x, avg qual score as y for instr
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_scores_instr.keys()), y=list(avg_scores_instr.values()), mode='lines+markers', name='Quality Score'))
fig.add_trace(go.Scatter(x=list(avg_accuracy_instr.keys()), y=list(avg_accuracy_instr.values()), mode='lines+markers', name='Accuracy'))


fig.update_layout(title='(b) Quality Score vs. Accuracy: <b>w/</b> Instr.', xaxis_title='Steering Weight <i>c</i>', yaxis_title='Value')

# resize plot
fig.update_layout(width=400, height=250)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# move legend to the bottom
fig.update_layout(legend=dict(
    yanchor="bottom",
    y=-.5,
    xanchor="center",
    x=0.5,
    orientation="h"
))

fig.show()

# store plot as pdf
fig.write_image(f'./plots_for_paper/quality_score/c_tradeoff_with_instr.pdf')



# %%
# make two line plots: one for avg qual score and one for avg accuracy, weigth as x, avg qual score as y for no_instr
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(avg_scores_no_instr.keys()), y=list(avg_scores_no_instr.values()), mode='lines+markers', name='Quality score'))
fig.add_trace(go.Scatter(x=list(avg_accuracy_no_instr.keys()), y=list(avg_accuracy_no_instr.values()), mode='lines+markers', name='Accuracy'))

fig.update_layout(title='(a) Quality Score vs. Accuracy: <b>w/o</b> Instr.', xaxis_title='Steering Weight <i>c</i>', yaxis_title='Value')

# resize plot
fig.update_layout(width=400, height=250)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# move legend to the bottom
fig.update_layout(legend=dict(
    yanchor="bottom",
    y=-.5,
    xanchor="center",
    x=0.5,
    orientation="h"
))

# set y-axis range
fig.update_layout(yaxis=dict(range=[0.6, 0.9]))

fig.show()

# store plot as pdf
fig.write_image(f'./plots_for_paper/quality_score/c_tradeoff_no_instr.pdf')


# %%

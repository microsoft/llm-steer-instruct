# %%
import os

if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on a sandbox machine')
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from collections import Counter
import torch
import numpy as np



# %%
model_name = 'phi-3'
# model_name = 'gemma-2-2b-it'
folder = f'./keywords/out/{model_name}/forbidden_validation/'
# folder = f'./keywords/out/{model_name}/existence_validation/'
# folder = f'./keywords/out/{model_name}/forbidden_validation_w_forbidden_rep/'
file_name = 'out_gen_data_perplexity.jsonl'
subfolders = os.listdir(folder)
result_dict = {}
paths_dict = {}
for subfolder in subfolders:
    print(subfolder)
    if 'adjust' in subfolder:
        continue
    if subfolder == 'no_instr' :
        layer = -1
        weight = -1
    else:
        layer = subfolder.split('_')[2]
        weight = subfolder.split('_')[5]
    print(os.listdir(folder + subfolder))
    if file_name not in os.listdir(folder + subfolder):
        print(f'{subfolder} does not have the file {file_name}')
        continue
    with open(folder + subfolder + f'/{file_name}' ) as f:
        results = [json.loads(line) for line in f]
    # file_name = 'out_gen_data_perplexity.jsonl'
    results_df = pd.DataFrame(results)
    result_dict[(int(layer), int(weight))] = results_df
    paths_dict[(int(layer), int(weight))] = folder + subfolder + f'/{file_name}'


# %%
# load model tokenizer

from transformers import AutoTokenizer
with open('./hf_token.txt') as f:
    hf_token = f.read()
if model_name == 'phi-3':
    model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'
elif model_name == 'gemma-2-2b-it':
    model_name_hf = 'google/gemma-2-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name_hf, token=hf_token)



device = 'mps'
perplexity_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
perplexity_model.to(device)
perplexity_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

def compute_perplexity(text):
    # Tokenize the input text
    inputs = perplexity_tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    # if longer than 1024 tokens, take the last 1024 tokens
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, -1024:]

    # Compute the log probabilities
    with torch.no_grad():
        outputs = perplexity_model(input_ids, labels=input_ids)
        loss = outputs.loss  # This is the average negative log-likelihood per token

    # Compute the perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

# %%
# =============================================================================
# compute perplexity for each response. Skip if the perplexity is already computed
# =============================================================================


skip = False
if 'perplexity' in list(result_dict.values())[0].columns:
    skip = True

skip = False

if not skip:
    accuracy_dict = {}
    broken_outputs_dict = {}
    lengths_dict = {}
    perplexitiy_dict = {}

    total = len(result_dict) * len(list(result_dict.values())[0])
    p_bar = tqdm(total=total)

    for key, value in list(result_dict.items()):
        # check if the perplexity is already computed
        if 'perplexity' in value.columns:
            print(f'Perplexity already computed for {key}')
            continue


        accuracy_dict[key] = value['follow_all_instructions'].mean()
        broken_outputs = []
        lengths = []
        perplexities = []
        for i, row in value.iterrows():
            response = row['response']
            tokens = tokenizer.tokenize(response)
            lengths.append(len(tokens))
            counter = Counter(tokens)
            # remove '▁the' and ',' from the counter
            counter.pop('▁the', None)
            counter.pop(',', None)
            counter.pop('.', None)
            # take the number of occurrences of the most common token
            most_common = counter.most_common(1)[0][1]
            # get most common token
            if most_common > 50:
                # print(f'key: {key}')    
                # print(f'Broken output: {response}')
                # # print most common token
                # print(counter.most_common(1))
                broken_outputs.append(1)
            else:
                broken_outputs.append(0)

            # compute perplexity
            perplexities.append(compute_perplexity(response))

            p_bar.update(1)

        value['broken_output'] = broken_outputs
        value['length'] = lengths
        value['perplexity'] = perplexities
        lengths_dict[key] = sum(lengths) / len(lengths)
        broken_outputs_dict[key] = sum(broken_outputs) / len(broken_outputs)
        perplexitiy_dict[key] = sum(perplexities) / len(perplexities)

        # store the updated dataframe as jsonl
        new_path = paths_dict[key].replace('.jsonl', '_perplexity.jsonl')
        print(f'Saving the file at {new_path}')
        value.to_json(new_path, orient='records', lines=True)


# %%
accuracy_dict = {}
broken_outputs_dict = {}
lengths_dict = {}
low_perplexity_dict = {}
for key, value in list(result_dict.items()):
    value['low_perplexity'] = value['perplexity'] < 4
    low_perplexity_dict[key] = value['low_perplexity'].mean()
    accuracy_dict[key] = value['follow_all_instructions'].mean()
    broken_outputs_dict[key] = value['broken_output'].mean()
    lengths_dict[key] = value['length'].mean()
# %%
accuracy_dict
# %%
broken_outputs_dict
# %%
lengths_dict
# %%
low_perplexity_dict
# %%
df = result_dict[(-1, -1)]
filtered_df = df[df.broken_output == 1]   
for i, r in filtered_df.iterrows():
    print(r['response'])
    print('---')
# %%
layers = [key[0] for key in result_dict.keys()]
weights = [key[1] for key in result_dict.keys()]

for layer in set(layers):
    for weight in set(weights):
        if (layer, weight) in result_dict.keys():
            print(f'Layer {layer}, weight {weight}')
            print(f'Accuracy: {accuracy_dict[(layer, weight)]}')
            print(f'Average length: {lengths_dict[(layer, weight)]}')
            print(f'Broken outputs: {broken_outputs_dict[(layer, weight)]}')
            print(f'Low perplexity: {low_perplexity_dict[(layer, weight)]}')
            print('---')

# %%
# make line plot of the accuracy values and broken outputs
accuracies = []
broken_outputs = []
perplexities = []
x_labels = []
for (layer, weight), accuracy in accuracy_dict.items():
    accuracies.append(accuracy)
    broken_outputs.append(broken_outputs_dict[(layer, weight)])
    perplexities.append(low_perplexity_dict[(layer, weight)])
    x_labels.append(f'L{layer}_W{weight}')

# sort the lists according to the x_labels
x_labels, accuracies, broken_outputs, perplexities = zip(*sorted(zip(x_labels, accuracies, broken_outputs, perplexities)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_labels, y=accuracies, mode='lines+markers', name='Accuracy'))
fig.add_trace(go.Scatter(x=x_labels, y=broken_outputs, mode='lines+markers', name='Broken Outputs'))
fig.add_trace(go.Scatter(x=x_labels, y=perplexities, mode='lines+markers', name='Perplexity'))
fig.update_layout(xaxis_tickangle=-45)
# add horizontal line at broken_outputs[(-1,-1)]
fig.add_shape(type="line", x0=0, x1=len(x_labels), y0=broken_outputs[0], y1=broken_outputs[0], line=dict(color="red", width=1))

task = 'Inclusion' if 'existence' in folder else 'Exclusion'
# add title and labels
fig.update_layout(title=f'Accuracy and Broken Outputs for {model_name} on Keyword {task}', xaxis_title='Layer-Weight Combination', yaxis_title='Value')


fig.show()

# %%

# Find the index of the point with broken_output value 0 and the largest accuracy
max_accuracy_idx = None
max_accuracy = -float('inf')
for i, (accuracy, broken_output) in enumerate(zip(accuracies, broken_outputs)):
    if broken_output == 0 and accuracy > max_accuracy:
        max_accuracy = accuracy
        max_accuracy_idx = i

# make scatter plot of the accuracy values and broken outputs
fig = go.Figure()

# Add all points
fig.add_trace(go.Scatter(
    x=accuracies,
    y=broken_outputs,
    mode='markers',
    text=x_labels,
    name='Accuracy vs Broken Outputs'
))

# Highlight the point with broken_output value 0 and the largest accuracy
if max_accuracy_idx is not None:
    fig.add_trace(go.Scatter(
        x=[accuracies[max_accuracy_idx]],
        y=[broken_outputs[max_accuracy_idx]],
        mode='markers',
        marker=dict(color='red', size=9),
        text=[x_labels[max_accuracy_idx]],
        name='Max Accuracy with 0 Broken Outputs'
    ))

# add title and labels
fig.update_layout(
    title=f'Accuracy vs Broken Outputs for {model_name} on Keyword {task}',
    xaxis_title='Accuracy',
    yaxis_title='Broken Outputs'
)

fig.show()

# %%
# Find the index of the point with broken_output value 0 and the largest accuracy
max_accuracy_idx = None
max_accuracy = -float('inf')
for i, (accuracy, broken_output) in enumerate(zip(accuracies, perplexities)):
    if broken_output == 0 and accuracy > max_accuracy:
        max_accuracy = accuracy
        max_accuracy_idx = i

# make scatter plot of the accuracy values and broken outputs
fig = go.Figure()

# Add all points
fig.add_trace(go.Scatter(
    x=accuracies,
    y=perplexities,
    mode='markers',
    text=x_labels,
    name='Validation Runs'
))

# Highlight the point with broken_output value 0 and the largest accuracy
if max_accuracy_idx is not None:
    fig.add_trace(go.Scatter(
        x=[accuracies[max_accuracy_idx]],
        y=[perplexities[max_accuracy_idx]],
        mode='markers',
        marker=dict(color='red', size=9),
        text=[x_labels[max_accuracy_idx]],
        name='Run with Max Accuracy with 0 Low-Perplexity Outputs'
    ))

pretty_model_name = 'Phi-3' if model_name == 'phi-3' else 'Gemma'
main_title = 'Validation Acc. vs Low Perplexity:' if model_name == 'phi-3' else 'Validation Acc. vs Low Perpl.:'

# add title and labels
fig.update_layout(
    title=f'{main_title} {pretty_model_name} on {task}',
    xaxis_title='Accuracy',
    yaxis_title='Fract. of Low-perpl. Outputs'
)

# make title smaller
fig.update_layout(title_font_size=16)

# add label to the red point
if max_accuracy_idx is not None:
    fig.add_annotation(
        x=accuracies[max_accuracy_idx],
        y=perplexities[max_accuracy_idx],
        text=x_labels[max_accuracy_idx].replace('_', ', ').replace('W', 'c='),
        showarrow=True,
        arrowhead=1
    )

# resize plot
fig.update_layout(width=450, height=275)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# move legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-.65,
    xanchor='right',
    x=1
))

# showlegend=False
fig.show()

# save the plot as pdf
fig.write_image(f'./plots_for_paper/keywords/validation/{model_name}_{task}.pdf')

# %%
# make scatter plot of the low perplexity values and broken outputs
fig = go.Figure()

# Add all points
fig.add_trace(go.Scatter(
    x=perplexities,
    y=broken_outputs,
    mode='markers',
    text=x_labels,
    name='Perplexity vs Broken Outputs'
))

# add title and labels
fig.update_layout(
    title=f'Perplexity vs Broken Outputs for {model_name} on Keyword {task}',
    xaxis_title='Perplexity',
    yaxis_title='Broken Outputs'
)

fig.show()
# %%
# print some low perplexity outputs
df = result_dict[(24, 100)]
filtered_df = df[df.perplexity < 5]
for i, r in filtered_df.iterrows():
    print(r['prompt'])
    print(f'Perplexity: {r["perplexity"]}')
    print(r['response'])
    print('---')
# %%
# =============================================================================
# Plot for paper: subtracting inclusion vector vs adding exclusion vector
# =============================================================================

model_name = 'phi-3'
# model_name = 'gemma-2-2b-it'
folder = f'./keywords/out/{model_name}/forbidden_validation_w_forbidden_rep/'
file_name = 'out_gen_data_perplexity.jsonl'
subfolders = os.listdir(folder)
result_dict_exclusion = {}
paths_dict = {}
for subfolder in subfolders:
    print(subfolder)
    if 'adjust' in subfolder:
        continue
    if subfolder == 'no_instr' :
        layer = -1
        weight = -1
    else:
        layer = subfolder.split('_')[2]
        weight = subfolder.split('_')[5]
    print(os.listdir(folder + subfolder))
    if file_name not in os.listdir(folder + subfolder):
        print(f'{subfolder} does not have the file {file_name}')
        continue
    with open(folder + subfolder + f'/{file_name}' ) as f:
        results = [json.loads(line) for line in f]
    results_df = pd.DataFrame(results)
    result_dict_exclusion[(int(layer), int(weight))] = results_df
    paths_dict[(int(layer), int(weight))] = folder + subfolder + f'/{file_name}'

# %%
accuracy_dict_exclusion = {}
broken_outputs_dict_exclusion = {}
lengths_dict_exclusion = {}
low_perplexity_dict_exclusion = {}

for key, value in list(result_dict_exclusion.items()):
    value['low_perplexity'] = value['perplexity'] < 2.5
    low_perplexity_dict_exclusion[key] = value['low_perplexity'].mean()
    accuracy_dict_exclusion[key] = value['follow_all_instructions'].mean()
    broken_outputs_dict_exclusion[key] = value['broken_output'].mean()
    lengths_dict_exclusion[key] = value['length'].mean()


# %%
# make scatter plot of the accuracy values and low perplexity for low_perplexity_dict_exclusion and low_perplexity_dict


fig = go.Figure()

# Add all points
fig.add_trace(go.Scatter(
    x=accuracies,
    y=perplexities,
    mode='markers',
    text=x_labels,
    name='Subtraction of Inclusion Vector',
    marker=dict(color=plotly.colors.qualitative.Plotly[2])
))

# Add all points for exclusion
accuracies_exclusion = []
broken_outputs_exclusion = []
perplexities_exclusion = []
x_labels_exclusion = []
for (layer, weight), accuracy in accuracy_dict_exclusion.items():
    accuracies_exclusion.append(accuracy)
    broken_outputs_exclusion.append(broken_outputs_dict_exclusion[(layer, weight)])
    perplexities_exclusion.append(low_perplexity_dict_exclusion[(layer, weight)])
    x_labels_exclusion.append(f'L{layer}_W{weight}')

fig.add_trace(go.Scatter(
    x=accuracies_exclusion,
    y=perplexities_exclusion,
    mode='markers',
    text=x_labels_exclusion,
    name='Addition of Exclusion Vector'
))

# add title and labels
fig.update_layout(
    title=f'Validation Acc. vs Low Perplexity: Phi-3 on Exclusion',
    xaxis_title='Accuracy',
    yaxis_title='Fract. of Low-perpl. Outputs'
)

# make title smaller
fig.update_layout(title_font_size=16)

# resize plot
fig.update_layout(width=450, height=275)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# move legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-.65,
    xanchor='right',
    x=.75
))

# showlegend=False

# store the plot as pdf
fig.write_image(f'./plots_for_paper/keywords/validation/add_exclusion_vs_subtract_inclusion.pdf')

fig.show()


# %%

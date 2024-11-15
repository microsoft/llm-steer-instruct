# %%
import os

if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
elif 'home' in os.getcwd():
    os.chdir('/home/t-astolfo/t-astolfo')
    print('We\'re on a sandbox machine')
import json
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import if_inference, generate_with_hooks, direction_ablation_hook, direction_projection_hook
from transformer_lens import utils as tlutils
from collections import Counter

# %%
model_name = 'phi-3'
# model_name = 'gemma-2-2b-it'
folder = f'./keywords/out/{model_name}/forbidden_validation/'
folder = f'./keywords/out/{model_name}/forbidden_validation_w_forbidden_rep/'
file_name = 'out_gen_data.jsonl'
subfolders = os.listdir(folder)
result_dict = {}
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
    result_dict[(int(layer), int(weight))] = results_df
# %%
# load model tokenizer
# %%
from transformers import AutoTokenizer
with open('./hf_token.txt') as f:
    hf_token = f.read()
if model_name == 'phi-3':
    model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'
elif model_name == 'gemma-2-2b-it':
    model_name_hf = 'google/gemma-2-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name_hf, token=hf_token)
# %%
accuracy_dict = {}
broken_outputs_dict = {}
lengths_dict = {}
for key, value in tqdm(list(result_dict.items())):
    print(key)
    accuracy_dict[key] = value['follow_all_instructions'].mean()
    broken_outputs = []
    lengths = []
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
    value['broken_output'] = broken_outputs
    value['length'] = lengths
    lengths_dict[key] = sum(lengths) / len(lengths)
    broken_outputs_dict[key] = sum(broken_outputs) / len(broken_outputs)

# %%
accuracy_dict
# %%
broken_outputs_dict
# %%
lengths_dict
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
            print('---')

# %%
# make line plot of the accuracy values and broken outputs
accuracies = []
broken_outputs = []
x_labels = []
for (layer, weight), accuracy in accuracy_dict.items():
    accuracies.append(accuracy)
    broken_outputs.append(broken_outputs_dict[(layer, weight)])
    x_labels.append(f'L{layer}_W{weight}')

# sort the lists according to the x_labels
x_labels, accuracies, broken_outputs = zip(*sorted(zip(x_labels, accuracies, broken_outputs)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_labels, y=accuracies, mode='lines+markers', name='Accuracy'))
fig.add_trace(go.Scatter(x=x_labels, y=broken_outputs, mode='lines+markers', name='Broken Outputs'))
fig.update_layout(xaxis_tickangle=-45)
# add horizontal line at broken_outputs[(-1,-1)]
fig.add_shape(type="line", x0=0, x1=len(x_labels), y0=broken_outputs[0], y1=broken_outputs[0], line=dict(color="red", width=1))

fig.show()

# %%
# make scatter plot of the accuracy values and broken outputs
fig = go.Figure()
fig.add_trace(go.Scatter(x=accuracies, y=broken_outputs, mode='markers', text=x_labels, name='Accuracy vs Broken Outputs'))
fig.update_layout(xaxis_title='Accuracy', yaxis_title='Broken Outputs')
fig.show()

# %%

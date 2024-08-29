# %%
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly

os.chdir('/home/t-astolfo/t-astolfo')
# %%
dir = 'keywords/layer_search_out/'
model_name = 'phi-3'
n_examples = 20
seed = 42
instr = 'no_instr'

file = f'{dir}/{model_name}/n_examples{n_examples}_seed{seed}/out_{instr}.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

results_df_no_instr = pd.DataFrame(results)

instr = 'instr'
file = f'{dir}/{model_name}/n_examples{n_examples}_seed{seed}/out_{instr}.jsonl'
with open(file, 'r') as f:
    results = [json.loads(line) for line in f]

results_df_instr = pd.DataFrame(results)
# %%
# %%
weights = [50,75,100,125,150]
layer_range = range(32 // 5, 32, 2)
layer_range = [-1] + list(layer_range)

uids = results_df_no_instr.question.unique()
results_df_no_instr['layer'] = [0 for _ in range(len(results_df_no_instr))]
layers = [l for l in layer_range[1:] for _ in range(len(weights))]
layers = [-1] + layers
# %%
results_df_no_instr['layer'] = [0 for _ in range(len(results_df_no_instr))]
results_df_no_instr['weight'] = [-1 for _ in range(len(results_df_no_instr))]
for uid in uids:
    # assign layer to each uid
    results_df_no_instr.loc[results_df_no_instr.question == uid, 'layer'] = layers
    for layer in layer_range:
        if layer == -1:
            continue
        results_df_no_instr.loc[(results_df_no_instr.question == uid) & (results_df_no_instr.layer == layer), 'weight'] = weights

# %%
## group by layer and weight and compute mean of "accuracy"
accuracy_values = { (l, w) : results_df_no_instr[(results_df_no_instr.layer == l) & (results_df_no_instr.weight == w)].accuracy.mean() for l in layer_range for w in weights } 
accuracy_values[(-1, -1)] = results_df_no_instr[results_df_no_instr.layer == -1].accuracy.mean()
# %%
# make a heatmap of the accuracy values
accuracy_values_df = pd.DataFrame(accuracy_values, index=['accuracy']).T.reset_index()
accuracy_values_df.columns = ['layer', 'weight', 'accuracy']
accuracy_values_df['layer'] = accuracy_values_df['layer']
accuracy_values_df['weight'] = accuracy_values_df['weight']
accuracy_values_df['accuracy'] = accuracy_values_df['accuracy']

# Create heatmap
fig = px.density_heatmap(accuracy_values_df, x='layer', y='weight', z='accuracy', histfunc='avg', nbinsx=len(layer_range), nbinsy=len(weights), labels={'x': 'Layer', 'y': 'Weight', 'z': 'Accuracy'})
fig.show()
# %%
# make a bar plot of the accuracy values
fig = go.Figure()
for layer in layer_range:
    for weight in weights:
        if layer == -1:
            continue
        layer_df = results_df_no_instr[(results_df_no_instr.layer == layer) & (results_df_no_instr.weight == weight)]
        fig.add_trace(go.Bar(x=[f'{layer}_{weight}'], y=[layer_df.accuracy.mean()], name=f'{layer}_{weight}'))

fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

# %%
occurrences = { (l, w) : results_df_no_instr[(results_df_no_instr.layer == l) & (results_df_no_instr.weight == w)].occurrences.mean() for l in layer_range for w in weights } 
occurrences[(-1, -1)] = results_df_no_instr[results_df_no_instr.layer == -1].occurrences.mean()

# make a heatmap of the occurrences values
accuracy_values_df = pd.DataFrame(occurrences, index=['accuracy']).T.reset_index()
accuracy_values_df.columns = ['layer', 'weight', 'accuracy']
accuracy_values_df['layer'] = accuracy_values_df['layer']
accuracy_values_df['weight'] = accuracy_values_df['weight']
accuracy_values_df['accuracy'] = accuracy_values_df['accuracy']

# Create heatmap
fig = px.density_heatmap(accuracy_values_df, x='layer', y='weight', z='accuracy', histfunc='avg', nbinsx=len(layer_range), nbinsy=len(weights), labels={'x': 'Layer', 'y': 'Weight', 'z': 'Accuracy'})
fig.show()
# %%
fig = go.Figure()
for layer in layer_range:
    for weight in weights:
        if layer == -1:
            continue
        layer_df = results_df_no_instr[(results_df_no_instr.layer == layer) & (results_df_no_instr.weight == weight)]
        fig.add_trace(go.Bar(x=[f'{layer}_{weight}'], y=[layer_df.occurrences.mean()], name=f'{layer}_{weight}'))

fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

# %%
# =============================================================================
# show some outputs
# =============================================================================

# get some outputs
question = uids[0]

# get outputs for the question
outputs = results_df_no_instr[results_df_no_instr.question == question]

for i, r in outputs.iterrows():
    print(f'Prompt: {r.model_input}')
    print(f'LAYER: {r.layer} | WEIGHT: {r.weight}')
    print(f'Response: {r.response}')
    print(f'Accuracy: {r.accuracy}')
    print(f'Occurrences: {r.occurrences}')
    print('--------------------------------')
# %%

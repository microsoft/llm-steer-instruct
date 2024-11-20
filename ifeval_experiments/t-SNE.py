# %%
# =============================================================================
# make t-SNE plot of instruction vectors
# =============================================================================

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

from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import os
import plotly.express as px
import sys
import plotly.graph_objects as go
from tqdm import tqdm

# %%
model_name = 'phi-3'
rep_dir = f'./ifeval_experiments/representations/{model_name}/single_instr_all_base_x_all_instr/'
device = 'cpu'
files = os.listdir(rep_dir)

# from the files, remove the ones that have "pre_computed" in the name
files = [f for f in files if 'pre_computed' not in f]
df = pd.read_hdf(rep_dir + files[0])

# max_length = min([x.shape[1] for x in results_df['last_token_rs_no_instr'].values])
# %%
# load pred_computed_ivs
file = f'./ifeval_experiments/representations/{model_name}/single_instr_all_base_x_all_instr/pre_computed_ivs_best_layer_validation_no_instr.h5'
df_ivs = pd.read_hdf(file)
best_layers_dict = {r.instruction.replace(':', '_') : r.max_diff_layer_idx for r in df_ivs.itertuples()}


# %%
instr_dirs = {}
repr_w_instr = {}
repr_wo_instr = {}

for f in tqdm(files):
    if 'length' in f:
        continue
    if 'keywords' in f:
        continue
    if 'detectable_content_' in f:
        continue
    if 'language_response_' in f:
        continue
    if 'combination' in f:
        continue
    print(f)
    df = pd.read_hdf(rep_dir + f)
    hs_instr = df['last_token_rs'].values
    hs_instr = np.array([example_array for example_array in list(hs_instr)])

    hs_no_instr = df['last_token_rs_no_instr'].values
    hs_no_instr = np.array([example_array for example_array in list(hs_no_instr)])

    repr_diffs = hs_instr - hs_no_instr
    mean_repr_diffs = repr_diffs.mean(axis=0)
    last_token_mean_diff = mean_repr_diffs[:, -1, :]

    instr_dir = last_token_mean_diff / np.linalg.norm(last_token_mean_diff)

    if f in best_layers_dict:
        layer_idx = best_layers_dict[f.replace('.h5', '')]
    else:
        layer_idx = -1

    f = f.replace('language_response_', '')
    f = f.replace('.h5', '')
    f = f.replace('detectable_format_', '')
    f = f.replace('detectable_content_', '')
    f = f.replace('change_case_', '')
    f = f.replace('punctuation_', '')
    f = f.replace('startend_', '')

    best_layers_dict[f] = layer_idx

    instr_dirs[f] = instr_dir
    repr_w_instr[f] = hs_instr[:300, :, -1, :]
    repr_wo_instr[f] = hs_no_instr[:300, :, -1, :]

# %%

layer_idx = 20

# rename instructions
new_names = { k: k.replace('_', ' ').title() for k in repr_w_instr.keys()}
new_names['number_highlighted_sections'] = 'Highlight Text'
new_names['english_lowercase'] = 'Lowercase'
new_names['json_format'] = 'JSON Format'
new_names['capital_word_frequency'] = 'Capital Word Freq.'
new_names['Response Language'] = 'Language'
new_names['english_capital'] = 'Capitalize'
new_names['constrained_response'] = 'Constrained Resp.'

per_example_vectors = {}
for k, v in repr_w_instr.items():
    key = new_names[k]
    print(f' Old name: {k}, New name: {key}')
    per_example_vectors[key] = v[:, layer_idx, :] - repr_wo_instr[k][:, layer_idx, :]

# %%

# Flatten the vectors and create labels
all_vectors = []
labels = []
for key, vectors in per_example_vectors.items():
    all_vectors.append(vectors)
    labels.extend([key] * vectors.shape[0])

all_vectors = np.concatenate(all_vectors, axis=0)

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=1000, learning_rate=200, n_iter=1000, init='pca', random_state=42, verbose=1)
tsne_results = tsne.fit_transform(all_vectors)

# Create a DataFrame for Plotly
df_tsne = pd.DataFrame(tsne_results, columns=['t-SNE component 1', 't-SNE component 2'])
df_tsne['Instruction'] = labels

colors = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
# %%

# Plot the results using Plotly
fig = px.scatter(df_tsne, x='t-SNE component 1', y='t-SNE component 2', color='Instruction', title='(a) t-SNE of Format Instruction Vectors', color_discrete_sequence=colors)
fig.update_traces(marker=dict(opacity=0.7, size=5))

# change title font size
fig.update_layout(title_font_size=17)

# resize plot
fig.update_layout(width=500, height=325)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

fig.show()

# save plot as pdf
fig.write_image(f'./plots_for_paper/format/t-sne.pdf')


# %%

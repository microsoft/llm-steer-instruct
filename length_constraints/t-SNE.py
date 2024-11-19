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

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import os
import plotly.express as px
import sys
import plotly.graph_objects as go
from tqdm import tqdm

# %%
model_name = 'phi-3'
rep_dir = f'./length_constraints/representations/{model_name}/'
device = 'cpu'

# from the files, remove the ones that have "pre_computed" in the name
file = '6sentences_300examples_hs.h5'
df = pd.read_hdf(rep_dir + file)

file = 'high_level_200examples_hs.h5'
high_level_df = pd.read_hdf(rep_dir + file)

repr_w_instr = {}
repr_wo_instr = {}

for length_constraint in df.length_constraint.unique():
    length_specific_df = df[df.length_constraint == length_constraint]

    hs_instr = length_specific_df['last_token_rs'].values
    hs_instr = np.array([example_array for example_array in list(hs_instr)])

    hs_no_instr = length_specific_df['last_token_rs_no_instr'].values
    hs_no_instr = np.array([example_array for example_array in list(hs_no_instr)])

    repr_diffs = hs_instr - hs_no_instr
    mean_repr_diffs = repr_diffs.mean(axis=0)
    if len(last_token_mean_diff.shape) == 3:
        last_token_mean_diff = mean_repr_diffs[:, -1, :]
    else:
        last_token_mean_diff = mean_repr_diffs

    instr_dir = last_token_mean_diff / np.linalg.norm(last_token_mean_diff)

    repr_w_instr[str(length_constraint)] = hs_instr[:300, :, :]
    repr_wo_instr[str(length_constraint)] = hs_no_instr[:300, :, :]

# %%
conciseness_df = high_level_df[high_level_df.length_constraint == 0]

hs_instr = conciseness_df['last_token_rs'].values
hs_instr = np.array([example_array for example_array in list(hs_instr)])

hs_no_instr = conciseness_df['last_token_rs_no_instr'].values
hs_no_instr = np.array([example_array for example_array in list(hs_no_instr)])

repr_diffs = hs_instr - hs_no_instr
mean_repr_diffs = repr_diffs.mean(axis=0)
if len(last_token_mean_diff.shape) == 3:
    last_token_mean_diff = mean_repr_diffs[:, -1, :]
else:
    last_token_mean_diff = mean_repr_diffs

instr_dir = last_token_mean_diff / np.linalg.norm(last_token_mean_diff)
repr_w_instr['conciseness'] = hs_instr[:300, :, :]
repr_wo_instr['conciseness'] = hs_no_instr[:300, :, :]

verbosity_df = high_level_df[high_level_df.length_constraint == 4]

hs_instr = verbosity_df['last_token_rs'].values
hs_instr = np.array([example_array for example_array in list(hs_instr)])

hs_no_instr = verbosity_df['last_token_rs_no_instr'].values
hs_no_instr = np.array([example_array for example_array in list(hs_no_instr)])

repr_diffs = hs_instr - hs_no_instr
mean_repr_diffs = repr_diffs.mean(axis=0)
if len(last_token_mean_diff.shape) == 3:
    last_token_mean_diff = mean_repr_diffs[:, -1, :]
else:
    last_token_mean_diff = mean_repr_diffs

instr_dir = last_token_mean_diff / np.linalg.norm(last_token_mean_diff)
repr_w_instr['verbosity'] = hs_instr[:300, :, :]
repr_wo_instr['verbosity'] = hs_no_instr[:300, :, :]



# %%

layer_idx = 12



per_example_vectors = {}
for k, v in repr_w_instr.items():
    key = f'n={int(k)+1}' if k not in ['conciseness', 'verbosity'] else k
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
tsne = TSNE(n_components=2, perplexity=1600, learning_rate=200, n_iter=1000, init='pca', random_state=42, verbose=1)
tsne_results = tsne.fit_transform(all_vectors)

# Create a DataFrame for Plotly
df_tsne = pd.DataFrame(tsne_results, columns=['t-SNE component 1', 't-SNE component 2'])
df_tsne['Instruction'] = labels

# %%
# Map the classes to numeric values
df_tsne['Instruction_numeric'] = df_tsne['Instruction'].astype('category').cat.codes

# Define a continuous color scale
color_scale = px.colors.sequential.YlOrRd

# Plot the results using Plotly with a continuous color scale
fig = px.scatter(df_tsne, x='t-SNE component 1', y='t-SNE component 2', color='Instruction_numeric',
                 title='t-SNE of Length Instruction Vectors', color_continuous_scale=color_scale, labels={'Instruction_numeric': 'Instruction'})

fig.update_traces(marker=dict(opacity=.8, size=5))

# change title font size
fig.update_layout(title_font_size=17)

# resize plot
fig.update_layout(width=500, height=325)

# remove padding
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

# Update color bar to show the original class labels
fig.update_coloraxes(colorbar=dict(
    tickvals=[0, 1, 2, 3, 4, 5, 6, 6.9],
    ticktext=df_tsne['Instruction'].astype('category').cat.categories
))

# Add an arrow from (1, -0.5) to (-2.5, -0.5)
fig.add_annotation(
    dict(
        ax=1.5,
        ay=-3.5,
        x=-2.7,
        y=-3.5,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black",
        text="Longer<br>Outputs", 
        font=dict(size=12, color="black"),
        align="center",
        textangle=0,
        height=50,
        borderpad=0,
        valign="middle",
    )
)

fig.show()

# save plot as pdf
fig.write_image(f'./plots_for_paper/length/t-sne.pdf')


# %%

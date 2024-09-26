# %%
import os 
import sys
if 'Users' in os.getcwd():
    os.chdir('/Users/alestolfo/workspace/llm-steer-instruct/composition')
    sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/')
    print('We\'re on the local machine')
elif 'cluster' in os.getcwd():
    os.chdir('/cluster/project/sachan/alessandro/llm-steer-instruct/composition')
    sys.path.append('/cluster/project/sachan/alessandro/llm-steer-instruct')
    print('We\'re on a sandbox machine')


import json
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly
# %%
model_name = 'phi-3'

folder = f'./casing_and_exclude_out/{model_name}/forbidden'
steering = 'instr_plus_adjust_rs_24_n_examples20'

path = f'{folder}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_steering_plus_instr = pd.DataFrame(data)

steering = 'standard'

path = f'{folder}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_standard = pd.DataFrame(data)

steering = 'no_instr'
path = f'{folder}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_no_instr = pd.DataFrame(data)

steering = 'adjust_rs_24_n_examples20'
path = f'{folder}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_steering = pd.DataFrame(data)

# %%
print(df_no_instr.follow_all_instructions.sum())
print(df_steering.follow_all_instructions.sum())
print(df_standard.follow_all_instructions.sum())
print(df_steering_plus_instr.follow_all_instructions.sum())

print(df_no_instr.follow_all_instructions.mean())
print(df_steering.follow_all_instructions.mean())
print(df_standard.follow_all_instructions.mean())
print(df_steering_plus_instr.follow_all_instructions.mean())
# %%
def add_per_instr_accuracy(df):
    case_accuracy = []
    word_accuracy = []
    for i, row in df.iterrows():
        case_acc, word_acc = row['follow_instruction_list_loose']
        case_accuracy.append(case_acc)
        word_accuracy.append(word_acc)

    df['case_accuracy'] = case_accuracy
    df['word_accuracy'] = word_accuracy

add_per_instr_accuracy(df_no_instr)
add_per_instr_accuracy(df_steering)
add_per_instr_accuracy(df_standard)
add_per_instr_accuracy(df_steering_plus_instr)
# %%
print(df_no_instr.case_accuracy.mean())
print(df_steering.case_accuracy.mean())
print(df_standard.case_accuracy.mean())
print(df_steering_plus_instr.case_accuracy.mean())

print(df_no_instr.word_accuracy.mean())
print(df_steering.word_accuracy.mean())
print(df_standard.word_accuracy.mean())
print(df_steering_plus_instr.word_accuracy.mean())
# %%
# print max length of responses
print(df_no_instr.response.apply(len).max())
print(df_steering.response.apply(len).max())
print(df_standard.response.apply(len).max())
print(df_steering_plus_instr.response.apply(len).max())


# %%
# =============================================================================
# make plots
# =============================================================================

# make scatter plot of follow_all_instructions vs length_accuracy
coordinates_no_steering_no_instr = (df_no_instr['word_accuracy'].mean(), df_no_instr['case_accuracy'].mean())
coordinates_steering_no_instr = (df_steering['word_accuracy'].mean(), df_steering['case_accuracy'].mean())
coordinates_no_steering_plus_instr = df_standard[['word_accuracy', 'case_accuracy']].mean()
coordinates_steering_plus_instr = df_steering_plus_instr[['word_accuracy', 'case_accuracy']].mean()

color1 = plotly.colors.qualitative.Plotly[0]
color2 = plotly.colors.qualitative.Plotly[3]


fig = go.Figure()
fig.add_trace(go.Scatter(x=[coordinates_no_steering_no_instr[0]], y=[coordinates_no_steering_no_instr[1]], mode='markers', name='<b>w/o</b> Instr.', marker=dict(size=10), marker_color=color1, opacity=0.6))
fig.add_trace(go.Scatter(x=[coordinates_steering_no_instr[0]], y=[coordinates_steering_no_instr[1]], mode='markers', name='+ Steering*', marker=dict(size=10), marker_color=color1))
fig.add_trace(go.Scatter(x=[coordinates_no_steering_plus_instr[0]], y=[coordinates_no_steering_plus_instr[1]], mode='markers', name='<b>w/</b> Instr.', marker=dict(size=10), marker_color=color2, opacity=0.6))
fig.add_trace(go.Scatter(x=[coordinates_steering_plus_instr[0]], y=[coordinates_steering_plus_instr[1]], mode='markers', name='+ Steering*', marker=dict(size=10), marker_color=color2))

# add labels
fig.update_layout(
    xaxis_title='Word Exclusion Accuracy',
    yaxis_title='Casing Accuracy'
)

# Add arrow annotation between the first and the second point
fig.add_annotation(
    dict(
        x=coordinates_steering_no_instr[0]-0.01,
        y=coordinates_steering_no_instr[1]-0.014,
        ax=coordinates_no_steering_no_instr[0],
        ay=coordinates_no_steering_no_instr[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black"
    )
)

# Add arrow annotation between the first and the second point
fig.add_annotation(
    dict(
        x=coordinates_steering_plus_instr[0]-0.01,
        y=coordinates_steering_plus_instr[1]-0.005,
        ax=coordinates_no_steering_plus_instr[0],
        ay=coordinates_no_steering_plus_instr[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black"
    )
)

# set y limits
fig.update_layout(
    yaxis=dict(
        range=[-0.035, 1]
    ),
    xaxis=dict(
        range=[0.55, 1.015]
    )
)

# add title
fig.update_layout(
    title='(b) Multi-instr.: Casing & Word Ex.',
    title_font_size=16
)

# resize the figure
fig.update_layout(
    width=300,
    height=250
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=00)
)

# move legend to the bottom
fig.update_layout(
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-.85,
        xanchor='right',
        x=0.85
    )
)


# store the figure as pdf
fig.write_image(f'../plots_for_paper/composition/{model_name}_case_word_exclusion.pdf')

fig.show()
# %%

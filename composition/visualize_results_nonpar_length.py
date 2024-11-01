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

folder = f'./nonpar_plus_length_out/{model_name}/all_base_x_all_instr'
setting = 'instr_w_length_instr'
steering = 'adjust_rs_-1_conciseness_L12_w20'

path = f'{folder}/{setting}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_steering_plus_instr = pd.DataFrame(data)

setting = 'instr_w_length_instr'
steering = 'no_steering_none_L12_w20'

path = f'{folder}/{setting}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_no_steering_plus_instr = pd.DataFrame(data)

setting = 'no_instr'
steering = 'adjust_rs_20_conciseness_L12_w20'

path = f'{folder}/{setting}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_steering_no_instr = pd.DataFrame(data)

setting = 'no_instr'
steering = 'no_steeringno_length_steering'

path = f'{folder}/{setting}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_no_steering_no_instr = pd.DataFrame(data)

# %%
for df in [df_steering_plus_instr, df_no_steering_plus_instr, df_steering_no_instr, df_no_steering_no_instr]:

    length_constraints = df.length_constraint.apply(lambda x: x+1)
    lengths = df.response_length_sent
    length_accuracy = (length_constraints >= lengths)
    df['length_accuracy'] = length_accuracy



# %%
print(df_no_steering_no_instr.follow_all_instructions.mean())
print(df_steering_no_instr.follow_all_instructions.mean())
print(df_no_steering_plus_instr.follow_all_instructions.mean())
print(df_steering_plus_instr.follow_all_instructions.mean())
# %%
print(df_no_steering_no_instr.length_accuracy.mean())
print(df_steering_no_instr.length_accuracy.mean())
print(df_no_steering_plus_instr.length_accuracy.mean())
print(df_steering_plus_instr.length_accuracy.mean())

# %%
# carry out mcNemar test
from statsmodels.stats.contingency_tables import mcnemar

for df1, df2 in [(df_no_steering_no_instr, df_steering_no_instr), (df_no_steering_plus_instr, df_steering_plus_instr)]: 
    correct1 = df1.length_accuracy
    correct2 = df2.length_accuracy

    table = [[0, 0], [0, 0]]
    for i in range(len(correct1)):
        if correct1.iloc[i] and correct2.iloc[i]:
            table[0][0] += 1
        elif correct1.iloc[i] and not correct2.iloc[i]:
            table[0][1] += 1
        elif not correct1.iloc[i] and correct2.iloc[i]:
            table[1][0] += 1
        elif not correct1.iloc[i] and not correct2.iloc[i]:
            table[1][1] += 1

    result = mcnemar(table, exact=False, correction=True)
    print(f'McNemar test for LENGTH: {result.pvalue}')

    correct1 = df1.follow_all_instructions
    correct2 = df2.follow_all_instructions

    table = [[0, 0], [0, 0]]
    for i in range(len(correct1)):
        if correct1.iloc[i] and correct2.iloc[i]:
            table[0][0] += 1
        elif correct1.iloc[i] and not correct2.iloc[i]:
            table[0][1] += 1
        elif not correct1.iloc[i] and correct2.iloc[i]:
            table[1][0] += 1
        elif not correct1.iloc[i] and not correct2.iloc[i]:
            table[1][1] += 1

    result = mcnemar(table, exact=False, correction=True)
    print(f'McNemar test for FOLLOWING ALL INSTRUCTIONS: {result.pvalue}')




# %%
def add_follow_everything(df):
    follow_everything = []
    for i in range(len(df)):
        follow_everything.append(all([df.iloc[i].follow_all_instructions, df.iloc[i].length_accuracy]))
    df['follow_everything'] = follow_everything

    # remove json and multi_section instructions
    # df = df[df.instruction_id_list.apply(lambda x: (x[0] != 'detectable_format:json_format') and (x[0] != 'detectable_format:multiple_sections'))]
    # df = df[df.instruction_id_list.apply(lambda x: ('highl' in x[0]))]

    return df

df_no_steering_no_instr = add_follow_everything(df_no_steering_no_instr)
df_steering_no_instr = add_follow_everything(df_steering_no_instr)
df_no_steering_plus_instr = add_follow_everything(df_no_steering_plus_instr)
df_steering_plus_instr = add_follow_everything(df_steering_plus_instr)

print(df_no_steering_no_instr.follow_everything.mean())
print(df_steering_no_instr.follow_everything.mean())
print(df_no_steering_plus_instr.follow_everything.mean())
print(df_steering_plus_instr.follow_everything.mean())

# %%
all_instructions = df_no_steering_plus_instr.instruction_id_list.apply(lambda x: x[0]).unique()

no_steering_no_instr_categories = { instr: 0 for instr in all_instructions }
steering_no_instr_categories = { instr: 0 for instr in all_instructions }
no_steering_plus_instr_categories = { instr: 0 for instr in all_instructions }
steering_plus_instr_categories = { instr: 0 for instr in all_instructions }

no_steering_no_instr_categories_length = { instr: 0 for instr in all_instructions }
steering_no_instr_categories_length = { instr: 0 for instr in all_instructions }
no_steering_plus_instr_categories_length = { instr: 0 for instr in all_instructions }
steering_plus_instr_categories_length = { instr: 0 for instr in all_instructions }

# for each instruction, compute the accuracy of following the instruction
for instr in all_instructions:
    instr_df_no_steering_no_instr = df_no_steering_no_instr[df_no_steering_no_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
    instr_df_steering_no_instr = df_steering_no_instr[df_steering_no_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
    instr_df_no_steering_plus_instr = df_no_steering_plus_instr[df_no_steering_plus_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
    instr_df_steering_plus_instr = df_steering_plus_instr[df_steering_plus_instr.instruction_id_list.apply(lambda x: x[0]) == instr]

    no_steering_no_instr_categories[instr] = instr_df_no_steering_no_instr.follow_all_instructions.mean()
    steering_no_instr_categories[instr] = instr_df_steering_no_instr.follow_all_instructions.mean()
    no_steering_plus_instr_categories[instr] = instr_df_no_steering_plus_instr.follow_all_instructions.mean()
    steering_plus_instr_categories[instr] = instr_df_steering_plus_instr.follow_all_instructions.mean()

    no_steering_no_instr_categories_length[instr] = instr_df_no_steering_no_instr.length_accuracy.mean()
    steering_no_instr_categories_length[instr] = instr_df_steering_no_instr.length_accuracy.mean()
    no_steering_plus_instr_categories_length[instr] = instr_df_no_steering_plus_instr.length_accuracy.mean()
    steering_plus_instr_categories_length[instr] = instr_df_steering_plus_instr.length_accuracy.mean()

# %%
# make histogram of instruction-following accuracy per instruction
fig = go.Figure()

type = 'follow_all_instructions'
if type == 'follow_all_instructions':
    fig.add_trace(go.Bar(x=list(no_steering_no_instr_categories.keys()), y=list(no_steering_no_instr_categories.values()), name='No steering no instr', marker_color=plotly.colors.qualitative.Plotly[0]))
    fig.add_trace(go.Bar(x=list(steering_no_instr_categories.keys()), y=list(steering_no_instr_categories.values()), name='Steering no instr', marker_color=plotly.colors.qualitative.Plotly[1]))
    fig.add_trace(go.Bar(x=list(no_steering_plus_instr_categories.keys()), y=list(no_steering_plus_instr_categories.values()), name='No steering plus instr', marker_color=plotly.colors.qualitative.Plotly[2]))
    fig.add_trace(go.Bar(x=list(steering_plus_instr_categories.keys()), y=list(steering_plus_instr_categories.values()), name='Steering plus instr', marker_color=plotly.colors.qualitative.Plotly[3]))
elif type == 'length':
    fig.add_trace(go.Bar(x=list(no_steering_no_instr_categories.keys()), y=list(no_steering_no_instr_categories_length.values()), name='No steering no instr', marker_color=plotly.colors.qualitative.Plotly[0]))
    fig.add_trace(go.Bar(x=list(steering_no_instr_categories.keys()), y=list(steering_no_instr_categories_length.values()), name='Steering no instr', marker_color=plotly.colors.qualitative.Plotly[1]))
    fig.add_trace(go.Bar(x=list(no_steering_plus_instr_categories.keys()), y=list(no_steering_plus_instr_categories_length.values()), name='No steering plus instr', marker_color=plotly.colors.qualitative.Plotly[2]))
    fig.add_trace(go.Bar(x=list(steering_plus_instr_categories.keys()), y=list(steering_plus_instr_categories_length.values()), name='Steering plus instr', marker_color=plotly.colors.qualitative.Plotly[3]))

# add labels
fig.update_layout(
    xaxis_title='Instruction',
    yaxis_title='Instruction-following Accuracy'
)

# add title
fig.update_layout(
    title='Instruction-following Accuracy per Instruction'
)

# resize the figure
fig.update_layout(
    width=900,
    height=500
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=20)
)

# store the figure as pdf
# fig.write_image(f'../plots/instr_following_accuracy_per_instr.pdf')

fig.show()

# %%
# print some outputs
# instr = [i for i in all_instructions if 'multiple_section' in i][0]

df1 = df_no_steering_no_instr
df2 = df_steering_no_instr

instr_df1 = df1[df1.instruction_id_list.apply(lambda x: x[0]) == instr]
instr_df2 = df2[df2.instruction_id_list.apply(lambda x: x[0]) == instr]

for i, r in instr_df1.iterrows():
    print(f'Prompt: {r.model_input}')
    print(f'Lenght constraint: {r.length_constraint+1}')
    print(f'Instruction: {r.instruction_id_list}')  
    print(f'Kwargs: {r.kwargs}')
    print(f'Response: {r.response}')
    print(f'Followed everything: {r.follow_everything}')
    print(f'Followed all instructions: {r.follow_all_instructions}')
    print('Length accuracy: ', r.length_accuracy)
    print('-----')
    print(f'Response w/ STEERING: {df2[df2.key == r.key].response.values[0]}')
    print('Follow everything w/ STEERING: ', df2[df2.key == r.key].follow_everything.values[0])
    print(f'Followed all instructions w/ STEERING: {df2[df2.key == r.key].follow_all_instructions.values[0]}')
    print('Length accuracy w/ STEERING: ', df2[df2.key == r.key].length_accuracy.values[0])
    print('=====================')


# %%
# print some outputs
keys = df_no_steering_no_instr.key
key = keys[7]

print(f'Prompt: {df_no_steering_no_instr[df_no_steering_no_instr.key == key].prompt.values[0]}')
print(f'Lenght constraint: {df_no_steering_no_instr[df_no_steering_no_instr.key == key].length_constraint.values[0]+1}')
print(f'Response: {df_no_steering_plus_instr[df_no_steering_plus_instr.key == key].response.values[0]}')
print(f'Followed all instructions: {df_no_steering_plus_instr[df_no_steering_plus_instr.key == key].follow_all_instructions.values[0]}')
print('-----')
print(f'Response w/ STEERING: {df_steering_plus_instr[df_steering_plus_instr.key == key].response.values[0]}')
print(f'Followed all instructions w/ STEERING: {df_steering_plus_instr[df_steering_plus_instr.key == key].follow_all_instructions.values[0]}')
# %%


# %%
# =============================================================================
# make plots for all instructions
# =============================================================================

df1 = df_no_steering_no_instr
df2 = df_steering_no_instr
df3 = df_no_steering_plus_instr
df4 = df_steering_plus_instr

# make scatter plot of follow_all_instructions vs length_accuracy
coordinates_no_steering_no_instr = df1[['follow_all_instructions', 'length_accuracy']].mean()
coordinates_steering_no_instr = df2[['follow_all_instructions', 'length_accuracy']].mean()
coordinates_no_steering_plus_instr = df3[['follow_all_instructions', 'length_accuracy']].mean()
coordinates_steering_plus_instr = df4[['follow_all_instructions', 'length_accuracy']].mean()

color1 = plotly.colors.qualitative.Plotly[0]
color2 = plotly.colors.qualitative.Plotly[3]


fig = go.Figure()
fig.add_trace(go.Scatter(x=[coordinates_no_steering_no_instr[0]], y=[coordinates_no_steering_no_instr[1]], mode='markers', name='<b>w/o</b> Instr.', marker=dict(size=10), marker_color=color1, opacity=0.6))
fig.add_trace(go.Scatter(x=[coordinates_steering_no_instr[0]], y=[coordinates_steering_no_instr[1]], mode='markers', name='+ Steering', marker=dict(size=10), marker_color=color1))
fig.add_trace(go.Scatter(x=[coordinates_no_steering_plus_instr[0]], y=[coordinates_no_steering_plus_instr[1]], mode='markers', name='<b>w/</b> Instr.', marker=dict(size=10), marker_color=color2, opacity=0.6))
fig.add_trace(go.Scatter(x=[coordinates_steering_plus_instr[0]], y=[coordinates_steering_plus_instr[1]], mode='markers', name='+ Steering', marker=dict(size=10), marker_color=color2))

# add labels
fig.update_layout(
    xaxis_title='Fomat Accuracy',
    yaxis_title='Length Accuracy'
)

# Add arrow annotation between the first and the second point
fig.add_annotation(
    dict(
        x=coordinates_steering_no_instr[0]-0.01,
        y=coordinates_steering_no_instr[1]-0.014,
        ax=coordinates_no_steering_no_instr[0]-0.01,
        ay=coordinates_no_steering_no_instr[1]-0.02,
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
        x=coordinates_steering_plus_instr[0]-0,
        y=coordinates_steering_plus_instr[1]-0,
        ax=coordinates_no_steering_plus_instr[0]-0.01,
        ay=coordinates_no_steering_plus_instr[1]-0.02,
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
        range=[0.15, 0.8]
    ),
    xaxis=dict(
        range=[0.05, 0.85]
    )
)

# add title
fig.update_layout(
    title='(a) Multi-instr.: Format & Length',
    title_font_size=16
)

# resize the figure
fig.update_layout(
    width=300,
    height=250
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

# remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=0)
)

# store the figure as pdf
fig.write_image(f'../plots_for_paper/composition/{model_name}_all_instr.pdf')

fig.show()

# %%
# =============================================================================
# make plots for all instructions
# =============================================================================

colors = plotly.colors.qualitative.Plotly * 2
fig = go.Figure()
filtered_instr = [i for i in all_instructions if 'english_capital' in i or 'multiple' in i ] #or 'highlighted' in i or 'title' in i]

for i, instr in enumerate(all_instructions):

    df1 = df_no_steering_no_instr[df_no_steering_no_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
    df2 = df_steering_no_instr[df_steering_no_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
    df3 = df_no_steering_plus_instr[df_no_steering_plus_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
    df4 = df_steering_plus_instr[df_steering_plus_instr.instruction_id_list.apply(lambda x: x[0]) == instr]


    # make scatter plot of follow_all_instructions vs length_accuracy
    coordinates_no_steering_no_instr = df1[['follow_all_instructions', 'length_accuracy']].mean()
    coordinates_steering_no_instr = df2[['follow_all_instructions', 'length_accuracy']].mean()
    coordinates_no_steering_plus_instr = df3[['follow_all_instructions', 'length_accuracy']].mean()
    coordinates_steering_plus_instr = df4[['follow_all_instructions', 'length_accuracy']].mean()


    label = instr.split(':')[-1].title().replace('_', ' ')
    location = 'bottom right' if 'english' in instr else 'bottom left'

    # fig.add_trace(go.Scatter(x=[coordinates_no_steering_no_instr[0]], y=[coordinates_no_steering_no_instr[1]], mode='markers', name='No Instr.', marker=dict(size=10), marker_color=color1, opacity=0.6))
    # fig.add_trace(go.Scatter(x=[coordinates_steering_no_instr[0]], y=[coordinates_steering_no_instr[1]], mode='markers', name='No Instr. + Steering', marker=dict(size=10), marker_color=color1))
    fig.add_trace(go.Scatter(
        x=[coordinates_no_steering_plus_instr[0]], 
        y=[coordinates_no_steering_plus_instr[1]], 
        mode='markers+text', 
        name=f'{instr} + Instr.', 
        marker=dict(size=10), 
        marker_color=colors[i], 
        opacity=0.8,
        text=[label],  # Add instruction name as text
        textposition=location,  # Position the text
        textfont=dict(color='black')  # Set text color to red
    ))
    fig.add_trace(go.Scatter(
        x=[coordinates_steering_plus_instr[0]], 
        y=[coordinates_steering_plus_instr[1]], 
        mode='markers+text', 
        name=f'{instr} + Steering', 
        marker=dict(size=10), 
        marker_color=colors[i]
    ))

    # Add arrow annotation between the first and the second point
    # fig.add_annotation(
    #     dict(
    #         x=coordinates_steering_no_instr[0]-0.01,
    #         y=coordinates_steering_no_instr[1]-0.014,
    #         ax=coordinates_no_steering_no_instr[0],
    #         ay=coordinates_no_steering_no_instr[1],
    #         xref="x",
    #         yref="y",
    #         axref="x",
    #         ayref="y",
    #         showarrow=True,
    #         arrowhead=2,
    #         arrowsize=1,
    #         arrowwidth=2,
    #         arrowcolor="black"
    #     )
    # )

    # Add arrow annotation between the first and the second point
    fig.add_annotation(
        dict(
            x=coordinates_steering_plus_instr[0],
            y=coordinates_steering_plus_instr[1],
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

# add labels
fig.update_layout(
    xaxis_title='Format Instruction-following Accuracy',
    yaxis_title='Length Accuracy'
)


# set y limits
fig.update_layout(
    yaxis=dict(
        range=[0.1, 1.03]
    ),
    xaxis=dict(
        range=[0.3, 1.1]
    )
)

# add title
fig.update_layout(
    title='Multi-instruction Steering: Format & Length'
)

# resize the figure
fig.update_layout(
    width=400,
    height=300
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=10, t=50, b=0)
)

# remove legend
fig.update_layout(
    showlegend=False
)

# store the figure as pdf
fig.write_image(f'../plots/{model_name}_combination_negative_ex.pdf')

fig.show()


# %%
# =============================================================================
# make plots for highlighted sections and length
# =============================================================================


instr = [i for i in all_instructions if 'highlighted' in i][0]
df1 = df_no_steering_no_instr[df_no_steering_no_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
df2 = df_steering_no_instr[df_steering_no_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
df3 = df_no_steering_plus_instr[df_no_steering_plus_instr.instruction_id_list.apply(lambda x: x[0]) == instr]
df4 = df_steering_plus_instr[df_steering_plus_instr.instruction_id_list.apply(lambda x: x[0]) == instr]

# make scatter plot of follow_all_instructions vs length_accuracy
coordinates_no_steering_no_instr = df1[['follow_all_instructions', 'length_accuracy']].mean()
coordinates_steering_no_instr = df2[['follow_all_instructions', 'length_accuracy']].mean()
coordinates_no_steering_plus_instr = df3[['follow_all_instructions', 'length_accuracy']].mean()
coordinates_steering_plus_instr = df4[['follow_all_instructions', 'length_accuracy']].mean()

color1 = plotly.colors.qualitative.Plotly[4]
color2 = plotly.colors.qualitative.Bold[8]


fig = go.Figure()
fig.add_trace(go.Scatter(x=[coordinates_no_steering_no_instr[0]], y=[coordinates_no_steering_no_instr[1]], mode='markers', name='No Instr.', marker=dict(size=10), marker_color=color1, opacity=0.6))
fig.add_trace(go.Scatter(x=[coordinates_steering_no_instr[0]], y=[coordinates_steering_no_instr[1]], mode='markers', name='No Instr. + Steering', marker=dict(size=10), marker_color=color1))
fig.add_trace(go.Scatter(x=[coordinates_no_steering_plus_instr[0]], y=[coordinates_no_steering_plus_instr[1]], mode='markers', name='Instr.', marker=dict(size=10), marker_color=color2, opacity=0.6))
fig.add_trace(go.Scatter(x=[coordinates_steering_plus_instr[0]], y=[coordinates_steering_plus_instr[1]], mode='markers', name='Instr. + Steering', marker=dict(size=10), marker_color=color2))

# add labels
fig.update_layout(
    xaxis_title='Highlighted Sections Accuracy',
    yaxis_title='Length Accuracy'
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
        x=coordinates_steering_plus_instr[0]-0.012,
        y=coordinates_steering_plus_instr[1]-0.012,
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
        range=[0, 0.9]
    ),
    xaxis=dict(
        range=[0, 1.02]
    )
)

# add title
fig.update_layout(
    title='Instruction Composition: Highlighted Sections & Length'
)

# resize the figure
fig.update_layout(
    width=500,
    height=300
)

# remove padding
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=20)
)

# store the figure as pdf
# fig.write_image(f'./plots/{model_name}_highlighted_sections_length2.pdf')

fig.show()
# %%
# =============================================================================
# error analysis
# =============================================================================

correct2wrong = [0 for _ in range(len(df_no_steering_no_instr))]
correct2correct = [0 for _ in range(len(df_no_steering_no_instr))]
wrong2correct = [0 for _ in range(len(df_no_steering_no_instr))]
wrong2wrong = [0 for _ in range(len(df_no_steering_no_instr))]

df1 = df_no_steering_plus_instr
df2 = df_steering_plus_instr

accuracy_type = 'length_accuracy'
for i in range(len(df1)):
    if df1.iloc[i][accuracy_type] and not df2.iloc[i][accuracy_type]:
        correct2wrong[i] = 1
    elif df1.iloc[i][accuracy_type] and df2.iloc[i][accuracy_type]:
        correct2correct[i] = 1
    elif not df1.iloc[i][accuracy_type] and df2.iloc[i][accuracy_type]:
        wrong2correct[i] = 1
    elif not df1.iloc[i][accuracy_type] and not df2.iloc[i][accuracy_type]:
        wrong2wrong[i] = 1

df1['correct2wrong'] = correct2wrong
df1['correct2correct'] = correct2correct
df1['wrong2correct'] = wrong2correct
df1['wrong2wrong'] = wrong2wrong
print(df1.correct2wrong.mean())
print(df1.wrong2correct.mean())
print(df1.correct2correct.mean())
print(df1.wrong2wrong.mean())
# %%
# filtered_df = df1[(df1.correct2wrong == 1)]
filtered_df = df1[df1.instruction_id_list.apply(lambda x: 'english_cap' in x[0])]

# print some outputs
for i, r in filtered_df.iterrows():
    print(f'Prompt: {r.model_input}')
    print(f'Lenght constraint: {r.length_constraint+1}')
    print(f'Instruction: {r.instruction_id_list}')  
    print(f'Kwargs: {r.kwargs}')
    print(f'Response: {r.response}')
    print(f'Followed everything: {r[accuracy_type]}')
    print(f'Followed all instructions: {r.follow_all_instructions}')
    print('Length accuracy: ', r.length_accuracy)
    print('-----')
    print(f'Response w/ STEERING: {df2[df2.key == r.key].response.values[0]}')
    print('Follow everything w/ STEERING: ', df2[df2.key == r.key][accuracy_type].values[0])
    print(f'Followed all instructions w/ STEERING: {df2[df2.key == r.key].follow_all_instructions.values[0]}')
    print('Length accuracy w/ STEERING: ', df2[df2.key == r.key].length_accuracy.values[0])
    print('=====================')
# %%

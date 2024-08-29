# %%
import os
os.chdir('/home/t-astolfo/t-astolfo/composition')

import json
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
# %%
model_name = 'phi-3'

folder = f'./out/{model_name}/all_base_x_all_instr'
steering = 'no_instr'

path = f'{folder}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_no_instr = pd.DataFrame(data)

steering = 'adjust_rs_-1'

path = f'{folder}/{steering}/out.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_steering = pd.DataFrame(data)

steering = 'standard'

path = f'{folder}/{steering}/eval_results_loose.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_steering_standard = pd.DataFrame(data)


steering = 'instr_plus_adjust_rs_20'

path = f'{folder}/{steering}/eval_results_loose.jsonl'

with open(path, 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

df_steering_plus_instr = pd.DataFrame(data)

# setting = 'no_instr'
# steering = 'no_steeringno_length_steering'

# path = f'{folder}/{setting}/{steering}/out.jsonl'

# with open(path, 'r') as f:
#     lines = f.readlines()
# data = [json.loads(line) for line in lines]

# df_no_steering_no_instr = pd.DataFrame(data)


# %%
print(df_no_instr.follow_all_instructions.mean())
print(df_steering.follow_all_instructions.mean())
print(df_steering_standard.follow_all_instructions.mean())
print(df_steering_plus_instr.follow_all_instructions.mean())
# %%

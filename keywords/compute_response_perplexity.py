# %%
import os
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(script_dir)
sys.path.append(project_dir)

from utils.generation_utils import compute_perplexity

# %%
# =============================================================================
# Load the raw validation results
# =============================================================================

model_name = 'phi-3'
folder = f'{script_dir}/out/{model_name}/existence_validation/'
file_name = 'out.jsonl'
subfolders = os.listdir(folder)
print(f'subfolders: {subfolders}')
result_dict = {}
paths_dict = {}
for subfolder in subfolders:
    if 'adjust' in subfolder:
        continue

    if subfolder == 'no_instr' :
        layer = -1
        weight = -1
    else:
        layer = subfolder.split('_')[4]
        weight = subfolder.split('_')[-1]

    print(os.listdir(folder + subfolder))

    if file_name not in os.listdir(folder + subfolder):
        print(f'{subfolder} does not have the file {file_name}')
        continue

    with open(folder + subfolder + f'/{file_name}' ) as f:
        results = [json.loads(line) for line in f]

    results_df = pd.DataFrame(results)
    result_dict[(int(layer), int(weight))] = results_df
    paths_dict[(int(layer), int(weight))] = folder + subfolder + f'/{file_name}'

# %%
# =============================================================================
# compute perplexity for each response and store the results
# =============================================================================

# load model tokenizer
if model_name == 'phi-3':
    model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'
elif model_name == 'gemma-2-2b-it':
    model_name_hf = 'google/gemma-2-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name_hf)

device = 'cuda'
perplexity_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
perplexity_model.to(device)
perplexity_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

# %%
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
    lengths = []
    perplexities = []
    for i, row in value.iterrows():
        response = row['response']
        tokens = tokenizer.tokenize(response)
        lengths.append(len(tokens))

        # compute perplexity
        perplexities.append(compute_perplexity(response, device=device, perplexity_model=perplexity_model, perplexity_tokenizer=perplexity_tokenizer))

        p_bar.update(1)

    value['length'] = lengths
    value['perplexity'] = perplexities
    lengths_dict[key] = sum(lengths) / len(lengths)
    perplexitiy_dict[key] = sum(perplexities) / len(perplexities)

    # store the updated dataframe as jsonl
    new_path = paths_dict[key].replace('.jsonl', '_perplexity.jsonl')
    print(f'Saving the file at {new_path}')
    value.to_json(new_path, orient='records', lines=True)

# %%

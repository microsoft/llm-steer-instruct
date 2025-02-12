# %%
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')


# %%
model_name = 'phi-3'
folder = f'{script_dir}/out/{model_name}/forbidden_validation/'
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

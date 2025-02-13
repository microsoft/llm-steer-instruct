# %%
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
os.chdir(project_dir)

import sys
sys.path.append(script_dir)
sys.path.append(project_dir)

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from ifeval_scripts.evaluation_main import test_instruction_following_loose
import re
import torch
from tqdm import tqdm

# %%
# =============================================================================
# check for broken outputs
# =============================================================================

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
        try:
            outputs = perplexity_model(input_ids, labels=input_ids)
            loss = outputs.loss  # This is the average negative log-likelihood per token
        except Exception as e:
            print(f'Error in computing perplexity for text: {text}')
            print(f'Error: {e}')
            loss = torch.tensor(0.0)

    # Compute the perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

# %%

folder = 'ifeval_experiments/layer_search_out'
n_examples = 6
seed = 42

model_names = ['phi-3', 'mistral-7b-instruct', 'gemma-2-2b-it', 'gemma-2-9b-it', ]
model_names = ['gemma-2-2b', 'gemma-2-9b']
settings = ['instr', 'no_instr', ]
n_examples_dict = {'phi-3' : 8, 'mistral-7b-instruct': 8, 'gemma-2-2b-it': 8, 'gemma-2-9b-it': 6, 'gemma-2-2b': 10, 'gemma-2-9b': 6}

all_dfs = {}
all_paths = {}

for model_name in model_names:
    for setting in settings:
        n_examples = n_examples_dict[model_name]
        print(f'Processing {model_name} | {setting} | {n_examples} examples | seed {seed}')

        path = f'{folder}/{model_name}/n_examples{n_examples}_seed{seed}'
        file = f'{path}/out_{setting}.jsonl'
        with open(file, 'r') as f:
            results = [json.loads(line) for line in f]

        results_df = pd.DataFrame(results)
        all_dfs[f'{model_name}_{setting}'] = results_df
        all_paths[f'{model_name}_{setting}'] = path

# %%
for model_name in model_names:
    for setting in settings:
        results_df = all_dfs[f'{model_name}_{setting}']
        print(f'{model_name} | {setting}')

        if model_name == 'phi-3':
            model_name_hf = 'microsoft/Phi-3-mini-4k-instruct'
        if 'gemma' in model_name:
            model_name_hf = f'google/{model_name}'
        elif model_name == 'mistral-7b-instruct':
            model_name_hf = 'mistralai/Mistral-7B-Instruct-v0.1'
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf)

        accuracy_with_quality_check = []
        perplexities = []

        p_bar = tqdm(total=len(results_df))
        for i, r in results_df.iterrows():
            # compute accuracy
            response  = r['response']
            tokens = tokenizer.tokenize(response)

            # fix problem with capital word frequency
            if r.single_instruction_id == 'change_case:capital_word_frequency':
                if 'less than' in r.prompt or 'at most' in r.prompt:
                    relation = 'less than'
                elif 'more than' in r.prompt or 'at least' in r.prompt:
                    relation = 'at least'
                # parse the last number in the prompt
                num = int(re.findall(r'\d+', r.prompt)[-1])

                new_kwargs = [{"capital_relation": relation,"capital_frequency": num}]
                r.kwargs = new_kwargs
                prompt_to_response = {}
                prompt_to_response[r['prompt']] = r['response']
                output = test_instruction_following_loose(r, prompt_to_response)
                # update follow_all_instructions
                results_df.loc[i, 'follow_all_instructions'] = output.follow_all_instructions

            # compute perplexity
            perplexities.append(compute_perplexity(response))
            p_bar.update(1)

        results_df['perplexity'] = perplexities

        # store the new results_df as a jsonl file
        new_dir = f'{all_paths[f"{model_name}_{setting}"]}_with_perplexity/'
        os.makedirs(new_dir, exist_ok=True)
        results_df.to_json(f'{new_dir}/out_{setting}.jsonl', orient='records', lines=True)

# %%
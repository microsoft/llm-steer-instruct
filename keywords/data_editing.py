# %%
import pandas as pd
import json
import os
import sys
import random
sys.path.append('/Users/alestolfo/workspace/llm-steer-instruct/')
# %%
path = '../data/keyword_test.jsonl'

with open(path, 'r') as f:
    results = [json.loads(line) for line in f]

data_df = pd.DataFrame(results)
# %%
phrasings_include = [' Include the word {} in your answer.', ' Make sure to include the word "{}".', ' The output should contain the word "{}".', ' The output must contain the word "{}".', ' Make sure that the word "{}" is included in the output.', ' The output must include the word "{}".']
new_rows = []
for i, r in data_df.iterrows():
    for word in r['unlikely_words']:
        new_row = r.copy()
        new_row['prompt_without_instruction'] = r['question'] 
        new_row["instruction_id_list"] = ["keywords:existence"]
        new_row["instruction_id_list_for_eval"] = ["keywords:existence"]
        new_row['kwargs'] = [{"keywords": [word]}]
        phrasing = random.choice(phrasings_include)
        new_prompt = new_row['prompt_without_instruction'] + phrasing.format(word)
        new_row['prompt'] = new_prompt
        new_rows.append(new_row)

new_data_df = pd.DataFrame(new_rows)
# %%
# store the new data
new_data_df.to_json('../data/keyword_test_inclusion.jsonl', orient='records', lines=True)
# %%

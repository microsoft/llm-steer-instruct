# %%
import json
import pandas as pd
import re

path = '/Users/alestolfo/workspace/llm-steer-instruct/data/format/all_base_x_all_instructions_filtered.jsonl'

with open(path, 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)


# %%
# Dictionary to map numeral words to numbers
numeral_words = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12
}
# %%
new_rows = []
for i, r in df.iterrows():
    new_row = r.copy()
    new_row['prompt'] = r.model_output
    new_row.pop('model_output')
    new_row.pop('icl_prompt_without_instruction')
    new_row.pop('icl_key')
    new_row.pop('icl_prompt_with_single_instruction')
    new_row.pop('prompt_hash')
    new_row.pop('is_valid')
    new_row.pop('single_instruction_kwargs_str')
    if r.single_instruction_id == 'change_case:capital_word_frequency':
        prompt = r.model_output
        if 'less than' in prompt.lower() or 'at most' in prompt.lower():
            relation = 'less than'
        elif 'more than' in prompt.lower() or 'at least' in prompt.lower():
            relation = 'at least'
        else:
            range_match = re.findall(r'(\d+)\s+to\s+(\d+)', r.model_output)
            if range_match:
                relation = 'at most'
            else:
                print(f'No relation found in prompt: {r.model_output}')
                break
        # parse the last number in the prompt
        num_match = re.findall(r'\d+', r.model_output)
        num = None
        if num_match:
            num = int(num_match[-1])
        else:
            words = r.model_output.replace(',', '').replace('.', '').split()
            for word in reversed(words):
                if word.lower() in numeral_words:
                    num = numeral_words[word.lower()]
                    break
        
        if num is None:
            print(f'Number not found in prompt: {r.model_output}')
            break

        new_kwargs = [{"capital_relation": relation,"capital_frequency": num}]
        new_row['kwargs'] = new_kwargs

    new_rows.append(new_row)

new_df = pd.DataFrame(new_rows)

# %%
# write out the new dataframe
new_path = '/Users/alestolfo/workspace/llm-steer-instruct/data/format/ifeval_augmented_filtered.jsonl'
with open(new_path, 'w') as f:
    for i, r in new_df.iterrows():
        f.write(json.dumps(r.to_dict()) + '\n')
# %%
path = '/Users/alestolfo/workspace/llm-steer-instruct/data/format/input_data_single_instr.jsonl'

with open(path, 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
print(f'Length of original dataframe: {len(df)}')
# %%
# filter for format only 

# filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
df = df[df.instruction_id_list.apply(lambda x: any([f in x[0] for f in filters]))]

print(f'Length of filtered dataframe: {len(df)}')
# %%
# write out the new dataframe
new_path = '/Users/alestolfo/workspace/llm-steer-instruct/data/format/ifeval_single_instr_format.jsonl'
df.to_json(new_path, orient='records', lines=True)
# %%
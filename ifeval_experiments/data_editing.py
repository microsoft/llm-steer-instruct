# %%
import json
import pandas as pd
import plotly.express as px
import random
# %%

# read in ../data/input_data.jsonl
with open('../data/input_data_no_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
data_df = pd.DataFrame(data)

with open('../data/input_data.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
data_df_with_kwargs = pd.DataFrame(data)

# join the dataframes using column "key" adding only kwargs from data_df_with_kwargs
data_df = data_df.join(data_df_with_kwargs[['key', 'kwargs']].set_index('key'), on='key')

# %%
# filter data_df keeping the rows for which instruction_id_list has length 1
filtered_data_df = data_df[data_df['instruction_id_list'].apply(len) == 1]
# %%
# store the filtered data as ../data/input_data_single_instr.jsonl

with open('../data/input_data_single_instr_no_instr.jsonl', 'w') as f:
    for line in filtered_data_df.to_dict(orient='records'):
        f.write(json.dumps(line) + '\n')

# %%
# =============================================================================

# load ./out/phi-3/single_instr/add_vector_1.0_20/out.jsonl
with open('./ifeval_experiments/out/phi-3/single_instr/adjust_rs_20/out.jsonl') as f:
    out = f.readlines()
    out = [json.loads(d) for d in out]
out_df = pd.DataFrame(out)
# %%
out_df['response_list'] = out_df['response']

# %%
# make 'response' a string (the first entropy in the list)
out_df['response'] = out_df['response'].apply(lambda x: x[0])
# %%
out_df
# %%
# store the out_df as ./out/phi-3/single_instr/add_vector_1.0_20/out_df.jsonl
out_df.to_json('./ifeval_experiments/out/phi-3/single_instr/adjust_rs_20/out_df.jsonl', orient='records', lines=True)

# %%
# =============================================================================
# create filtered data for IV computation
# =============================================================================

# load ../data/all_base_x_all_instructions.jsonl
with open('../data/all_base_x_all_instructions.jsonl') as f:
    all_base_x_all_instructions = f.readlines()
    all_base_x_all_instructions = [json.loads(d) for d in all_base_x_all_instructions]

data_df = pd.DataFrame(all_base_x_all_instructions)

# %%
for i, d in data_df[data_df['single_instruction_id'] == 'detectable_format:json_format' ].iterrows():
    ex = d.model_output
    prompt_without_instruction = d.prompt_without_instruction
    print(f'ex: {ex}')
    print(f'prompt_without_instruction: {prompt_without_instruction}')
    print('-----------------')
# %%
c = 0
for i, d in data_df.iterrows():
    if '<original_question>' not in d.model_output and '<output>' in d.model_output:
        print(d.model_output)
        c += 1
print(c)
# %%
# load ../data/input_Data_no_instr.jsonl
with open('../data/input_data_no_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
ifeval_data_df = pd.DataFrame(data)
# %%
all_instructions = data_df.single_instruction_id.unique()
all_instructions
# %%
filtered_data = []
c = 0
for i, r in data_df.iterrows():
    if '<original_question>' in r.model_output:
        continue
    if '<output>' in r.model_output:
        continue
    if 'json{' in r.model_output:
        continue 
    # get the entry in ifeval_data_df that corresponds to the current row, matching on 'key'
    instr_data = ifeval_data_df[ifeval_data_df['key'] == r.key]
    if len(instr_data) != 1:
        raise ValueError(f'len(instr_data) = {len(instr_data)}')
    instr_data = instr_data.iloc[0]
    r['instruction_id_list_og'] = [r.single_instruction_id]
    if r.single_instruction_id in instr_data.instruction_id_list:
        continue

    if r.single_instruction_id == 'language:response_language':
        language = r.single_instruction_kwargs['language']
        r.single_instruction_id = f'language:response_language_{language}'
    if r.single_instruction_id == 'keywords:frequency':
        if 'relation' in r.single_instruction_kwargs:
            relation = r.single_instruction_kwargs['relation']
            r.single_instruction_id = f'keywords:frequency_{relation}'
        else:
            r.single_instruction_id = f'keywords:frequency_at least'
    if r.single_instruction_id == 'keywords:letter_frequency':
        relation = r.single_instruction_kwargs['let_relation']
        r.single_instruction_id = f'keywords:letter_frequency_{relation}'
    if r.single_instruction_id == 'length_constraints:number_sentences':
        relation = r.single_instruction_kwargs['relation']
        r.single_instruction_id = f'length_constraints:number_sentences_{relation}'
    if r.single_instruction_id == 'length_constraints:number_words':
        if 'relation' not in r.single_instruction_kwargs:
            c += 1
            print(f'single_instruction_kwargs: {r.single_instruction_kwargs}')
            continue
        relation = r.single_instruction_kwargs['relation']
        r.single_instruction_id = f'length_constraints:number_words_{relation}'

    filtered_data.append(r.to_dict())

filtered_data_df = pd.DataFrame(filtered_data)
# %%
filtered_data_df['kwargs'] = [[d] for d in filtered_data_df['single_instruction_kwargs']]
filtered_data_df['instruction_id_list'] = [[d] for d in filtered_data_df['single_instruction_id']]
update_instr_name(filtered_data_df)
len(filtered_data_df)

# %%
# make histogram of single_instruction_id
fig = px.histogram(filtered_data_df, x='single_instruction_id')
# sort the histogram
fig.update_xaxes(categoryorder='total descending') 

# incline the text
fig.update_xaxes(tickangle=45)
fig.show()


# %%
# store the filtered data as ../data/all_base_x_all_instructions_filtered.jsonl

with open('../data/all_base_x_all_instructions_filtered.jsonl', 'w') as f:
    for line in filtered_data_df.to_dict(orient='records'):
        f.write(json.dumps(line) + '\n')
# %%
# =============================================================================
# update the instr names for evaluation
# =============================================================================

# read in ../data/input_data.jsonl
with open('../data/input_data_single_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
data_df_wit = pd.DataFrame(data)

# %%
data_df_wit
# %%
def update_instr_name(df):

    def merge_dicts(dict_list):
        merged_dict = {}
        for d in dict_list:
            merged_dict.update(d)
        return merged_dict

    updated_instr_list = []
    for i, r in df.iterrows():
        new_i = []
        kwargs = merge_dicts(r.kwargs)
        for instr in r.instruction_id_list:
            if instr == 'language:response_language':
                language = kwargs['language']
                instr = f'language:response_language_{language}'
            if instr == 'keywords:frequency':
                if 'relation' in kwargs:
                    relation = kwargs['relation']
                    instr = f'keywords:frequency_{relation}'
                else:
                    instr = f'keywords:frequency_at least'
            if instr == 'keywords:letter_frequency':
                relation = kwargs['let_relation']
                instr = f'keywords:letter_frequency_{relation}'
            if instr == 'length_constraints:number_sentences':
                relation = kwargs['relation']
                instr = f'length_constraints:number_sentences_{relation}'
            if instr == 'length_constraints:number_words':
                relation = kwargs['relation']
                instr = f'length_constraints:number_words_{relation}'
            new_i.append(instr)
        updated_instr_list.append(new_i)

    df['instruction_id_list_for_eval'] = updated_instr_list

    return df
# %%
data_df_wit = update_instr_name(data_df_wit)
data_df_wit.head(30)
# %%
# store the df as jsonl
data_df_wit.to_json('../data/input_data_single_instr.jsonl', orient='records', lines=True)
# %%
with open('./data/input_data_single_instr_no_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
data_df_wo = pd.DataFrame(data)
# %%
data_df_wo = update_instr_name(data_df_wo)
data_df_wo.head(30)

# %%
# store the df as jsonl
data_df_wo.to_json('./data/input_data_single_instr_no_instr.jsonl', orient='records', lines=True)
# %%
# =============================================================================
# data for multiple instrctions
# =============================================================================

# read in ../data/input_data.jsonl
with open('../data/input_data.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df = pd.DataFrame(data)
# %%
# add prompt_without_instruction to data_df
with open('../data/input_data_no_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df_no_instr = pd.DataFrame(data)

# rename the column 'prompt' to 'prompt_without_instruction'
data_df_no_instr.rename(columns={'prompt': 'prompt_without_instruction'}, inplace=True)

# join the dataframes using column "key" adding only prompt_without_instruction from data_df_no_instr
data_df = data_df.join(data_df_no_instr[['key', 'prompt_without_instruction']].set_index('key'), on='key')


# %%
# filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
filtered_data_df = data_df[data_df.instruction_id_list.apply(lambda x: all([any([f in item for f in filters]) for item in x]))]


# filter instruction_id list to keep only the ones with length > 1
filtered_data_df = filtered_data_df[data_df['instruction_id_list'].apply(len) > 1]
# %%
update_instr_name(filtered_data_df)
filtered_data_df
# %%
# store the filtered data as ../data/input_data_multiple_instr_nonpar.jsonl

with open('../data/input_data_multiple_instr_nonpar.jsonl', 'w') as f:
    for line in filtered_data_df.to_dict(orient='records'):
        f.write(json.dumps(line) + '\n')

# %%
# =============================================================================
# add prompt w/o instr to data
# =============================================================================
# read in ../data/input_data.jsonl
with open('./data/input_data_single_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df = pd.DataFrame(data)

# load ../data/input_data_single_instr_no_instr.jsonl
with open('./data/input_data_single_instr_no_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df_no_instr = pd.DataFrame(data)
# %%
# rename the column 'prompt' to 'prompt_without_instruction'
data_df_no_instr.rename(columns={'prompt': 'prompt_without_instruction'}, inplace=True)

# join the dataframes using column "key" adding only prompt_without_instruction from data_df_no_instr
data_df = data_df.join(data_df_no_instr[['key', 'prompt_without_instruction']].set_index('key'), on='key')
# %%
# store the df as jsonl
data_df.to_json('./data/input_data_single_instr.jsonl', orient='records', lines=True)
# %%
# =============================================================================
# look at keyword data
# =============================================================================

# load ../data/all_base_x_all_instructions_filtered.jsonl
with open('./data/input_data_single_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df = pd.DataFrame(data)

# %%
filtered_df = data_df[data_df['instruction_id_list'].apply(lambda x: any('forbidden' in y for y in x))]
print(len(filtered_df))
# %%
for i in filtered_df.index:
    print(f'prompt: {filtered_df.loc[i].prompt}')
    print(f'kwargs: {filtered_df.loc[i].kwargs}')
    print('-----------------')
# %%
# gather all the keywords
keywords = []
new_prompts = []
for i in filtered_df.index:
    keywords.extend(filtered_df.loc[i].kwargs[0]['forbidden_words'])
    
# %%
# store keywords in a txt file
with open('./data/ifeval_keywords_exclude.txt', 'w') as f:
    for k in keywords:
        f.write(k + '\n')
# %%
phrasings_exclude = [' Do not include the word {}.', ' Make sure not to include the word "{}".', ' Do not use the word {}.', ' Do not say "{}".', ' Please exclude the word "{}".', ' The output should not contain the word "{}".']
new_rows = []
for i, r in data_df.iterrows():
    instr = r.instruction_id_list[0]
    if 'forbidden' not in instr:
        continue
    # get words to exclude
    word_list = r.kwargs[0]['forbidden_words']
    for word in word_list:
        new_row = dict(r)
        new_row['kwargs'] = [{'forbidden_words': [word]}]
        phrasing = random.choice(phrasings_exclude)
        new_prompt = r.prompt_without_instruction + phrasing.format(word)
        new_row['old_prompt'] = r.prompt
        new_row['prompt'] = new_prompt
        new_rows.append(new_row)


new_df = pd.DataFrame(new_rows)
print(len(new_df))
# %%
# store the new_df as jsonl
new_df.to_json('../data/ifeval_single_keyword_exclude.jsonl', orient='records', lines=True)
# %%

# =============================================================================
# look at keyword data: include
# =============================================================================

# load ../data/all_base_x_all_instructions_filtered.jsonl
with open('./data/input_data_single_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df = pd.DataFrame(data)

# %%
filtered_df = data_df[data_df['instruction_id_list'].apply(lambda x: any('existence' in y for y in x))]
print(len(filtered_df))
# %%
for i in filtered_df.index:
    print(f'prompt: {filtered_df.loc[i].prompt}')
    print(f'kwargs: {filtered_df.loc[i].kwargs}')
    print('-----------------')
# %%
# gather all the keywords
keywords = []
new_prompts = []
for i in filtered_df.index:
    keywords.extend(filtered_df.loc[i].kwargs[0]['keywords'])
    
# %%
# store keywords in a txt file
with open('./data/ifeval_keywords_include.txt', 'w') as f:
    for k in keywords:
        f.write(k + '\n')
# %%
phrasings_exclude = [' Include the word {}.', ' Make sure to include the word "{}".', ' The output should contain the word "{}".', ' The output must contain the word "{}".', ' Make sure that the word "{}" is included in the output.', ' The output must include the word "{}".']
new_rows = []
for i, r in data_df.iterrows():
    instr = r.instruction_id_list[0]
    if 'existence' not in instr:
        continue
    # get words to exclude
    word_list = r.kwargs[0]['keywords']
    for word in word_list:
        new_row = dict(r)
        new_row['kwargs'] = [{'keywords': [word]}]
        phrasing = random.choice(phrasings_exclude)
        new_prompt = r.prompt_without_instruction + phrasing.format(word)
        new_row['old_prompt'] = r.prompt
        new_row['prompt'] = new_prompt
        new_rows.append(new_row)


new_df = pd.DataFrame(new_rows)
print(len(new_df))
# %%
# store the new_df as jsonl
new_df.to_json('./data/ifeval_single_keyword_include.jsonl', orient='records', lines=True)
# %%

# =============================================================================
# create data for multiple instructions: casing + word exclusion
# =============================================================================

# load ../data/input_data.jsonl
with open('../data/input_data_single_instr.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

data_df = pd.DataFrame(data)
# %%
# filter for instructions that are change_case:english_lowercase
filtered_data_df = data_df[data_df['instruction_id_list'].apply(lambda x: any('change_case:english_lowercase' in y for y in x))]
filtered_data_df = filtered_data_df.reset_index(drop=True)
len(filtered_data_df)
# %%
filtered_data_df
# %%
for i, r in filtered_data_df.iterrows():
    print(f'prompt: {r.prompt}')
    print(f'kwargs: {r.kwargs}')
    print('-----------------')
# %%
words_to_exclude1 = ['what', 'vote', 'movie', 'debt', 'friend', 'united', 'etching', 'children', 'sleep', 'health', 'growth', 'motor', 'bird']
words_to_exclude2 = ['boys', 'duty', 'fincher', 'issue', 'dear', 'president', 'inscribing', 'teaching', 'breathing', 'greens', 'market', 'engine', 'mouse']

words_to_exclude = words_to_exclude1 + words_to_exclude2

# duplicate the rows in filtered_data_df for each word in words_to_exclude
filtered_data_df = pd.concat([filtered_data_df]*2, ignore_index=True)

filtered_data_df['kwargs'] = [[{}, {'forbidden_words': [w]}] for w in words_to_exclude]

# %%
phrasings_exclude = [' Do not include the word {}.', ' Make sure not to include the word "{}".', ' Do not use the word {}.', ' Do not say "{}".', ' Please exclude the word "{}".', ' The output should not contain the word "{}".']

new_rows = []

for i, r in filtered_data_df.iterrows():
    new_row = dict(r)
    phrasing = random.choice(phrasings_exclude)
    new_prompt = filtered_data_df.loc[i].prompt + phrasing.format(words_to_exclude[i])
    new_row['old_prompt'] = filtered_data_df.loc[i].prompt
    new_row['prompt'] = new_prompt
    new_instr_list = new_row['instruction_id_list'].copy() + ['keywords:forbidden_words']
    new_row['instruction_id_list'] = new_instr_list
    new_rows.append(new_row)

new_df = pd.DataFrame(new_rows)
# %%
len(new_df)
# %%
# store the new_df as jsonl
new_df.to_json('../data/ifeval_multiple_instr_casing_exclude.jsonl', orient='records', lines=True)
# %%
# =============================================================================
# add steering layer to out data
# =============================================================================
# load ./ifeval_experiments/out/phi-3/single_instr/all_base_x_all_instr/adjust_rs_-1/out.jsonl
with open('./ifeval_experiments/out/phi-3/single_instr/all_base_x_all_instr/adjust_rs_-1/out.jsonl') as f:
    out = f.readlines()
    out = [json.loads(d) for d in out]

out_df = pd.DataFrame(out)
# %%

# load ifeval_experiments/representations/phi-3/single_instr_all_base_x_all_instr/pre_computed_ivs_best_layer_validation_no_instr.h5
ivs = pd.read_hdf('./ifeval_experiments/representations/phi-3/single_instr_all_base_x_all_instr/pre_computed_ivs_best_layer_validation_no_instr.h5')

# %%
best_layer_dict = { r.instruction: r.max_diff_layer_idx for i, r in ivs.iterrows() }
steering_layers = []
for i, r in out_df.iterrows():
    if r.instruction_id_list[0] not in best_layer_dict:
        steering_layers.append(-1)
        continue
    steering_layers.append(best_layer_dict[r.instruction_id_list[0]])

out_df['steering_layer'] = steering_layers
# %%
# write out_df to ./ifeval_experiments/out/phi-3/single_instr/all_base_x_all_instr/adjust_rs_-1/out_df.jsonl
out_df.to_json('./ifeval_experiments/out/phi-3/single_instr/all_base_x_all_instr/adjust_rs_-1/out_w_steering_layer.jsonl', orient='records', lines=True)

# %%

# %%
import os
from tqdm import tqdm
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import hydra
from omegaconf import DictConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')

sys.path.append(script_dir)
sys.path.append(project_dir)

from utils.generation_utils import compute_perplexity

config_path = os.path.join(project_dir, 'config/format')


@hydra.main(config_path=config_path, config_name='compute_perplexity')
def compute_response_perplexity(args: DictConfig):
        
        folder = f'{script_dir}/{args.layer_search_folder}'
        
        setting = 'instr' if args.include_instructions else 'no_instr'

        print(f'Processing {args.model_name} | {setting} | {args.n_examples} examples | seed {args.seed}')

        path = f'{folder}/{args.model_name}/n_examples{args.n_examples}_seed{args.seed}'
        file = f'{path}/out_{setting}.jsonl'
        with open(file, 'r') as f:
            results = [json.loads(line) for line in f]

        results_df = pd.DataFrame(results)
        perplexities = []

        perplexity_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2', cache_dir=args.transformers_cache_dir)
        perplexity_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2', cache_dir=args.transformers_cache_dir)
        perplexity_model.to(args.device)

        p_bar = tqdm(total=len(results_df))
        for i, r in results_df.iterrows():
            # compute accuracy
            response  = r['response']

            # compute perplexity
            perplexities.append(compute_perplexity(response, perplexity_model=perplexity_model, perplexity_tokenizer= perplexity_tokenizer, device=args.device))
            p_bar.update(1)

        results_df['perplexity'] = perplexities

        # store the new results_df as a jsonl file
        new_dir = f'{path}_with_perplexity/'
        os.makedirs(new_dir, exist_ok=True)
        results_df.to_json(f'{new_dir}/out_{setting}.jsonl', orient='records', lines=True)


if __name__ == '__main__':
    compute_response_perplexity()
# %%
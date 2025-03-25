from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer, loading_from_pretrained

def tl_name_to_hf_name(model_name): 
    hf_model_name = loading_from_pretrained.get_official_model_name(model_name)
    #if "llama" in hf_model_name.lower(): 
    #    hf_model_name = "meta-llama/"+hf_model_name
    return hf_model_name

def load_model_from_tl_name(model_name, device='cuda', cache_dir=None, hf_token=None, hf_model=False): 
    hf_model_name = tl_name_to_hf_name(model_name)
    print(f"Loading model from {hf_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True, cache_dir=cache_dir, token=hf_token)

    if hf_model:
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, token=hf_token, cache_dir=cache_dir, local_files_only=False)
        return model, tokenizer

    #loading model 
    if True or "llama" in model_name.lower() or "gemma" in model_name.lower() or "mistral" in model_name.lower(): 
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, token=hf_token, cache_dir=cache_dir)
        model = HookedTransformer.from_pretrained(model_name=model_name, hf_model=hf_model, tokenizer=tokenizer, device=device, cache_dir=cache_dir, local_files_only=False)
    else: 
        # model = HookedTransformer.from_pretrained(model_name, device=device, cache_dir=cache_dir, tokenizer=tokenizer)
        print(f'chache dir: {cache_dir}')
        model = HookedTransformer.from_pretrained(model_name, device=device, cache_dir=cache_dir)


    return model, tokenizer 
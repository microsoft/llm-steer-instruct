import torch
from tqdm import tqdm

def cot_inference(model, tokenizer, problem, device, use_qa_pattern=False):
    """
    zero-shot chain-of-thought inference using the two-stage prompt method proposed by Kojima et al. (2022)
    """
    if use_qa_pattern:
        eval_prompt = "Q: " + problem + "\nA: Let's think step by step. " 
    else:
        eval_prompt = problem + "\nLet's think step by step. " 

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        len_prompt_pre_cot = model_input['input_ids'].shape[1]
        output = model.generate(model_input['input_ids'], max_new_tokens=256, do_sample=False)[0,:]
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)

    eval_prompt2 = decoded_output + "\nTherefore, the answer (Arabic numerals) is "

    model_input = tokenizer(eval_prompt2, return_tensors="pt").to(device)
    with torch.no_grad():
        len_prompt = model_input['input_ids'].shape[1]
        output = model.generate(model_input['input_ids'], max_new_tokens=16, do_sample=False)
        decoded_output_with_cot = tokenizer.decode(output[0,len_prompt_pre_cot:], skip_special_tokens=True)
        decoded_output_with_cot = decoded_output_with_cot.replace(',', '')
        decoded_output_without_cot = tokenizer.decode(output[0,len_prompt:], skip_special_tokens=True)
        decoded_output_without_cot = decoded_output_without_cot.replace(',', '')
    
    return decoded_output_with_cot, decoded_output_without_cot

def direct_inference(model, tokenizer, problem, device, use_qa_pattern=False):
    """
    standard direct inference, 0-shot
    """
    if use_qa_pattern:
        eval_prompt = 'Q: ' + problem + "\nA: The answer (Arabic numerals) is "
    else:
        eval_prompt = problem + "\nThe answer (Arabic numerals) is "

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        len_prompt = model_input['input_ids'].shape[1]
        output = model.generate(model_input['input_ids'], max_new_tokens=16, do_sample=False)[0,len_prompt:]
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    
    decoded_output = decoded_output.replace(',', '') # models sometimes output 1,000 instead of 1000
    return decoded_output


def if_inference(model, tokenizer, problem, device, max_new_tokens=512):
    """
    standard direct inference, 0-shot
    """
    eval_prompt = problem 

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        len_prompt = model_input['input_ids'].shape[1]
        # check whether model is an instance of HookedTransformer
        if hasattr(model, 'W_in'):
            # TODO the 32007 token is phi-specific
            output = model.generate(model_input['input_ids'], max_new_tokens=max_new_tokens, do_sample=False, verbose=False, stop_at_eos=True, eos_token_id=[tokenizer.eos_token_id, 32007])[0,len_prompt:]
        else:
            output = model.generate(model_input['input_ids'], max_new_tokens=max_new_tokens, do_sample=False)[0,len_prompt:]
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    
    return decoded_output


def adjust_vectors(v, u, target_values):
    """
    Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values.

    Parameters:
    - v: A 2D tensor of shape (n, d), representing the batch of vectors to be adjusted.
    - u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made.
    - target_values: A 1D tensor of shape (n,), representing the desired projection values of the vectors in v along u.

    Returns:
    - adjusted_v: The adjusted batch of vectors such that their projections along u are equal to the target values.
    """
    current_projections = v @ u  # Current projections of v onto u
    delta = target_values - current_projections  # Differences needed to reach the target projections
    adjusted_v = v + delta[:, None] * u  # Adjust v by the deltas along the direction of u
    return adjusted_v


def generate_with_hooks(
    model,
    toks,
    max_tokens_generated: int = 64,
    fwd_hooks = [],
    verbose: bool = False,
    decode_directly=False
):

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    p_bar = tqdm(range(max_tokens_generated)) if verbose else range(max_tokens_generated)
    with torch.no_grad():
        for i in p_bar:
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :-max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
                if next_tokens[0] == model.tokenizer.eos_token_id or next_tokens[0] == 32007:
                    break
                if next_tokens[0] == 235292 and all_toks[0, -max_tokens_generated+i-1] == 235368:
                    print(f'Stopping the generation as the model is generating a new question (Q:)')
                    # remove the Q
                    all_toks[0, -max_tokens_generated+i-1] = 0
                    break
                all_toks[:,-max_tokens_generated+i] = next_tokens

    # truncate the tensor to remove padding
    all_toks = all_toks[:, :toks.shape[1] + i]

    if decode_directly:
        return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)
    else:
        return all_toks

def direction_ablation_hook(
    activation,
    hook,
    direction,
    weight=1,
):
    return activation + (direction * weight)

def direction_projection_hook(
    activation,
    hook,
    direction,
    value_along_direction,
):
    adjusted_activations = adjust_vectors(activation.squeeze(), direction, value_along_direction)
    return adjusted_activations.unsqueeze(0)
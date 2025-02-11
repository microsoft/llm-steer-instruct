import torch
from tqdm import tqdm


def generate(model, tokenizer, prompt, device, max_new_tokens=512):
    """
    standard direct inference, 0-shot
    """

    model_input = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        len_prompt = model_input['input_ids'].shape[1]
        # check whether model is an instance of HookedTransformer
        if hasattr(model, 'W_in'):
            # token 32007 is phi-specific
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


def activation_addition_hook(
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


def generate_with_hooks(
    model,
    toks,
    max_tokens_generated: int = 64,
    fwd_hooks = [],
    verbose: bool = False,
    return_decoded=False
):

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    p_bar = tqdm(range(max_tokens_generated)) if verbose else range(max_tokens_generated)
    with torch.no_grad():
        for i in p_bar:
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :-max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy decoding
                if next_tokens[0] == model.tokenizer.eos_token_id or next_tokens[0] == 32007:
                    break
                if next_tokens[0] == 235292 and all_toks[0, -max_tokens_generated+i-1] == 235368:
                    # Stopping the generation as the model is generating a new question (Q:)
                    # remove the Q
                    all_toks[0, -max_tokens_generated+i-1] = 0
                    break
                all_toks[:,-max_tokens_generated+i] = next_tokens

    # truncate the tensor to remove padding
    all_toks = all_toks[:, :toks.shape[1] + i]

    if return_decoded:
        return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)
    else:
        return all_toks


def compute_perplexity(text, device='cuda', perplexity_model=None, perplexity_tokenizer=None):
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

        print

    # Compute the perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()


def extract_representation(model, tokenizer, problem, device, num_final_tokens=8):
    """
    extract the representation of the final token in the direct inference prompt
    """
    eval_prompt = problem

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(model_input['input_ids'])
        del logits
        final_token_rs = torch.stack([cache['resid_post', layer_idx][:, -num_final_tokens:, :].squeeze() for layer_idx in range(model.cfg.n_layers)]).cpu().numpy()
        del cache
    
    return final_token_rs
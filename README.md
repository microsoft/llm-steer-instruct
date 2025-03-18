# Improving Instruction-Following in Language Models through Activation Steering

This repository contains the code for the paper “Improving Instruction-Following in Language Models through Activation Steering,” under submission at ICLR 2025.

## Setup

1. Install the necessary dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Add your HuggingFace token to `./hf_token.txt` for accessing gated repositories (e.g., Gemma 2).

3. We use Hydra for parameter configuration. All config files can be found in `config/`.

---

## Format Instructions

1. **Compute Representations**
   - Script: `ifeval_experiments/compute_representations.py`
   - Description: Runs the model on pairs of inputs (with and without instructions) and stores the hidden states of the last token for each layer.
   - Config: `config/compute_representations.yaml`

2. **Find Best Layer for Steering**
   - Script: `ifeval_experiments/find_best_layer.py`
   - Description: Runs the model on validation data, performs steering at multiple layers, and stores the outputs.
   - Config: `config/find_best_layer.yaml`

3. **Pre-compute Instruction Steering Vectors**
   - Script: `ifeval_experiments/pre_compute_ivs.py`
   - Description: Computes the instruction vectors at the optimal steering layer based on the representations and validation scores.

4. **Evaluate Format Instructions**
   - Script: `ifeval_experiments/ifeval_evaluation.py`
   - Config: `config/conf.yaml`
   - Description: Evaluates the model on a subset of IFEval (set `nonparametric_only=True` for format instructions). For cross-model experiments, set `model_name` to either `gemma-2-2b` or `gemma-2-9b` and enable the `cross-model` flag. Note: Cross-model experiments require representations from the instruction-tuned counterpart.

---

## Length Instructions

1. **Compute Representations**
   - Script: `length_constraints/compute_length_representations.py`
   - Config: `config/compute_length_representations.yaml`
   - Description: Computes representations for length constraints and stores them in `length_constraints/representations/`.

2. **Evaluate Length Instructions**
   - Script: `length_constraints/evaluate_length_constraints.py`
   - Config: `config/conf_length.yaml`
   - Description: Evaluates the model on length constraints, analogous to the format instructions evaluation.

---

## Word-Specific Instructions

1. **Compute Keyword Representations**
   - Script: `keywords/compute_keywords_representations.py`
   - Config: `config/compute_keyword_representations.yaml`
   - Description: Computes representations for keyword constraints using base queries from `data/ifeval_wo_instructions.py` and stores them in `keywords/representations/`.

2. **Evaluate Keyword Instructions**
   - Script: `keywords/eval_keyword_constraints.py`
   - Config: Specify `specific_constraint` as `existence` (for inclusion) or `forbidden` (for exclusion). To test on validation data, use `existence_validation` or `forbidden_validation`.

---

## Multi-instruction Steering

1. **Evaluate Format + Length Instructions**
   - Script: `composition/nonpar_plus_length.py`
   - Config: `config/nonpar_plus_length.yaml`
   - Description: Evaluates the model on format and length instructions simultaneously. Requires representations from both `length_constraints/compute_length_representations.py` and `ifeval_experiments/compute_representations.py`.

2. **Evaluate Lowercase + Exclude Word Instructions**
   - Script: `composition/casing_and_exclude_word.py`
   - Config: Available in `config/`
   - Description: Evaluates the model on the "lowercase" and "exclude-word" instructions. Requires representations from `keywords/compute_keywords_representations.py`.

---

Please refer to the individual config files and script arguments for specific parameters to control the behavior of each experiment.
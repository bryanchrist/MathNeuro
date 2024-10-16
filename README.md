# MathNeuro

[Paper](https://arxiv.org/abs/2402.15861)

Codebase for Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes

# Overview 
INSERT ABSTRACT

# License and Intended Use
Our code is released under the GNU GPLv3 license. Our codebase also contains a copy of the [Eleuther AI Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to run all experimental evaluations. 

# Getting Started
After installing PyTorch (follow instructions [here](https://pytorch.org/get-started/locally/)), to install the dependencies for this codebase, you can run: 
```bash
pip install -U -r requirements.txt
```

# Running MathNeuro Pruning Experiments 
To run MathNeuro for identifying and pruning math-specific parameters, you will need to run the MathNeuro.py file. Here is an example of how to conduct the pruning experiment for Llama 3.2 1B IT: 
```bash
python MathNeuro.py --model meta-llama/Llama-3.2-1B-Instruct --save_path /results_path --train_dataset data/gsm8k.csv --eval_datasets race mmlu --calibration_datasets data/race.csv data/mmlu.csv --eval_dataset_subset 200 --calibration_dataset_names Race MMLU --train_lm_eval_task gsm8k_cot --pre_train_eval
```
# Running MathNeuro Scaling Experiments 
To run MathNeuro for identifying and scaling math-specific parameters, you will need to run the MathNeuro.py file specifying your desired scalar. Here is an example of how to conduct the scaling experiment for Llama 3.2 1B IT using a scalar of 1.1: 
```bash
python MathNeuro.py --model meta-llama/Llama-3.2-1B-Instruct --save_path /results_path --train_dataset data/gsm8k.csv --eval_datasets race mmlu --calibration_datasets data/race.csv data/mmlu.csv --eval_dataset_subset 200 --calibration_dataset_names Race MMLU --train_lm_eval_task gsm8k_cot --pre_train_eval --scalar 1.1
```
# Running MathNeuro Ablations
To run the ablations for MathNeuro (Wanda and random identification), you will need to run the MathNeuro_Ablations.py file. Here is an example of how to conduct the pruning experiment ablations for Llama 3.2 1B IT: 
```bash
python MathNeuro_Ablations.py --model meta-llama/Llama-3.2-1B-Instruct --save_path /results_path --train_dataset data/gsm8k.csv --eval_datasets race mmlu --calibration_datasets data/race.csv data/mmlu.csv --eval_dataset_subset 200 --calibration_dataset_names Race MMLU --train_lm_eval_task gsm8k_cot --pre_train_eval
```
# Running LAPE
To run the LAPE comparison method, you will need to run the LAPE.py file. Please note that if you want to run this experiment for a model other than one from the Phi 1.5, Gemma 2, or Llama 3 families, you will need to create a customized implementation of its forward loop. Here is an example of how to conduct the pruning experiment for Llama 3.2 1B IT using LAPE: 
```bash
python LAPE.py --model meta-llama/Llama-3.2-1B-Instruct --save_path /results_path --train_dataset data/gsm8k.csv --eval_datasets race mmlu --calibration_datasets data/race.csv data/mmlu.csv --eval_dataset_subset 200 --calibration_dataset_names Race MMLU --train_lm_eval_task gsm8k_cot --pre_train_eval
```

# Citation
```bash
INSERT
```

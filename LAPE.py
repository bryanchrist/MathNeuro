import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="Huggingface model to train, entered as string; must be Phi 1.5, Gemma 2, or Llama 3 model; otherwise you will need to adjust this code to run with another model", type = str)
parser.add_argument('--eval_datasets', nargs='+', help="dataset(s) to evaluate models on post pruning to evalaute catastrophic forgetting, entered as strings; should be task names from Eleuther AI LM Evaluation Harness", type = str)
parser.add_argument('--train_dataset', help="path to math train dataset; should be a path to a CSV file with question/solution pairs in a columns titled 'question' and 'solution' along with ground-truth answers in a column called 'answer'", type = str)
parser.add_argument('--calibration_datasets', nargs='+', help="path to calibration datasets; should be paths to CSV files with instruction/response pairs in a column titled 'qa'", type = str)
parser.add_argument('--save_path', help="save path for eval results after running Eleuther AI LM Evaluation Harness post pruning", type = str)
parser.add_argument('--num_repeats', help="number of repeats for pruning or scaling experiment", type = int, default = 5)
parser.add_argument('--pre_train_eval', help="bool to indicate if full evaluation on eval and train datasets should be conducted before training", action="store_true")
parser.add_argument('--random_state', help="random state for initial dataset shuffling and creating train/eval split for train dataset", type = int, default = 42)
parser.add_argument('--scalar', help="scale factor for top parameters; default is 0 to run pruning experiments", type = float, default = 0)
parser.add_argument('--eval_dataset_size', help="desired number of samples for task specific eval dataset", type = int, default = None)
parser.add_argument('--eval_dataset_subset', help="desired number of samples for task specific eval dataset if subsetting to reduce run time", type = int, default = 100)
parser.add_argument('--calibration_dataset_names', nargs='+', help="desired name of calibration datasets; should be strings entered in same order as calibration_datasets", type = str)
parser.add_argument('--num_samples', help="desired number of samples for calculating task specific parameters", type = int, default = 100)
parser.add_argument('--train_lm_eval_task', help="if your training dataset is an Eleuther AI LM Evaluation Harness task, specify the associated task for the test set.", type = str, default = None)
parser.add_argument('--ratio', help="for specifying the ratio of top language-specific params for calculation", type = float, default = 0)
args = parser.parse_args()
from transformers import AutoTokenizer, AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np
import re
import lm_eval
import json 
from types import MethodType

output_file = f"{args.save_path}/eval_results/_{args.model}/{args.text_file}"
results_path =  f"{args.save_path}/eval_results/_{args.model}/"
os.makedirs(os.path.dirname(results_path), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
if 'sgsm' not in args.train_dataset:
    train = pd.read_csv(args.train_dataset) # Load SGSM dataset for few-shot prompting
    train = train.sample(frac = 1, random_state = args.random_state)
    if '/' in args.train_dataset:
        dataset_name = args.train_dataset.split('/')[-1]
        dataset_name = dataset_name.split('.csv')[0]
        train.name = dataset_name
    else:
        dataset_name = args.train_dataset.split('.csv')[0]
        train.name = dataset_name

calibration_datasets = []
for dataset in args.calibration_datasets:
    if '/' in dataset:
        dataset_name = dataset.split('/')[-1]
        dataset_name = dataset_name.split('.csv')[0]
        calibration_datasets.append(dataset_name)
    else:
        dataset_name = dataset.split('.csv')[0]
        calibration_datasets.append(dataset_name)

dataset_list = []
for dataset, dataset_name, name in zip(args.calibration_datasets, calibration_datasets, args.calibration_dataset_names):
    # Load the dataset into a DataFrame
    globals()[dataset_name] = pd.read_csv(dataset).sample(frac=1, random_state=args.random_state)  # Shuffle the DataFrame
    
    # Assign a name attribute to the DataFrame
    globals()[dataset_name].name = name
    
    # Append the actual DataFrame object to the list
    dataset_list.append(globals()[dataset_name])

dataset_list.append(train)

if args.pre_train_eval:
    if args.train_lm_eval_task is not None:
        task_manager = lm_eval.tasks.TaskManager()
        #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.train_lm_eval_task,
            task_manager=task_manager,
            log_samples = False, 
            batch_size = 'auto:4',
            limit = args.eval_dataset_subset, 
            random_seed = args.random_state
        )
        results_path = f"{args.save_path}/eval_results/_{args.model}/pre_results_train_task.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
        
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.eval_datasets,
            task_manager=task_manager,
            log_samples = False, 
            batch_size = 'auto:4'
        )
        results_path = f"{args.save_path}/eval_results/_{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
                        
num_repeats = 5
if args.ratio == 0:
    ratios = [.0001, .001, .005, .01, .025, .05, .1, .15]
    
if args.ratio!=0:
    ratios = [args.ratio]
    
for repeat in range(0, num_repeats):
    for ratio in ratios: 
        # Get model config to determine characteristics like number of layers
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
    
        is_llama = 'llama' in args.model.lower()
        is_phi = 'phi' in args.model.lower()
        is_gemma = 'gemma' in args.model.lower()
        num_layers = model.config.num_hidden_layers
        intermediate_size = model.config.intermediate_size if hasattr(model.config, 'intermediate_size') else model.config.hidden_size * 4
    
        # Initialize tensors to store activations and statistics
        sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
        sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
        sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
        sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    
        # Define factory function for activation tracking
        def factory(idx):
            def llama_forward(self, x):
                x1 = self.gate_proj(x)
                x1 = self.act_fn(x1)
                x2 = self.up_proj(x)
                #x2 = self.act_fn(x2)
                activation = x1.float()
                sum1[idx, :] += activation.sum(dim=(0, 1)).to('cuda')
                sum2[idx, :] += activation.pow(2).sum(dim=(0, 1)).to('cuda')
                sum3[idx, :] += activation.pow(3).sum(dim=(0, 1)).to('cuda')
                sum4[idx, :] += activation.pow(4).sum(dim=(0, 1)).to('cuda')
                over_zero[idx, :] += (activation > 0).sum(dim=(0, 1)).to('cuda')
                x = x1 * x2
                x = self.down_proj(x)
                #x = self.act_fn(x)
                return x
        
            def bloom_forward(self, x: torch.Tensor):
                x, _ = self.dense_h_to_4h(x)
                x = self.gelu_impl(x)
                activation = x.float()
                sum1[idx, :] += activation.sum(dim=(0, 1)).to('cuda')
                sum2[idx, :] += activation.pow(2).sum(dim=(0, 1)).to('cuda')
                sum3[idx, :] += activation.pow(3).sum(dim=(0, 1)).to('cuda')
                sum4[idx, :] += activation.pow(4).sum(dim=(0, 1)).to('cuda')
                over_zero[idx, :] += (activation > 0).sum(dim=(0, 1)).to('cuda')
                x, _ = self.dense_4h_to_h(x)
                return x
        
            def phi_forward(self, x):
                x = self.fc1(x)
                x = self.activation_fn(x)
                activation = x.float()
                sum1[idx, :] += activation.sum(dim=(0, 1)).to('cuda')
                sum2[idx, :] += activation.pow(2).sum(dim=(0, 1)).to('cuda')
                sum3[idx, :] += activation.pow(3).sum(dim=(0, 1)).to('cuda')
                sum4[idx, :] += activation.pow(4).sum(dim=(0, 1)).to('cuda')
                over_zero[idx, :] += (activation > 0).sum(dim=(0, 1)).to('cuda')
                x = self.fc2(x)
                return x
        
            def gemma_forward(self, x):
               #gate_up, _ = self.gate_proj(x)
                x1 = self.gate_proj(x)
                x1 = self.act_fn(x1)
                x2 = self.up_proj(x)
                #x2 = self.act_fn(x2)
                activation = x1.float()
                sum1[idx, :] += activation.sum(dim=(0, 1)).to('cuda')
                sum2[idx, :] += activation.pow(2).sum(dim=(0, 1)).to('cuda')
                sum3[idx, :] += activation.pow(3).sum(dim=(0, 1)).to('cuda')
                sum4[idx, :] += activation.pow(4).sum(dim=(0, 1)).to('cuda')
                over_zero[idx, :] += (activation > 0).sum(dim=(0, 1)).to('cuda')
                x = x1 * x2
                x = self.down_proj(x)
                #x = self.act_fn(x)
                return x
            if is_llama:
                return llama_forward
            if is_phi:
                return phi_forward
            if is_gemma:
                return gemma_forward
    
        # Attach forward hooks to MLP layers
        for i in range(num_layers):
            if is_llama:
                obj = model.model.layers[i].mlp
            if is_phi:
                obj = model.model.layers[i].mlp
            if is_gemma:
                obj = model.model.layers[i].mlp
            obj.forward = MethodType(factory(i), obj)
    
        for dataset in dataset_list:
            text_file = f"data/id.{dataset.name}.train.txt"
            sampled_dataset = dataset.sample(n = args.num_samples, replace = True)
            for i in range(0, args.num_samples):
                sample = sampled_dataset.iloc[i]['qa']
                with open(text_file, "w") as f:  # Open the file in append mode ("a")
                    f.write(f"{sample}\n\n")
        
        for dataset in dataset_list:
            input_texts = open(f"data/id.{dataset.name}.train.txt", 'r').readlines()
            input_ids = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=model.config.max_position_embeddings).input_ids.to(model.device)
            l = input_ids.size(0)
            # Forward pass through the model (dummy output to trigger forward hooks)
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
            outputs = model(input_ids)
    
            # Save the output statistics
            output = dict(n = l, sum1=sum1.to('cpu'), sum2=sum2.to('cpu'), sum3=sum3.to('cpu'), sum4=sum4.to('cpu'), over_zero=over_zero.to('cpu'))
            torch.save(output, f"data/activation.{dataset.name}.train.{args.model.split('/')[-1]}")
        
        # Initialize lists to store 'n' and 'over_zero' for each language
        n, over_zero = [], []

        for dataset in dataset_list:
            data = torch.load(f"data/activation.{dataset.name}.train.{args.model.split('/')[-1]}")
            n.append(data['n'])
            over_zero.append(data['over_zero'])
    
        # Convert lists to tensors for easier computation
        n = torch.tensor(n)
        over_zero = torch.stack(over_zero, dim=-1)  # Stack along the last dimension
    
        # Get the dimensions from the data
        num_layers, intermediate_size, lang_num = over_zero.size()
    
        def activation(top_rate = ratio):
            #top_rate = ratio
            filter_rate = 0.95
            activation_bar_ratio = 0.95
    
            # Calculate activation probabilities across languages
            activation_probs = over_zero / n  # layer x intermediate_size x lang_num
            normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
            normed_activation_probs[torch.isnan(normed_activation_probs)] = 0  # Handle NaNs
            log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
            entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)  # Calculate entropy
            largest = False  # Set this based on your preference for topk
    
            if torch.isnan(entropy).sum():
                print(f"Found {torch.isnan(entropy).sum()} NaNs in entropy calculation")
                raise ValueError
    
            # Flatten and filter activation probabilities
            flattened_probs = activation_probs.flatten()
            top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
            print(f"Top probability value for filtering: {top_prob_value}")
    
            # Dismiss neurons if no language has activation over the threshold
            top_position = (activation_probs > top_prob_value).sum(dim=-1)
            entropy[top_position == 0] = -torch.inf if largest else torch.inf
    
            # Select top-k entropy values
            flattened_entropy = entropy.flatten()
            top_entropy_value = round(len(flattened_entropy) * top_rate)
            _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    
            # Map indices back to layer and intermediate positions
            row_index = index // entropy.size(1)
            col_index = index % entropy.size(1)
            selected_probs = activation_probs[row_index, col_index]  # n x lang
    
            print(f"Selected probabilities: {selected_probs.size(0)}, Max activations: {torch.bincount(selected_probs.argmax(dim=-1))}")
    
            # Transpose to get activations per language
            selected_probs = selected_probs.transpose(0, 1)
            activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
            print(f"Activation threshold: {activation_bar}")
    
            # Find activations above the threshold
            lang, indice = torch.where(selected_probs > activation_bar)
    
            # Merge row and column indices
            merged_index = torch.stack((row_index, col_index), dim=-1)
            final_indice = []
    
            # Group by language and sort
            for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
                lang_index = [tuple(row.tolist()) for row in merged_index[index]]
                lang_index.sort()
                layer_index = [[] for _ in range(num_layers)]
                for l, h in lang_index:
                    layer_index[l].append(h)
                for l, h in enumerate(layer_index):
                    layer_index[l] = torch.tensor(h).long()
                final_indice.append(layer_index)
    
            # Save the final indices for the activation mask
            #torch.save(final_indice, f"language_specific_neurons_method/{args.model.split('/')[-1]}")
            return final_indice
        # Run the activation analysis
        activation_mask = activation()
        scalar = args.scalar
        def factory(mask):
            def llama_forward(self, x):
                x1 = self.gate_proj(x)
                x2 = self.up_proj(x)
                x1 = self.act_fn(x1)
                indices = mask.to(x.device)
        
                # Gather the values at the specific indices in dimension 2
                selected_values = x1.index_select(2, indices)
        
                # Multiply the selected values by the scalar
                scaled_values = selected_values * scalar
        
                # Use index_copy_ to place the scaled values back in the activation tensor
                x1.index_copy_(2, indices, scaled_values.to(torch.bfloat16))
                x = x1 * x2
                x = self.down_proj(x)
                return x
        
            def phi_forward(self, x):
                x = self.fc1(x)
                x = self.activation_fn(x)
                indices = mask.to(x.device)
        
                # Gather the values at the specific indices in dimension 2
                selected_values = x.index_select(2, indices)
        
                # Multiply the selected values by the scalar
                scaled_values = selected_values * scalar
                
                # Use index_copy_ to place the scaled values back in the activation tensor
                x.index_copy_(2, indices, scaled_values.to(torch.bfloat16))
        
                x = self.fc2(x)
                return x
        
            def gemma_forward(self, x):
                x1 = self.gate_proj(x)
                x2 = self.up_proj(x)
                x1 = self.act_fn(x1)
                indices = mask.to(x.device)
        
                # Gather the values at the specific indices in dimension 2
                selected_values = x1.index_select(2, indices)
        
                # Multiply the selected values by the scalar
                scaled_values = selected_values * scalar
        
                # Use index_copy_ to place the scaled values back in the activation tensor
                x1.index_copy_(2, indices, scaled_values.to(torch.bfloat16))
                x = x1 * x2
                x = self.down_proj(x)
                return x
            if is_llama:
                return llama_forward
            if is_phi:
                return phi_forward
            if is_gemma:
                return gemma_forward

        for i, layer_mask in enumerate(activation_mask[0]):
            if is_llama:
                obj = model.model.layers[i].mlp
            if is_phi:
                obj = model.model.layers[i].mlp
            if is_gemma:
                obj = model.model.layers[i].mlp
            #obj.forward = MethodType(factory(i), obj)
            obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)
                
        if args.train_lm_eval_task is not None:
            task_manager = lm_eval.tasks.TaskManager()
            #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
            # Setting `task_manager` to the one above is optional and should generally be done
            # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
            # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
            results = lm_eval.simple_evaluate( # call simple_evaluate
                model = 'hf',
                model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                tasks=args.train_lm_eval_task,
                task_manager=task_manager,
                log_samples = False, 
                batch_size = 'auto:4',
                limit = args.eval_dataset_subset, 
                random_seed = args.random_state
            )
            results_path = f"{args.save_path}/eval_results/_{args.model}/run{repeat}_ratio{ratio}_train_task.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as outfile: 
                json.dump(results['results'], outfile)
    
            results = lm_eval.simple_evaluate( # call simple_evaluate
                model = 'hf',
                model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                tasks=args.eval_datasets,
                task_manager=task_manager,
                log_samples = False,
                batch_size = 'auto:4'
            )
            results_path = f"{args.save_path}/eval_results/_{args.model}/run{repeat}_ratio{ratio}.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as outfile: 
                json.dump(results['results'], outfile)
    
        del model
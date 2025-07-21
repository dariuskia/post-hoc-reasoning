# %% Import necessary libraries
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import pickle
import re
import gc
import argparse
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
load_dotenv()
from models import ChatModel
from data_loading import create_cot_dataset, create_dataset
from utils import generate_with_hooks
from transformer_lens.utils import Slice
from matplotlib import pyplot as plt
from nnsight import LanguageModel
from aim import Run, Repo
WORKSPACE_PATH = "/workspace/post-hoc-reasoning"
# %%
def load_dataset(name, train_size=200):
    """Load and split the sports understanding dataset"""
    print("Loading sports understanding dataset...")
    examples = create_dataset(name)
    cot_dataset = create_cot_dataset(name, examples)
    print(f"Loaded {len(cot_dataset)} examples")
    
    # Split into train and test
    train_dataset, test_dataset = train_test_split(cot_dataset, train_size=train_size, random_state=42)
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, test_dataset
# %%
def format_turns_deepseek(item):
    new_item = deepcopy(item[:-1])
    for turn in new_item:
        if turn['role'] == 'model':
            turn['role'] = 'assistant'
    return new_item
def format_turns_gemma(item):
    prompt_replaced = []
    last_msg = None
    for msg in item[:-1]:
        if last_msg is None:
            last_msg = deepcopy(msg)
            continue
        if last_msg['role'] == msg['role']:
            last_msg['content'] += "\n" + msg['content']
        else:
            if last_msg['role'] == 'model':
                last_msg['role'] = 'assistant'
            prompt_replaced.append(last_msg)
            last_msg = deepcopy(msg)
    prompt_replaced.append(last_msg)
    return prompt_replaced
def format_turns(item, model_name):
    FORMAT_REGISTRY = {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": format_turns_deepseek,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": format_turns_deepseek,
        "google/gemma-2-9b-it": format_turns_gemma,
        "google/gemma-3-12b-it": format_turns_gemma,
    }
    return FORMAT_REGISTRY[model_name](item)
# %%
def parse_response(response: str) -> str:
    # TODO: Make more robust; this only works for gemma
    start_answer_string = "the best answer is:"
    if start_answer_string not in response.lower():
        return ""
    answer_part = response.split(start_answer_string)[-1]
    letter_match = re.search(r"\((.)\)", answer_part)
    if not letter_match:
        return ""
    text_answer = (
        answer_part.split(")")[-1]
        .strip()
        .split(", ")[0]
        .lower()
        .replace(".", "")
        .strip()
    )
    return text_answer
# %%
class ReasoningDataset(Dataset):
    def __init__(self, dataset, tokenizer, model_name):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_name = model_name
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]['prompt']
        formatted_prompt = format_turns(item, self.model_name)
        return formatted_prompt, self.dataset[idx]['correct_answer']

def collate_fn_reasoning(batch, tokenizer):
    tokens = tokenizer.apply_chat_template([item[0] for item in batch], return_tensors="pt", padding=True, add_generation_prompt=True)
    return tokens, [item[1] for item in batch]

# %%
class ResidualsDataset(Dataset):
    def __init__(self, evaluation_results, tokenizer):
        self.evaluation_results = evaluation_results
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.evaluation_results['samples'])
    def __getitem__(self, idx):
        return self.evaluation_results['samples'][idx]['question'], self.evaluation_results['samples'][idx]['predicted_answer']

def collate_fn_residuals(batch, tokenizer):
    tokens = tokenizer([item[0] for item in batch], return_tensors="pt", padding=True)
    return tokens, [item[1] for item in batch]

# %%
@torch.inference_mode()
def generate_with_sampling(model, toks, max_new_tokens=1000, temperature=0.6):
    for _ in tqdm(range(max_new_tokens)):
        with model.trace(toks):
            logits = model.lm_head.output.save()
        probs = (logits / temperature).softmax(dim=-1)[:, -1]  # Temperature scaling
        next_tok = torch.multinomial(probs, 1).cpu()
        toks = torch.cat([toks, next_tok], dim=-1)
        if (next_tok == model.tokenizer.eos_token_id).all():
            break
        del logits, probs, next_tok
        torch.cuda.empty_cache()
    return toks

# %%
def generate_with_nnsight(model, toks, max_new_tokens=1000, temperature=0.6):
    with model.generate(toks, max_new_tokens=max_new_tokens, temperature=temperature) as tracer:
        out = model.generator.output.save()
    return out

def generate(model, toks, run, **kwargs):
    if run["hparams"]["model"] == "google/gemma-2-9b-it":
        return generate_with_sampling(model, toks, **kwargs)
    else:
        return generate_with_nnsight(model, toks, **kwargs)

# %%
def eval_model_with_cot(model, reasoning_dataset, gen_params, run, batch_size=8):
    dataloader = DataLoader(reasoning_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn_reasoning, tokenizer=model.tokenizer))
    model.eval()
    correct = 0
    total = 0
    
    # Dictionary to store all logs
    logs = {
        'samples': [],
        'metrics': {}
    }
    
    for i, batch in enumerate(dataloader):
        tokens, answers = batch
        seq_len = tokens.shape[1]
        out = generate(model, tokens, run, **gen_params)
        
        # Get model predictions and responses
        responses = [model.tokenizer.decode(out[j, seq_len:].cpu()) for j in range(len(out))]
        preds = [parse_response(response) for response in responses]
        
        # Log predictions and responses
        for j, (pred, answer, response) in enumerate(zip(preds, answers, responses)):
            sample_log = {
                'sample_id': i * batch_size + j,
                'question': model.tokenizer.decode(tokens[j].cpu()),
                'model_response': response,
                'expected_answer': answer,
                'predicted_answer': pred,
                'correct': pred == answer
            }
            logs['samples'].append(sample_log)
            
            # Log to Aim for tracking
            run.log_info(
                f"Sample {sample_log['sample_id']}:\n"
                f"Question: {sample_log['question']}\n"
                f"Model Response: {sample_log['model_response']}\n"
                f"Expected Answer: {sample_log['expected_answer']}\n"
                f"Predicted Answer: {sample_log['predicted_answer']}\n"
                f"Correct: {sample_log['correct']}\n"
                "----------------------------------------"
            )
            
            if pred == answer:
                correct += 1
                
        total += len(answers)
        # Track running accuracy
        run.track(correct/total, name="accuracy", step=i)
        run.track(correct, name="correct", step=i)
        run.track(total, name="total", step=i)
            
    accuracy = correct / total
    
    # Store final metrics
    logs['metrics'] = {
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': accuracy
    }
    
    # Log final results to Aim
    run.log_info(
        f"\nFinal Results:\n"
        f"Total samples: {total}\n"
        f"Correct predictions: {correct}\n"
        f"Final accuracy: {accuracy:.2%}"
    )
    
    # Save logs to file
    # save_dir = os.path.join(WORKSPACE_PATH, 'artifacts', run.hash)
    with open(os.path.join(WORKSPACE_PATH, 'tmp', 'logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    run.log_artifact(os.path.join(WORKSPACE_PATH, 'tmp', 'logs.json'), name="evaluation_results_test")
    # Delete the local evaluation logs file after uploading to Aim
    os.remove(os.path.join(WORKSPACE_PATH, 'tmp', 'logs.json'))
        
    return accuracy, logs

# %%
@torch.inference_mode()
def collect_residuals(model, residuals_dataset, layers, batch_size=8, max_samples=None): # TODO: dont pass in run, only use run in the main method
    dataloader = DataLoader(residuals_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn_residuals, tokenizer=model.tokenizer))
    model.eval()
    all_residuals = {i: torch.empty((0, model.config.hidden_size)) for i in layers}
    all_preds = []
    for i, batch in enumerate(dataloader):
        if max_samples is not None and i * batch_size >= max_samples:
            break
        tokens, preds = batch
        with model.trace(tokens):
            residuals = {i: model.model.layers[i].output[0][:, -1].save() for i in layers}
        for layer in layers:
            all_residuals[layer] = torch.cat((all_residuals[layer], residuals[layer]), dim=0)
        all_preds.extend(preds)
    return all_residuals, all_preds
# %%
class Probe(nn.Module):
    def __init__(self, input_dim, layer_dims):
        super().__init__()
        layers = []
        for dim in layer_dims[:-1]:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, layer_dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def train_probe(residuals, residuals_test, preds, preds_test, layer_dims, learning_rate, epochs, layer, run, patience=50):
    probe = Probe(residuals[layer].shape[1], layer_dims)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    best_auroc = 0
    patience_counter = 0
    best_state_dict = None

    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        outputs = torch.sigmoid(probe(residuals[layer]))
        assert outputs.shape == preds.shape
        loss = nn.BCELoss()(outputs, preds)
        loss.backward()
        optimizer.step()
        run.track(loss, name=f"loss_{layer}", step=epoch)

        probe.eval()
        with torch.inference_mode():
            preds_probe = torch.sigmoid(probe(residuals_test[layer])).squeeze().cpu().numpy()
        current_auroc = roc_auc_score(preds_test.squeeze().cpu().numpy(), preds_probe)
        run.track(current_auroc, name=f"roc_auc_score_{layer}", step=epoch)

        # Early stopping based on AUROC
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            patience_counter = 0
            best_state_dict = deepcopy(probe.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} with loss {loss.item():.4f}")
            probe.load_state_dict(best_state_dict)
            break
    print(f"Layer {layer} Best AUROC: {best_auroc:.4f}")
    with open(os.path.join(WORKSPACE_PATH, 'tmp', f'probe_{layer}.pkl'), 'wb') as f:
        pickle.dump(probe.state_dict(), f)
    run.log_artifact(os.path.join(WORKSPACE_PATH, 'tmp', f'probe_{layer}.pkl'), name=f"probe_{layer}")
    os.remove(os.path.join(WORKSPACE_PATH, 'tmp', f'probe_{layer}.pkl'))
    return probe

# %%

def main():
    parser = argparse.ArgumentParser(description='Run reasoning probes evaluation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Subparser for eval_model_with_cot
    eval_parser = subparsers.add_parser('eval', help='Evaluate model with chain-of-thought reasoning')
    eval_parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                           help='Model name to evaluate (default: google/gemma-2-9b-it)')
    eval_parser.add_argument('--dataset', type=str, default="logical_deduction",
                           help='Dataset name to use (default: logical_deduction)')
    eval_parser.add_argument('--batch-size', type=int, default=4,
                           help='Batch size for evaluation (default: 4)')
    eval_parser.add_argument('--max-new-tokens', type=int, default=200,
                           help='Maximum new tokens to generate (default: 200)')
    eval_parser.add_argument('--temperature', type=float, default=0.6,
                           help='Temperature for generation (default: 0.6)')
    eval_parser.add_argument('--train-size', type=int, default=200,
                           help='Number of training examples (default: 200)')
    
    residuals_parser = subparsers.add_parser('residuals', help='Compute residuals')
    residuals_parser.add_argument('--run-hash', type=str, required=True,
                             help='Run hash to compute residuals on')
    residuals_parser.add_argument('--layer', type=int, default=None,
                             help='Specific layer to probe (default: None for all layers)')
    residuals_parser.add_argument('--max-samples', type=int, default=None,
                             help='Maximum number of samples to use (default: None for all samples)')
    residuals_parser.add_argument('--train-size', type=int, default=200,
                             help='Number of training examples (default: 200)')
    residuals_parser.add_argument('--batch-size', type=int, default=4,
                             help='Batch size for evaluation (default: 4)')
    # Subparser for train_probes
    train_parser = subparsers.add_parser('train', help='Train reasoning probes')
    train_parser.add_argument('--run-hash', type=str, required=True,
                             help='Run hash to train probes on')
    train_parser.add_argument('--layer-dims', type=str, default='1',
                             help='Comma-separated layer dimensions for probe (default: 1)')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4,
                             help='Learning rate for probe training (default: 1e-4)')
    train_parser.add_argument('--epochs', type=int, default=2000,
                             help='Number of training epochs (default: 2000)')
    train_parser.add_argument('--layer', type=int, default=None,
                             help='Specific layer to probe (default: None for all layers)')
    
    args = parser.parse_args()
    
    if args.command == 'eval':
        # Set up Aim run
        run = Run(repo=WORKSPACE_PATH, experiment="reasoning_probes")
        run.set_artifacts_uri(f"file://{WORKSPACE_PATH}/artifacts")
        
        # Configuration
        MODEL_NAME = args.model
        DATASET_NAME = args.dataset
        BATCH_SIZE = args.batch_size
        gen_params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }
        
        run["hparams"] = {
            "model": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "dataset": f"{DATASET_NAME}",
            **gen_params,
        }
        
        print(f"Loading model: {MODEL_NAME}")
        model = LanguageModel(MODEL_NAME, device_map="auto")
        
        print(f"Loading dataset: {DATASET_NAME}")
        _, test_dataset = load_dataset(DATASET_NAME, train_size=args.train_size)
        reasoning_dataset = ReasoningDataset(test_dataset, model.tokenizer, MODEL_NAME)
        
        print("Starting evaluation...")
        eval_model_with_cot(model, reasoning_dataset, gen_params, run, batch_size=BATCH_SIZE)
        
    elif args.command == 'residuals':
        repo = Repo(WORKSPACE_PATH)
        eval_run = repo.get_run(run_hash=args.run_hash)
        if eval_run is None:
            raise ValueError(f"Run {args.run_hash} not found")

        out_path = eval_run.artifacts['evaluation_results'].download(dest_dir=os.path.join(WORKSPACE_PATH, 'tmp'))
        with open(out_path, 'r') as f:
            evaluation_results = json.load(f)
        os.remove(out_path)
        # out_path = eval_run.artifacts['evaluation_results_test'].download(dest_dir=os.path.join(WORKSPACE_PATH, 'tmp'))
        out_path = os.path.join(WORKSPACE_PATH, "artifacts", args.run_hash, "evaluation_results_test")
        with open(out_path, 'r') as f:
            evaluation_results_test = json.load(f)
        os.remove(out_path)

        run = Run(repo=WORKSPACE_PATH, experiment="residuals")
        run.set_artifacts_uri(f"file://{WORKSPACE_PATH}/artifacts")
        run["hparams"] = {**eval_run['hparams'], "batch_size": args.batch_size, "max_samples": args.max_samples, "layer": args.layer}
        print("Run hash:", run.hash)
        
        model_name: str = eval_run['hparams']['model']
        print(f"Loading model: {model_name}")
        model = LanguageModel(model_name, device_map="auto")

        residuals_dataset = ResidualsDataset(evaluation_results, model.tokenizer)
        residuals_dataset_test = ResidualsDataset(evaluation_results_test, model.tokenizer)
        
        if args.layer is not None:
            layers = [args.layer]
        else:
            # Get number of layers from model config
            num_layers = getattr(model.config, 'num_hidden_layers', None)
            if num_layers is None:
                # Try alternative attribute names
                num_layers = getattr(model.config, 'n_layers', None)
            if num_layers is None:
                # Default fallback
                num_layers = 32  # Common default for many models
            layers = range(num_layers)
        print("Collecting residuals...")
        residuals, preds = collect_residuals(model, residuals_dataset, layers, batch_size=args.batch_size, max_samples=args.max_samples)
        residuals_test, preds_test = collect_residuals(model, residuals_dataset_test, layers, batch_size=args.batch_size, max_samples=args.max_samples)
        print(f"{len(residuals[0])} residuals collected")
        with open(os.path.join(WORKSPACE_PATH, 'tmp', 'residuals.pkl'), 'wb') as f:
            pickle.dump((residuals, preds), f)
        with open(os.path.join(WORKSPACE_PATH, 'tmp', 'residuals_test.pkl'), 'wb') as f:
            pickle.dump((residuals_test, preds_test), f)
        run.log_artifact(os.path.join(WORKSPACE_PATH, 'tmp', 'residuals.pkl'), name="residuals")
        run.log_artifact(os.path.join(WORKSPACE_PATH, 'tmp', 'residuals_test.pkl'), name="residuals_test")
        # Delete the local evaluation logs file after uploading to Aim
        os.remove(os.path.join(WORKSPACE_PATH, 'tmp', 'residuals.pkl'))
        os.remove(os.path.join(WORKSPACE_PATH, 'tmp', 'residuals_test.pkl'))

    elif args.command == 'train':
        repo = Repo(WORKSPACE_PATH)
        residuals_run = repo.get_run(run_hash=args.run_hash)
        if residuals_run is None:
            raise ValueError(f"Run {args.run_hash} not found")

        out_path = residuals_run.artifacts['residuals'].download(dest_dir=os.path.join(WORKSPACE_PATH, 'tmp'))
        with open(out_path, 'rb') as f:
            residuals, preds = pickle.load(f)
        if os.path.exists(out_path):
            os.remove(out_path)
        out_path = residuals_run.artifacts['residuals_test'].download(dest_dir=os.path.join(WORKSPACE_PATH, 'tmp'))
        with open(out_path, 'rb') as f:
            residuals_test, preds_test = pickle.load(f)
        if os.path.exists(out_path):
            os.remove(out_path)

        run = Run(repo=WORKSPACE_PATH, experiment="probes")
        run.set_artifacts_uri(f"file://{WORKSPACE_PATH}/artifacts")
        run["hparams"] = {
            **residuals_run['hparams'],
            "layer_dims": args.layer_dims,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "layer": args.layer,
        }
        print("Run hash:", run.hash)
        
        if args.layer is not None:
            layers = [args.layer]
        else:
            layers = list(residuals.keys())
        layer_dims = [int(dim) for dim in args.layer_dims.split(',')]
        preds = torch.tensor([1 if pred == "yes" else 0 for pred in preds]).unsqueeze(1).float()
        preds_test = torch.tensor([1 if pred == "yes" else 0 for pred in preds_test]).unsqueeze(1).float()
        for layer in layers:
            train_probe(residuals, residuals_test, preds, preds_test, layer_dims, args.learning_rate, args.epochs, layer, run)        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
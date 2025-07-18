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
from aim import Run
WORKSPACE_PATH = "/workspace/post-hoc-reasoning"
# %%
from aim import Repo
repo = Repo(WORKSPACE_PATH)
# %%
repo.get_run(run_hash="b5833934d5074df2bd06f7d5")
# %%
for attr in dir(repo):
    if "run" in attr:
        print(attr)
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
        return formatted_prompt, self.dataset[idx]['correct_answer'], self.dataset[idx]['prompt'][-2]['content']

def collate_fn(batch, tokenizer):
    tokens = tokenizer.apply_chat_template([item[0] for item in batch], return_tensors="pt", padding=True, add_generation_prompt=True)
    return tokens, [item[1] for item in batch], [item[2] for item in batch]

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
    dataloader = DataLoader(reasoning_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, tokenizer=model.tokenizer))
    model.eval()
    correct = 0
    total = 0
    
    # Dictionary to store all logs
    logs = {
        'samples': [],
        'metrics': {}
    }
    
    for i, batch in enumerate(dataloader):
        tokens, answers, prompts = batch
        seq_len = tokens.shape[1]
        out = generate(model, tokens, run, **gen_params)
        
        # Get model predictions and responses
        responses = [model.tokenizer.decode(out[j, seq_len:].cpu()) for j in range(len(out))]
        preds = [parse_response(response) for response in responses]
        
        # Log predictions and responses
        for j, (pred, answer, response) in enumerate(zip(preds, answers, responses)):
            sample_log = {
                'sample_id': i * batch_size + j,
                'question': prompts[j],
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
    with open(os.path.join(WORKSPACE_PATH, 'temp_logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    run.log_artifact(os.path.join(WORKSPACE_PATH, 'temp_logs.json'), name="evaluation_results")
    # Delete the local evaluation logs file after uploading to Aim
    os.remove(os.path.join(WORKSPACE_PATH, 'temp_logs.json'))
        
    return accuracy, logs

def main():
    parser = argparse.ArgumentParser(description='Run reasoning probes evaluation')
    parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                       help='Model name to evaluate (default: google/gemma-2-9b-it)')
    parser.add_argument('--dataset', type=str, default="logical_deduction",
                       help='Dataset name to use (default: logical_deduction)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation (default: 4)')
    parser.add_argument('--max-new-tokens', type=int, default=200,
                       help='Maximum new tokens to generate (default: 200)')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Temperature for generation (default: 0.6)')
    parser.add_argument('--train-size', type=int, default=200,
                       help='Number of training examples (default: 200)')
    
    args = parser.parse_args()
    
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
        "dataset": f"{DATASET_NAME}_test",
        **gen_params,
    }
    
    print(f"Loading model: {MODEL_NAME}")
    model = LanguageModel(MODEL_NAME, device_map="auto")
    
    print(f"Loading dataset: {DATASET_NAME}")
    train_dataset, test_dataset = load_dataset(DATASET_NAME, train_size=args.train_size)
    reasoning_dataset = ReasoningDataset(test_dataset, model.tokenizer, MODEL_NAME)
    
    print("Starting evaluation...")
    eval_model_with_cot(model, reasoning_dataset, gen_params, run, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()
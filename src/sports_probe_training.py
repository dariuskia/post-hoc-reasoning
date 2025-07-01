# %% Import necessary libraries
import json
import numpy as np
import pandas as pd
import torch
import os
import re
import gc
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()
from models import ChatModel
from data_loading import create_cot_dataset, create_dataset
from utils import generate_with_hooks

THINKING = True
# %% Load the model
model = ChatModel("google/gemma-2-9b-it", device='cuda', n_devices=2, cache_dir=os.environ['HF_HOME'])
print(f"Model loaded: {model.model_name}")
print(f"Number of layers: {model.cfg.n_layers}")
print(f"Model dimension: {model.cfg.d_model}")

# %%
import importlib
importlib.reload(data_loading)
from data_loading import create_cot_dataset, create_dataset
# %% Load sports understanding dataset
print("Loading sports understanding dataset...")
examples = create_dataset("sports_understanding")
cot_dataset = create_cot_dataset("sports_understanding", examples, thinking=THINKING)
print(f"Loaded {len(cot_dataset)} examples")
# %%
# Split into train and test
train_size = 100
train_dataset, test_dataset = train_test_split(cot_dataset, train_size=train_size, random_state=42)
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
# %%
print("\n".join([turn['content'] for turn in train_dataset[0]['prompt']]))

# %% Function to extract residual activations
def get_resid_activations(prompts, model, batch_size=1):
    """Extract residual activations from all layers for given prompts in batches"""
    layers = list(range(model.cfg.n_layers))
    all_activations = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokens = model.to_tokens(batch_prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, pos_slice=-1)
        
        batch_activations = torch.zeros((len(batch_prompts), model.cfg.n_layers, model.cfg.d_model))
        
        for layer in layers:
            layer_activations = cache["resid_post", layer]
            layer_activations = layer_activations.squeeze().detach().cpu()
            batch_activations[:, layer, :] = layer_activations
            del layer_activations
            torch.cuda.empty_cache()
            gc.collect()
            
        all_activations.append(batch_activations)
        
        # Clear cache after each batch to save memory
        del cache, tokens
        torch.cuda.empty_cache()
        gc.collect()
        
    # Concatenate all batches
    activations = np.concatenate(all_activations, axis=0)
    return activations


# %%
# Helper functions
def parse_response(response: str, thinking: bool = True) -> Tuple[str, str]:
    # TODO: Make more robust; this only works for gemma
    response = (
        response.strip()
        .replace("<eos>", "")
        .replace("<pad>", "")
        .replace("<end_of_turn>", "")
        .strip()
    )
    if thinking:
        start_answer_string = "the best answer is:"
        if start_answer_string not in response.lower():
            return "", ""
        answer_part = response.split(start_answer_string)[-1]
        letter_match = re.search(r"\((.)\)", answer_part)
        if not letter_match:
            return "", ""
        letter = letter_match.group(1)
        text_answer = (
            answer_part.split(")")[-1]
            .strip()
            .split(", ")[0]
            .lower()
            .replace(".", "")
            .strip()
        )
    else:
        letter = "A" if "(A)" in response else "B"
        text_answer = "yes" if "yes" in response.lower() else "no"
    return letter, text_answer
# %%
def format_prompt(model, prompt):
    prompt_replaced = []
    last_msg = None
    for msg in prompt:
        if last_msg is None:
            last_msg = deepcopy(msg)
            continue
        if last_msg['role'] == msg['role']:
            last_msg['content'] += "\n" + msg['content']
        else:
            prompt_replaced.append(last_msg)
            last_msg = deepcopy(msg)
    prompt_replaced.append(last_msg)
    return model.apply_chat_template(prompt_replaced)
# %% Function to generate model predictions
def generate_predictions(dataset, model, temperature=0.7, max_new_tokens=100):
    """Generate predictions for given prompts"""
    prompts = [format_prompt(model, item['prompt']) for item in dataset]
    predictions = []
    
    # Process one prompt at a time to minimize memory usage
    for i, prompt in enumerate(prompts):
        # Generate tokens and prediction for single prompt
        tokens = model.to_tokens([prompt], prepend_bos=True)
        generation = model.generate(
            tokens[:, :-2],
            max_new_tokens=max_new_tokens,
            temperature=temperature, 
            do_sample=True
        )
        
        # Extract response
        response = model.tokenizer.decode(generation[0][tokens.shape[1]-2:])
        letter, text_answer = parse_response(response, thinking=THINKING)
        print(f"Response {i}: {response}")
        
        predictions.append({
            'prompt': prompt,
            'response': response,
            'pred_letter': letter,
            'pred_answer': text_answer,
            'correct_letter': dataset[i]['correct_letter'],
            'correct_answer': dataset[i]['correct_answer']
        })
        
        # Clean up tensors
        del tokens, generation
        torch.cuda.empty_cache()
        gc.collect()
        
    return predictions

# %%
# Clear memory before processing
torch.cuda.empty_cache()
gc.collect()
# %%
# Generate predictions first (needed for probe training)
print("Generating predictions...")
train_predictions = generate_predictions(train_dataset, model)
print(f"Generated predictions for {len(train_predictions)} examples")
# Save predictions to file
print("Saving predictions...")
torch.save(train_predictions, f"train_predictions_{'cot' if THINKING else 'noncot'}.pt")
# %%
letter_accuracy = np.mean([pred['pred_letter'] == pred['correct_letter'] for pred in train_predictions])
answer_accuracy = np.mean([pred['pred_answer'] == pred['correct_answer'] for pred in train_predictions])
print(f"Letter Accuracy: {letter_accuracy:.2%}")
print(f"Answer Accuracy: {answer_accuracy:.2%}")
# %%
# Generate test predictions
print("Generating predictions...")
test_predictions = generate_predictions(test_dataset, model)
print(f"Generated predictions for {len(test_predictions)} examples")
# Save predictions to file
print("Saving predictions...")
torch.save(test_predictions, f"test_predictions_{'cot' if THINKING else 'noncot'}.pt")
# %%
letter_accuracy = np.mean([pred['pred_letter'] == pred['correct_letter'] for pred in test_predictions])
answer_accuracy = np.mean([pred['pred_answer'] == pred['correct_answer'] for pred in test_predictions])
print(f"Letter Accuracy: {letter_accuracy:.2%}")
print(f"Answer Accuracy: {answer_accuracy:.2%}")

# %% Function to extract activations for a single layer (memory efficient)
@torch.inference_mode()
def get_layer_activations(prompts, model, layer, batch_size=1):
    """Extract activations for a single layer to save memory"""
    all_activations = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokens = model.to_tokens(batch_prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, pos_slice=-1)
        
        # Extract only the specified layer
        layer_activations = cache["resid_post", layer]
        layer_activations = layer_activations.squeeze().detach().cpu()
        all_activations.append(layer_activations)
        
        # Clear cache immediately
        del cache, tokens, layer_activations
        torch.cuda.empty_cache()
        gc.collect()
        
    # Concatenate all batches
    activations = torch.stack(all_activations, dim=0)
    print(f"{activations.shape=}")
    return activations
    
def prepare_probe_data_layer(results, dataset, model, layer, batch_size=1):
    """Prepare data for training a probe on a specific layer - memory efficient"""
    data = []
    prompts = [format_prompt(model, item['prompt']) for item in dataset]
    # Get activations for this layer only
    layer_activations = get_layer_activations(prompts, model, layer, batch_size)
    
    for idx, result in enumerate(results):
        if result['pred_answer'] == result['correct_answer']:
            activation = layer_activations[idx]
            data.append(activation.tolist() + [result['pred_answer']])
    
    # Clear activations to save memory
    del layer_activations
    torch.cuda.empty_cache()
    gc.collect()
    
    df = pd.DataFrame(
        data, 
        columns=[f"ac{i}" for i in range(model.cfg.d_model)] + ["pred"]
    )
    df = df[df["pred"].isin(["yes", "no"])]
    print(f"{len(df)=}")
    return df

# Function to train a probe
def train_probe(train_data):
    """Train a logistic regression probe"""
    X = train_data[[col for col in train_data.columns if col.startswith("ac")]]
    y = train_data["pred"]
    return LogisticRegression(random_state=0).fit(X, y)

# Function to evaluate a probe
def evaluate_probe(clf, test_data):
    """Evaluate probe using AUROC"""
    X = test_data[[col for col in test_data.columns if col.startswith("ac")]]
    y = test_data["pred"]
    y = y.apply(lambda x: 1 if x == "yes" else 0)
    try:
        return roc_auc_score(y, clf.predict_proba(X)[:, 1])
    except ValueError:
        return 0

# Function to extract coefficient vector
def extract_coef_vector(clf):
    """Extract coefficient vector from trained probe"""
    return clf.coef_[0]

print("Training probes for all layers...")
layers = list(range(model.cfg.n_layers))
all_probes = []
all_coef_vectors = []
auc_scores = []

for layer in layers:
    print(f"\nTraining probe for layer {layer}...")
    
    # Prepare data for this layer
    train_data = prepare_probe_data_layer(train_predictions, train_dataset, model, layer)
    test_data = prepare_probe_data_layer(test_predictions, test_dataset, model, layer)
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    if len(train_data) == 0 or len(test_data) == 0:
        print(f"  Skipping layer {layer} - insufficient data")
        all_probes.append(None)
        all_coef_vectors.append(None)
        auc_scores.append(0)
        continue
    
    # Train probe
    clf = train_probe(train_data)
    
    # Evaluate probe
    auc_score = evaluate_probe(clf, test_data)
    auc_scores.append(auc_score)
    
    # Extract coefficient vector
    coef_vector = extract_coef_vector(clf)
    
    print(f"  AUROC: {auc_score:.4f}")
    
    all_probes.append(clf)
    all_coef_vectors.append(coef_vector)

print(f"\nProbe training completed for {len(layers)} layers")

# %% Analyze results
print("\n=== PROBE TRAINING RESULTS ===")
print(f"Best layer: {layers[np.argmax(auc_scores)]}")
print(f"Best AUROC: {np.max(auc_scores):.4f}")
print(f"Average AUROC: {np.mean(auc_scores):.4f}")
print(f"Std AUROC: {np.std(auc_scores):.4f}")

# Show top 5 layers
print("\nTop 5 layers by AUROC:")
top_indices = np.argsort(auc_scores)[::-1][:5]
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Layer {layers[idx]}: AUROC = {auc_scores[idx]:.4f}")

# Show bottom 5 layers
print("\nBottom 5 layers by AUROC:")
bottom_indices = np.argsort(auc_scores)[:5]
for i, idx in enumerate(bottom_indices):
    print(f"  {i+1}. Layer {layers[idx]}: AUROC = {auc_scores[idx]:.4f}")

# %% Visualize AUROC scores across layers
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(layers, auc_scores, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Layer')
plt.ylabel('AUROC Score')
plt.title('Probe Performance Across Layers - Sports Understanding Task')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Highlight best layer
best_layer = layers[np.argmax(auc_scores)]
best_auc = np.max(auc_scores)
plt.plot(best_layer, best_auc, 'ro', markersize=10, label=f'Best: Layer {best_layer} (AUROC={best_auc:.3f})')
plt.legend()

plt.tight_layout()
plt.show()

# %% Save results
results = {
    'layers': layers,
    'auc_scores': auc_scores,
    'all_probes': all_probes,
    'all_coef_vectors': all_coef_vectors,
    'best_layer': layers[np.argmax(auc_scores)],
    'best_auc': np.max(auc_scores),
    'train_size': len(train_dataset),
    'test_size': len(test_dataset)
}

print("Results summary:")
print(f"  Best layer: {results['best_layer']}")
print(f"  Best AUROC: {results['best_auc']:.4f}")
print(f"  Train samples: {results['train_size']}")
print(f"  Test samples: {results['test_size']}")
print(f"  Total probes trained: {len([p for p in all_probes if p is not None])}")

# You can save the results to a file if needed
# import pickle
# with open('sports_understanding_probes.pkl', 'wb') as f:
#     pickle.dump(results, f)

# %% Optional: Test a specific probe
# Test the best probe on a few examples
best_layer_idx = np.argmax(auc_scores)
best_probe = all_probes[best_layer_idx]

if best_probe is not None:
    print(f"\nTesting best probe (layer {best_layer_idx}) on a few examples:")
    
    # Get a few test examples
    test_data = prepare_probe_data_layer(test_predictions, test_dataset, model, best_layer_idx)
    if len(test_data) > 0:
        X_test = test_data[[col for col in test_data.columns if col.startswith("ac")]]
        y_test = test_data["pred"]
        
        # Make predictions
        predictions = best_probe.predict(X_test)
        probabilities = best_probe.predict_proba(X_test)
        
        print(f"Sample predictions (first 5):")
        for i in range(min(5, len(predictions))):
            print(f"  True: {y_test.iloc[i]}, Pred: {predictions[i]}, Prob: {probabilities[i][1]:.3f}") 
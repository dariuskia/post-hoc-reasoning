# %% Import necessary libraries
import json
import numpy as np
import pandas as pd
import torch
import os
import gc
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

# %% Load the model
model = ChatModel("google/gemma-2-9b-it", device='cuda', cache_dir=os.environ['HF_HOME'])
print(f"Model loaded: {model.model_name}")
print(f"Number of layers: {model.cfg.n_layers}")
print(f"Model dimension: {model.cfg.d_model}")

# %% Load sports understanding dataset
print("Loading sports understanding dataset...")
examples = create_dataset("sports_understanding")
cot_dataset = create_cot_dataset("sports_understanding", examples)
print(f"Loaded {len(cot_dataset)} examples")

# Split into train and test
train_size = 100
train_dataset, test_dataset = train_test_split(cot_dataset, train_size=train_size, random_state=42)
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
# %%
print("\n".join([turn['content'] for turn in train_dataset[0]['prompt']]))

# %% Function to extract residual activations
def get_resid_activations(prompts, model, batch_size=2):
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
        
    # Concatenate all batches
    activations = np.concatenate(all_activations, axis=0)
    return activations

# %% Function to generate model predictions
def generate_predictions(prompts, model, temperature=0.7, max_new_tokens=100):
    """Generate predictions for given prompts"""
    tokens = model.to_tokens(prompts, prepend_bos=True)
    generations = model.generate(
        tokens, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature,
        do_sample=True
    )
    
    # Parse responses to extract predictions
    predictions = []
    for i, (prompt, generation) in enumerate(zip(prompts, generations)):
        response = generation[len(prompt):]
        # Simple parsing - look for (A) or (B) in the response
        if "(A)" in response:
            pred_letter = "A"
        elif "(B)" in response:
            pred_letter = "B"
        else:
            pred_letter = "Unknown"
        
        # Map to yes/no based on the prompt structure
        if "(A) Yes" in prompt or "(A) No" in prompt:
            pred_answer = "yes" if pred_letter == "A" else "no"
        else:
            pred_answer = "no" if pred_letter == "A" else "yes"
        
        predictions.append({
            'prompt': prompt,
            'response': response,
            'pred_letter': pred_letter,
            'pred_answer': pred_answer,
            'correct_letter': train_dataset[i]['correct_letter'],
            'correct_answer': train_dataset[i]['correct_answer']
        })
    
    return predictions

# %%
def format_prompt(model, prompt):
    prompt_replaced = []
    last_msg = None
    for msg in prompt:
        if msg['role'] == 'model':
            msg['role'] = 'assistant'
        if last_msg and last_msg['role'] == msg['role']:
            last_msg['content'] += "\n" + msg['content']
        elif not last_msg:
            last_msg = msg
        else:
            prompt_replaced.append(last_msg)
            last_msg = msg
    prompt_replaced.append(last_msg)
    return model.apply_chat_template(prompt_replaced)
# %%
print("Processing training data...")
train_prompts = [item['prompt'] for item in train_dataset]
train_prompts = [format_prompt(model, prompt) for prompt in train_prompts]

# %%
batch_prompts = train_prompts[6:8]
tokens = model.to_tokens(batch_prompts, prepend_bos=True)
_, cache = model.run_with_cache(tokens, pos_slice=-1)
del _, cache, batch_prompts, tokens
torch.cuda.empty_cache()
gc.collect()

# %%
batch_activations = torch.zeros((len(batch_prompts), model.cfg.n_layers, model.cfg.d_model))

layers = list(range(model.cfg.n_layers))
for layer in layers:
    layer_activations = cache["resid_post", layer]
    layer_activations = layer_activations.squeeze().detach().cpu()
    batch_activations[:, layer, :] = layer_activations
    del layer_activations
    torch.cuda.empty_cache()
    gc.collect()
# %%
# Get activations
print("Extracting activations...")
train_activations = get_resid_activations(train_prompts, model)
print(f"Train activations shape: {train_activations.shape}")
# %%
# Generate predictions
print("Generating predictions...")
train_predictions = generate_predictions(train_prompts, model)
print(f"Generated predictions for {len(train_predictions)} examples")

# %% Process test data
print("Processing test data...")
test_prompts = [item['prompt'] for item in test_dataset]

# Get activations
print("Extracting activations...")
test_activations = get_resid_activations(test_prompts, model)
print(f"Test activations shape: {test_activations.shape}")

# Generate predictions
print("Generating predictions...")
test_predictions = generate_predictions(test_prompts, model)
print(f"Generated predictions for {len(test_predictions)} examples")

# %% Function to prepare data for probe training
def prepare_probe_data(results, activations, layer):
    """Prepare data for training a probe on a specific layer"""
    data = []
    for idx, result in enumerate(results):
        if result['pred_answer'] == result['correct_answer']:
            activation = activations[idx][layer]
            data.append(activation.tolist() + [result['pred_answer']])
    
    df = pd.DataFrame(
        data, 
        columns=[f"ac{i}" for i in range(model.cfg.d_model)] + ["pred"]
    )
    df = df[df["pred"].isin(["yes", "no"])]
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

# %% Train probes for all layers
print("Training probes for all layers...")
layers = list(range(model.cfg.n_layers))
all_probes = []
all_coef_vectors = []
auc_scores = []

for layer in layers:
    print(f"\nTraining probe for layer {layer}...")
    
    # Prepare data for this layer
    train_data = prepare_probe_data(train_predictions, train_activations, layer)
    test_data = prepare_probe_data(test_predictions, test_activations, layer)
    
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
    test_data = prepare_probe_data(test_predictions, test_activations, best_layer_idx)
    if len(test_data) > 0:
        X_test = test_data[[col for col in test_data.columns if col.startswith("ac")]]
        y_test = test_data["pred"]
        
        # Make predictions
        predictions = best_probe.predict(X_test)
        probabilities = best_probe.predict_proba(X_test)
        
        print(f"Sample predictions (first 5):")
        for i in range(min(5, len(predictions))):
            print(f"  True: {y_test.iloc[i]}, Pred: {predictions[i]}, Prob: {probabilities[i][1]:.3f}") 
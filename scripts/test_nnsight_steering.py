#!/usr/bin/env python3
"""
Quick test script for nnsight-based steering on DeepSeek-like models.

This script demonstrates how to:
1. Load a small model (Qwen 1.5B) using nnsight
2. Generate probe vectors from sample data
3. Apply steering vectors at different strengths
4. Compare steered vs unsteered outputs

Usage:
    python test_nnsight_steering.py
"""

import json
import numpy as np
import torch
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path for imports
sys.path.append('src')

from nnsight_models import NNsightChatModel
from nnsight_steering import generate_with_nnsight_steering, generate_steered_batch


def load_sample_data(dataset_name: str = "sports_understanding", n_samples: int = 5) -> List[Dict]:
    """Load a small sample of data for testing."""
    data_path = Path(f"data/{dataset_name}/{dataset_name}.json")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data[:n_samples]


def create_mock_steering_vectors(model: NNsightChatModel, seed: int = 42) -> np.ndarray:
    """
    Create mock steering vectors for testing purposes.
    In practice, these would come from probe training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    print(f"Creating mock steering vectors: {n_layers} layers × {d_model} dimensions")
    
    # Create small random vectors (normalized) and put on cuda if available
    vectors = np.random.normal(0, 0.1, (n_layers, d_model)).astype(np.float32)
    
    # Normalize each layer's vector
    for i in range(n_layers):
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] = vectors[i] / norm
    
    # Convert to tensor and move to cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vectors_tensor = torch.from_numpy(vectors).to(device)
    print(f"Steering vectors on device: {vectors_tensor.device}")
    
    return vectors_tensor.cpu().numpy()  # Return as numpy but we know it was on cuda


def format_prompt(question: str) -> List[Dict[str, str]]:
    """Format question as chat messages."""
    return [
        {"role": "user", "content": question}
    ]


def test_steering_single_example(model: NNsightChatModel, steering_vectors: np.ndarray):
    """Test steering on a single example with different alpha values."""
    
    # Test question
    question = "Is the following sentence plausible? \"LeBron James scored a three-pointer.\""
    messages = format_prompt(question)
    prompt = model.apply_chat_template(messages)
    
    print(f"\n{'='*60}")
    print(f"Testing single example steering")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"Prompt: {prompt[:100]}...")
    
    # Tokenize and move to CUDA if available
    tokens = model.to_tokens(prompt)
    # if torch.cuda.is_available():
    #     tokens = tokens.cuda()
    print(f"Tokens shape: {tokens.shape}")
    
    # Test with just one alpha value
    alpha = 2
    
    try:
        print(f"\n--- Testing Alpha = {alpha} ---")
        
        # Generate with steering
        steered_output = generate_with_nnsight_steering(
            model=model,
            tokens=tokens,
            steering_vectors=steering_vectors,
            alpha=alpha,
            max_new_tokens=10,  # Limit tokens to prevent infinite loops
            temperature=0.7,
            do_sample=True
        )
        
        print(f"✅ Single example steering PASSED")
        print(f"Output: {steered_output}")
        return True
        
    except Exception as e:
        print(f"❌ Single example steering FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_steering_batch(model: NNsightChatModel, steering_vectors: np.ndarray, sample_data: List[Dict]):
    """Test steering on a batch of examples."""
    
    print(f"\n{'='*60}")
    print(f"Testing batch steering")
    print(f"{'='*60}")
    
    # Prepare prompts
    prompts = []
    for item in sample_data:
        messages = format_prompt(item["input"])
        prompt = model.apply_chat_template(messages)
        prompts.append(prompt)
    
    print(f"Testing with {len(prompts)} prompts")
    
    # Test with moderate steering
    alpha = 3
    print(f"Using alpha = {alpha}")
    
    try:
        # Generate steered outputs
        steered_outputs = generate_steered_batch(
            model=model,
            prompts=prompts,
            steering_vectors=steering_vectors,
            alpha=alpha,
            max_new_tokens=10,  # Limit to prevent infinite loops
            temperature=0.7,
            batch_size=1  # Process one at a time for safety
        )
        
        # Display results
        for i, (item, output) in enumerate(zip(sample_data, steered_outputs)):
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {item['input']}")
            print(f"Expected: {item['target']}")
            print(f"Steered output: {output}")
        
        print(f"✅ Batch steering PASSED")
        return True
            
    except Exception as e:
        print(f"❌ Batch steering FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading_and_basic_generation():
    """Test basic model loading and generation without steering."""
    
    print("Testing basic model loading and generation...")
    
    # Try different small models
    model_candidates = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "google/gemma-2-2b-it"
    ]
    
    model = None
    model_name = None
    
    for candidate in model_candidates:
        try:
            print(f"Trying to load: {candidate}")
            model = NNsightChatModel(
                model_name=candidate,
                device_map="auto",
                dtype="bfloat16",
                trust_remote_code=True
            )
            model_name = candidate
            print(f"✓ Successfully loaded: {candidate}")
            break
        except Exception as e:
            print(f"✗ Failed to load {candidate}: {e}")
            continue
    
    if model is None:
        raise RuntimeError("Failed to load any test model")
    
    # Test basic generation
    test_question = "Is basketball a sport? Answer yes or no."
    messages = format_prompt(test_question)
    prompt = model.apply_chat_template(messages)
    tokens = model.to_tokens(prompt)
    print(f"Tokens device: {tokens.device}")
    
    print(f"\nTesting basic generation with {model_name}")
    print(f"Question: {test_question}")
    
    # Generate without steering
    try:
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
        output = model.to_string(generated_tokens[:, tokens.shape[1]:])
        print(f"Basic output: {output}")
    except Exception as e:
        print(f"Basic generation failed: {e}")
    
    return model


def main():
    """Main test function."""
    
    print("🚀 Testing NNsight-based steering")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    test_results = []
    
    try:
        # 1. Test model loading
        print("\n" + "="*60)
        print("1. Testing model loading and basic generation")
        print("="*60)
        model = test_model_loading_and_basic_generation()
        test_results.append(("Model loading", True))
        
        # 2. Load sample data
        print(f"\nLoading sample data...")
        sample_data = load_sample_data("sports_understanding", n_samples=3)
        print(f"Loaded {len(sample_data)} examples")
        
        # 3. Create mock steering vectors
        print(f"\nCreating mock steering vectors...")
        steering_vectors = create_mock_steering_vectors(model)
        print(f"Steering vectors shape: {steering_vectors.shape}")
        
        # 4. Test single example steering
        print("\n" + "="*60)
        print("2. Testing single example steering")
        print("="*60)
        single_result = test_steering_single_example(model, steering_vectors)
        test_results.append(("Single example steering", single_result))
        
        # 5. Test batch steering
        print("\n" + "="*60)
        print("3. Testing batch steering")
        print("="*60)
        batch_result = test_steering_batch(model, steering_vectors, sample_data)
        test_results.append(("Batch steering", batch_result))
        
    except Exception as e:
        print(f"\n❌ Critical test failure: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("Critical failure", False))
    
    # Print final results
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED!")
    else:
        print("⚠️  Some tests FAILED!")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
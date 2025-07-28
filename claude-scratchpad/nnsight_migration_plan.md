# NNsight Migration Plan: Expanding Model Support

## Executive Summary

This plan outlines migrating from transformer_lens to nnsight for the main experiment framework to:
1. **Support any HuggingFace model** (including DeepSeek, newer models)
2. **Improve intervention system** with cleaner, more intuitive API
3. **Maintain backward compatibility** with existing experiments
4. **Future-proof** the framework for new model architectures

## Current Limitations Analysis

### Model Compatibility Issues
- **transformer_lens limitations**: Only supports explicitly implemented models
- **DeepSeek support**: Current ChatModel fails with DeepSeek models due to chat template requirements and lack of explicit support
- **Future-proofing**: Need to support newer models without waiting for transformer_lens updates
- **API complexity**: Hook system is complex and requires deep understanding of transformer_lens internals

## Phase 1: Create NNsight-Based Model Wrapper

### 1.1 New NNsightChatModel Class
**File**: `src/nnsight_models.py`

```python
from nnsight import LanguageModel
from typing import Dict, List, Optional
import torch

class NNsightChatModel:
    def __init__(self, model_name: str, device_map="auto", dtype="bfloat16"):
        self.model_name = model_name
        self.model = LanguageModel(model_name, device_map=device_map, torch_dtype=dtype)
        
        # Model-specific formatting (from reasoning_probes.py)
        self.format_registry = {
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": self._format_turns_deepseek,
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": self._format_turns_deepseek,
            "google/gemma-2-9b-it": self._format_turns_gemma,
            # Add more as needed
        }
    
    def _format_turns_deepseek(self, messages):
        # Convert "model" -> "assistant" for DeepSeek
        formatted = []
        for msg in messages:
            if msg["role"] == "model":
                msg["role"] = "assistant"
            formatted.append(msg)
        return formatted
    
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        # Apply model-specific formatting first
        if self.model_name in self.format_registry:
            messages = self.format_registry[self.model_name](messages)
        
        return self.model.tokenizer.apply_chat_template(messages, tokenize=False)
    
    def to_tokens(self, text: str, **kwargs) -> torch.Tensor:
        return self.model.tokenizer(text, return_tensors="pt", **kwargs)["input_ids"]
    
    def to_string(self, tokens: torch.Tensor) -> str:
        return self.model.tokenizer.decode(tokens.squeeze(), skip_special_tokens=False)
```

**Key Features**:
- **Model-specific chat formatting** (DeepSeek, Gemma, etc.)
- **Compatible API** with existing ChatModel
- **Broader model support** via nnsight's HuggingFace integration

## Phase 2: NNsight-Based Activation Extraction

### 2.1 Batch Activation Extraction
**File**: `src/nnsight_utils.py`

```python
import torch
import numpy as np
from typing import List, Dict

def batch_get_resid_activations(model: NNsightChatModel, prompts: List[str]):
    """Extract residual activations using nnsight (adapted from reasoning_probes.py)"""
    tokens = model.model.tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
    
    # Get all layer indices
    num_layers = model.model.config.num_hidden_layers or model.model.config.n_layers
    layers = list(range(num_layers))
    
    with model.model.trace(tokens):
        # Extract residual activations from all layers at once
        residuals = {
            layer: model.model.model.layers[layer].output[0][:, -1].save()  # Final position
            for layer in layers
        }
    
    # Convert to numpy format matching original
    activations = np.zeros((len(prompts), num_layers, model.model.config.hidden_size))
    for layer in layers:
        activations[:, layer, :] = residuals[layer].detach().cpu().numpy()
    
    return activations
```

**Improvements**:
- **All layers at once**: No sequential processing like current implementation
- **Clean API**: Uses nnsight's `trace()` context manager
- **Memory efficient**: Automatic cleanup with context management

## Phase 3: NNsight-Based Steering Implementation

### 3.1 Steering with NNsight Interventions
**File**: `src/nnsight_steering.py`

```python
import torch
from typing import List, Optional
import numpy as np

def generate_with_nnsight_steering(
    model: NNsightChatModel,
    tokens: torch.Tensor,
    steering_vectors: np.ndarray,
    alpha: float = 1.0,
    instruction_pos: int = 0,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    layers: Optional[List[int]] = None,
) -> str:
    """
    Generate with steering using nnsight interventions.
    Uses nnsight's cleaner intervention API for position-based steering.
    """
    if layers is None:
        layers = list(range(len(steering_vectors)))
    
    # Convert steering vectors to tensors
    steering_tensors = {
        layer: torch.tensor(steering_vectors[layer], device=model.model.device)
        for layer in layers
    }
    
    with model.model.generate(
        tokens, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature,
        do_sample=True
    ) as generator:
        
        # Apply steering interventions during generation
        for layer in layers:
            # Get the layer's residual output
            residual = model.model.model.layers[layer].output[0]
            
            # Apply steering only to positions after instruction_pos
            def steer_residual(residual_tensor, layer_idx=layer):
                batch_size, seq_len, hidden_size = residual_tensor.shape
                
                # Only modify positions beyond instruction_pos (during generation)
                if seq_len > instruction_pos:
                    steering_vector = steering_tensors[layer_idx]
                    # Add steering to positions after instruction
                    residual_tensor[:, instruction_pos:, :] += alpha * steering_vector
                
                return residual_tensor
            
            # Apply the intervention
            residual.intervene(steer_residual)
        
        # Get the generated output
        output = generator.output.save()
    
    return model.to_string(output[0])
```

**Key Improvements**:
- **Cleaner intervention API**: Uses nnsight's built-in intervention system
- **Simpler implementation**: No manual hook management required
- **Better error handling**: More intuitive debugging and error messages

## Phase 4: Updated Experiment Runner

### 4.1 Modified Experiment Runner Class
**File**: `src/nnsight_experiment_runner.py`

```python
from nnsight_models import NNsightChatModel
from nnsight_steering import generate_with_nnsight_steering
from nnsight_utils import batch_get_resid_activations

class NNsightExperimentRunner:
    def __init__(self, run_config):
        self.run_config = run_config
        # Rest of initialization same as before
    
    def batch_get_resid_activations(self, prompts: List[str], model: NNsightChatModel):
        """Use nnsight-based activation extraction"""
        return batch_get_resid_activations(model, prompts)
    
    def generate_steered_examples(
        self,
        model: NNsightChatModel,
        test_data: List[Dict],
        all_coef_vectors: List,
        layers: List[int],
        alpha: float,
        config,
    ):
        """Generate steered examples using nnsight"""
        steered_results = []
        
        for example in test_data:
            prompt = example["prompt"]
            tokens = model.to_tokens(prompt)
            instruction_pos = tokens.size(1)  # End of prompt
            
            # Generate with steering
            generation = generate_with_nnsight_steering(
                model=model,
                tokens=tokens,
                steering_vectors=np.array(all_coef_vectors),
                alpha=alpha,
                instruction_pos=instruction_pos,  # Only steer during generation
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                layers=layers,
            )
            
            # Parse and evaluate results (same logic as before)
            new_letter, new_answer = self.parse_response(generation)
            # ... rest of evaluation logic
        
        return steered_results
```

**Key Changes**:
- **Same interface**: Drop-in replacement for existing experiment runner
- **Cleaner steering implementation**: Uses nnsight's intervention system
- **Broader compatibility**: Works with any HuggingFace model

## Phase 5: Configuration and Integration

### 5.1 Model Factory Pattern
**File**: `src/model_factory.py`

```python
def create_model(model_name: str, backend: str = "auto", **kwargs):
    """Factory to create models with different backends"""
    
    if backend == "auto":
        # Try nnsight first for broader compatibility
        try:
            return NNsightChatModel(model_name, **kwargs)
        except Exception:
            # Fallback to transformer_lens
            return ChatModel(model_name, **kwargs)
    
    elif backend == "nnsight":
        return NNsightChatModel(model_name, **kwargs)
    
    elif backend == "transformer_lens":
        return ChatModel(model_name, **kwargs)
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

### 5.2 Configuration Updates
**File**: `configs/nnsight_config.yaml`

```yaml
models:
- name: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  backend: nnsight  # Force nnsight for DeepSeek
  batch_size: 2
- name: google/gemma-2-9b-it
  backend: auto  # Auto-detect best backend
  batch_size: 2

datasets:
- name: sports_understanding
  train_size: 200
  test_size: 800

steering:
  alpha_range: [0, 2, 4, 6]
  temperature: 0.7
  max_new_tokens: 100
```

## Phase 6: Migration Strategy

### 6.1 Gradual Migration
1. **Keep existing transformer_lens code** for backward compatibility
2. **Add nnsight implementations** as alternatives
3. **Use model factory** to choose backend automatically
4. **Test extensively** with both backends on supported models

### 6.2 Implementation Order
1. **Phase 1**: Create `NNsightChatModel` class
2. **Phase 2**: Implement activation extraction utilities
3. **Phase 3**: Build steering system with interventions
4. **Phase 4**: Create experiment runner
5. **Phase 5**: Add configuration and factory pattern
6. **Phase 6**: Test and validate results

### 6.3 Testing Strategy
- **Unit tests**: Each component individually
- **Integration tests**: Full pipeline with known models
- **Comparison tests**: Verify results match between backends where possible
- **Model compatibility**: Test with DeepSeek, Gemma, and other models

## Benefits of This Approach

### 6.4 Technical Advantages
1. **Broader model support**: Works with any HuggingFace model
2. **Cleaner API**: nnsight's intervention system is more intuitive
3. **Better performance**: More efficient activation extraction
4. **Future-proof**: Easy to add new models without framework changes
5. **Simplified debugging**: Better error messages and clearer control flow

### 6.5 Research Advantages
1. **Model diversity**: Can test hypotheses across more model families
2. **Methodological rigor**: Cleaner intervention semantics
3. **Easier debugging**: Better error handling and logging
4. **Reproducible results**: Consistent behavior across different model architectures

## Risk Mitigation

### 6.6 Potential Issues
- **API differences**: nnsight vs transformer_lens behavioral differences
- **Performance**: Initial implementation might be slower
- **Dependencies**: Additional nnsight dependency
- **Learning curve**: Team familiarity with nnsight API

### 6.7 Mitigation Strategies
- **Extensive testing**: Compare outputs between backends
- **Performance profiling**: Optimize bottlenecks as needed
- **Documentation**: Comprehensive API documentation and examples
- **Training**: Team education on nnsight concepts

## Success Metrics

### 6.8 Technical Metrics
- [ ] All existing experiments produce identical results with nnsight backend
- [ ] DeepSeek models successfully run steering experiments
- [ ] Memory usage comparable or better than current implementation
- [ ] Performance within 20% of current implementation

### 6.9 Research Metrics
- [ ] Steering effects are consistent across different model architectures
- [ ] Results are reproducible across different model families
- [ ] New models can be added without code changes
- [ ] Experiment setup time reduced

## Timeline Estimate

- **Week 1**: Phase 1 (Model wrapper)
- **Week 2**: Phase 2 (Activation extraction)
- **Week 3**: Phase 3 (Steering implementation)
- **Week 4**: Phase 4 (Experiment runner)
- **Week 5**: Phase 5 (Configuration and integration)
- **Week 6**: Phase 6 (Testing and validation)

**Total**: ~6 weeks for complete migration with testing

## Conclusion

This migration plan significantly expands model compatibility while improving the intervention system's usability. The gradual migration strategy ensures backward compatibility while enabling new capabilities. The investment in this migration will pay dividends in research reproducibility and the ability to test hypotheses across a broader range of models, including cutting-edge architectures like DeepSeek that aren't supported by transformer_lens.
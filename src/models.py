from typing import Dict, List
import os
from transformer_lens import HookedTransformer


class ChatModel:
    def __init__(self, model_name: str, device: str = "cpu", n_devices: int = 1, dtype: str = "bfloat16", cache_dir: str = os.environ['HF_HOME']):
        """
        Initialize the ChatModel.

        Args:
            model_name: Name of the model to load via transformer_lens.
            device: Device to run the model on.
            dtype: Data type for model weights.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name, device=self.device, dtype=self.dtype, cache_dir=cache_dir, n_devices=n_devices
        )

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a list of chat messages according to the model's chat template.
        """
        return self.model.tokenizer.apply_chat_template(messages, tokenize=False)

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying transformer lens model.
        return getattr(self.model, attr)

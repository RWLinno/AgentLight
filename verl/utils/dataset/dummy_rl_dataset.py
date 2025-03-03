import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

class DummyRLDataset(Dataset):
    """A dummy dataset that doesn't rely on actual files for RL environments."""
    
    def __init__(self, length=10, tokenizer=None):
        """
        Initialize the dummy dataset.
        
        Args:
            length: Number of dummy samples to generate
            tokenizer: Tokenizer to use (not actually used but needed for compatibility)
        """
        self.length = length
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Return a minimal valid structure that won't cause errors in pipeline."""
        return {
            'prompt': "dummy prompt",
            'response': "dummy response",
            'input_ids': torch.ones(1, dtype=torch.long),
            'attention_mask': torch.ones(1, dtype=torch.long),
            'prompt_with_chat_template': "dummy chat template",
            'meta': {'original_prompt': "dummy prompt"}
        } 
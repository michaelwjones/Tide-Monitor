import torch
from torch.utils.data import Dataset
import numpy as np

class TidalDataset(Dataset):
    """
    PyTorch Dataset for tidal LSTM training.
    
    Handles variable-length sequences with padding for batch training.
    """
    
    def __init__(self, X, y, max_length=4320):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences (num_samples, sequence_length)
            y: Target values (num_samples, 1)
            max_length: Maximum sequence length for padding
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.max_length = max_length
        
        print(f"Dataset initialized:")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Target shape: {self.y.shape}")
        print(f"  Max sequence length: {self.max_length}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            input_seq: Input sequence tensor (max_length, 1)
            target: Target value tensor (1,)
            seq_length: Actual sequence length
        """
        input_seq = self.X[idx]
        target = self.y[idx]
        
        # Get actual sequence length
        seq_length = len(input_seq)
        
        # Pad sequence to max_length if needed
        if seq_length < self.max_length:
            # Pad with -1 values (will be handled in model)
            padding = torch.full((self.max_length - seq_length,), -1.0)
            input_seq = torch.cat([padding, input_seq])
        elif seq_length > self.max_length:
            # Truncate to max_length (take last max_length values)
            input_seq = input_seq[-self.max_length:]
            seq_length = self.max_length
        
        # Reshape for LSTM input (sequence_length, input_size)
        input_seq = input_seq.unsqueeze(-1)
        
        return input_seq, target, seq_length
    
    def get_stats(self):
        """Return dataset statistics"""
        return {
            'num_samples': len(self.X),
            'input_mean': float(self.X.mean()),
            'input_std': float(self.X.std()),
            'input_min': float(self.X.min()),
            'input_max': float(self.X.max()),
            'target_mean': float(self.y.mean()),
            'target_std': float(self.y.std()),
            'target_min': float(self.y.min()),
            'target_max': float(self.y.max())
        }

def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of (input_seq, target, seq_length) tuples
    
    Returns:
        inputs: Batch of input sequences (batch_size, max_length, 1)
        targets: Batch of targets (batch_size, 1)
        lengths: Actual sequence lengths (batch_size,)
    """
    inputs, targets, lengths = zip(*batch)
    
    # Stack tensors
    inputs = torch.stack(inputs)  # (batch_size, max_length, 1)
    targets = torch.stack(targets)  # (batch_size, 1)
    lengths = torch.LongTensor(lengths)  # (batch_size,)
    
    return inputs, targets, lengths
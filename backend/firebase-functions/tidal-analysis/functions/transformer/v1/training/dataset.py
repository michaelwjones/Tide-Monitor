import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os

class TidalDataset(Dataset):
    """
    PyTorch Dataset for tidal prediction with seq2seq transformer.
    
    Loads preprocessed sequences from data-preparation stage:
    - Input sequences: 433 time steps (72 hours at 10-minute intervals)
    - Target sequences: 144 time steps (24 hours at 10-minute intervals)
    """
    
    def __init__(self, data_dir='../data-preparation/data', split='train'):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing preprocessed .npy files
            split: 'train' or 'val' for training/validation split
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load the appropriate split
        if split == 'train':
            self.X = np.load(os.path.join(data_dir, 'X_train.npy'))
            self.y = np.load(os.path.join(data_dir, 'y_train.npy'))
        elif split == 'val':
            self.X = np.load(os.path.join(data_dir, 'X_val.npy'))
            self.y = np.load(os.path.join(data_dir, 'y_val.npy'))
        else:
            raise ValueError(f"Split must be 'train' or 'val', got '{split}'")
        
        # Load normalization parameters
        norm_path = os.path.join(data_dir, 'normalization_params.json')
        with open(norm_path, 'r') as f:
            self.norm_params = json.load(f)
        
        # Load timestamps for reference
        timestamp_path = os.path.join(data_dir, 'timestamps.json')
        with open(timestamp_path, 'r') as f:
            self.timestamp_info = json.load(f)
        
        print(f"Loaded {split} dataset:")
        print(f"  Sequences: {len(self.X)}")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Target shape: {self.y.shape}")
        print(f"  Normalization: mean={self.norm_params['mean']:.2f}, std={self.norm_params['std']:.2f}")
    
    def __len__(self):
        """Return the number of sequences in the dataset"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a single sequence pair.
        
        Returns:
            src: Input sequence tensor (433, 1)
            tgt: Target sequence tensor (144, 1) 
        """
        # Convert to tensors and add feature dimension
        src = torch.from_numpy(self.X[idx]).float().unsqueeze(-1)  # (433, 1)
        tgt = torch.from_numpy(self.y[idx]).float().unsqueeze(-1)  # (144, 1)
        
        return src, tgt
    
    def denormalize(self, normalized_data):
        """
        Denormalize data back to original scale.
        
        Args:
            normalized_data: Normalized tensor or numpy array
            
        Returns:
            Denormalized data in mm units
        """
        mean = self.norm_params['mean']
        std = self.norm_params['std']
        
        if isinstance(normalized_data, torch.Tensor):
            return normalized_data * std + mean
        else:
            return normalized_data * std + mean
    
    def get_stats(self):
        """Get dataset statistics"""
        return {
            'split': self.split,
            'num_sequences': len(self.X),
            'input_sequence_length': self.X.shape[1],
            'output_sequence_length': self.y.shape[1],
            'mean': self.norm_params['mean'],
            'std': self.norm_params['std'],
            'data_range': self.timestamp_info['data_range']
        }

def create_data_loaders(data_dir='../data-preparation/data', 
                       batch_size=8, 
                       num_workers=0,
                       shuffle_train=True):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        data_dir: Directory containing preprocessed data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        datasets: Dict containing train/val datasets for reference
    """
    print("Creating data loaders...")
    
    # Create datasets
    train_dataset = TidalDataset(data_dir, split='train')
    val_dataset = TidalDataset(data_dir, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset
    }
    
    print(f"Data loaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, datasets

def test_dataset():
    """Test dataset loading and data loader creation"""
    print("Testing TidalDataset...")
    
    try:
        # Test dataset creation
        dataset = TidalDataset(split='train')
        print(f"Dataset loaded: {len(dataset)} sequences")
        
        # Test single item access
        src, tgt = dataset[0]
        print(f"Sample shapes - src: {src.shape}, tgt: {tgt.shape}")
        print(f"Sample dtypes - src: {src.dtype}, tgt: {tgt.dtype}")
        
        # Test denormalization
        original = dataset.denormalize(src)
        print(f"Denormalized range: {original.min():.1f} - {original.max():.1f} mm")
        
        # Test data loader
        train_loader, val_loader, datasets = create_data_loaders(batch_size=4)
        
        # Test batch loading
        for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
            print(f"Batch {batch_idx}: src {src_batch.shape}, tgt {tgt_batch.shape}")
            if batch_idx >= 2:  # Just test first few batches
                break
        
        print("Dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        return False

if __name__ == "__main__":
    test_dataset()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
import os
from datetime import datetime
import math

from model import create_model

class TidalDataset(Dataset):
    """
    PyTorch Dataset for loading transformer v2 training data.
    Handles normalized water level sequences for tidal prediction.
    """
    def __init__(self, X_path, y_path):
        """
        Initialize dataset by loading NumPy arrays.
        
        Args:
            X_path: Path to input sequences (X_train.npy or X_val.npy)
            y_path: Path to target sequences (y_train.npy or y_val.npy)
        """
        print(f"Loading dataset from {X_path} and {y_path}")
        
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)
        
        print(f"Loaded {len(self.X)} sequences")
        print(f"Input shape: {self.X.shape}, Output shape: {self.y.shape}")
        print(f"Input range: [{self.X.min():.3f}, {self.X.max():.3f}]")
        print(f"Output range: [{self.y.min():.3f}, {self.y.max():.3f}]")
        
        assert len(self.X) == len(self.y), "Input and output must have same length"
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Returns:
            x: Input sequence of shape [input_length, 1]
            y: Target sequence of shape [output_length]
        """
        x = torch.from_numpy(self.X[idx]).unsqueeze(-1)  # Add feature dimension
        y = torch.from_numpy(self.y[idx])
        return x, y

class TidalTrainer:
    """
    Trainer class for TidalTransformerV2 model.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create output directory
        self.output_dir = config.get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train_epoch(self):
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                elapsed = time.time() - start_time
                progress = batch_idx / num_batches * 100
                print(f'Epoch {self.current_epoch:3d} [{batch_idx:4d}/{num_batches:4d} ({progress:5.1f}%)] '
                      f'Loss: {loss.item():.6f}, Time: {elapsed:.1f}s')
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f'Epoch {self.current_epoch:3d} Training: Avg Loss: {avg_loss:.6f}, Time: {epoch_time:.1f}s')
        return avg_loss
    
    def validate(self):
        """Validate model performance."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        rmse = math.sqrt(avg_loss)  # Since we're using MSE
        
        print(f'Epoch {self.current_epoch:3d} Validation: Avg Loss: {avg_loss:.6f}, RMSE: {rmse:.6f}')
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'New best model saved with validation loss: {self.best_val_loss:.6f}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        
        print(f"Resumed training from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['epochs']} epochs")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            print(f'Epoch {epoch:3d} Summary: Train: {train_loss:.6f}, Val: {val_loss:.6f}, '
                  f'LR: {current_lr:.2e}, Best Val: {self.best_val_loss:.6f}')
            print('-' * 80)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f} seconds")
        
        # Save final training log
        self.save_training_log()
    
    def save_training_log(self):
        """Save training history and configuration."""
        log = {
            'config': self.config,
            'model_info': self.model.get_model_info(),
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates
            },
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.train_losses[-1] if self.train_losses else None,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None
            },
            'training_completed': datetime.now().isoformat()
        }
        
        log_path = os.path.join(self.output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"Training log saved to {log_path}")

def main():
    """Main training function."""
    
    # Training configuration
    config = {
        # Data paths (will be updated by Modal)
        'train_x_path': '/data/X_train.npy',
        'train_y_path': '/data/y_train.npy',
        'val_x_path': '/data/X_val.npy',
        'val_y_path': '/data/y_val.npy',
        
        # Model configuration
        'model_config': {
            'input_length': 432,
            'output_length': 144,
            'd_model': 512,
            'nhead': 16,
            'num_encoder_layers': 8,
            'dim_feedforward': 2048,
            'dropout': 0.1
        },
        
        # Training configuration
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'log_interval': 50,
        
        # Output configuration
        'output_dir': '/output'
    }
    
    print("Transformer v2 Training Script")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        if key != 'model_config':
            print(f"  {key}: {value}")
    print("\nModel Configuration:")
    for key, value in config['model_config'].items():
        print(f"  {key}: {value}")
    print()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TidalDataset(config['train_x_path'], config['train_y_path'])
    val_dataset = TidalDataset(config['val_x_path'], config['val_y_path'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config['model_config'])
    
    # Create trainer
    trainer = TidalTrainer(model, train_loader, val_loader, config)
    
    # Start training
    trainer.train()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
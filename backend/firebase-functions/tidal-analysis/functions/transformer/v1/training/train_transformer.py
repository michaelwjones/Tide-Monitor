import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
import time
from datetime import datetime
import argparse

from model import TidalTransformer, create_model
from dataset import create_data_loaders

class TransformerTrainer:
    """
    Trainer class for seq2seq transformer tidal prediction.
    
    Features:
    - AdamW optimizer with cosine annealing
    - Mixed precision training (if available)
    - Gradient clipping for stability
    - TensorBoard logging
    - Model checkpointing
    - Early stopping
    """
    
    def __init__(self, model, train_loader, val_loader, datasets, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.datasets = datasets
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_t0'],
            T_mult=2,
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Mixed precision training
        self.use_amp = config['use_amp'] and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training")
        
        # TensorBoard logging
        self.writer = SummaryWriter(f"runs/transformer_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Use teacher forcing during training
                    output = self.model(src, tgt, teacher_forcing_ratio=0.8)
                    loss = self.criterion(output, tgt)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(src, tgt, teacher_forcing_ratio=0.8)
                loss = self.criterion(output, tgt)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % self.config['log_interval'] == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                
                print(f'Epoch {epoch:3d} | Batch {batch_idx:4d}/{num_batches:4d} | '
                      f'Loss {loss.item():.6f} | {batches_per_sec:.1f} batches/sec')
                
                # TensorBoard logging
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], step)
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch:3d} | Train Loss {avg_loss:.6f} | Time {epoch_time:.1f}s')
        
        # Log to TensorBoard
        self.writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # No teacher forcing during validation
                        output = self.model(src)
                        loss = self.criterion(output, tgt)
                else:
                    output = self.model(src)
                    loss = self.criterion(output, tgt)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch:3d} | Val Loss   {avg_loss:.6f}')
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/EpochLoss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'normalization_params': self.datasets['train'].norm_params
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'checkpoints/latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'checkpoints/best.pth')
            print(f"New best model saved with validation loss: {loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['loss']
        
        print(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_val_loss:.6f}")
    
    def train(self):
        """Main training loop"""
        print("Starting transformer training...")
        print(f"Model: {self.model.get_model_info()}")
        print(f"Training sequences: {len(self.datasets['train'])}")
        print(f"Validation sequences: {len(self.datasets['val'])}")
        print(f"Training for {self.config['num_epochs']} epochs")
        print("-" * 80)
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_loss = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"Early stopping triggered after {self.config['patience']} epochs without improvement")
                break
            
            print("-" * 80)
        
        # Training completed
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.best_val_loss

def get_default_config():
    """Get default training configuration"""
    return {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'scheduler_t0': 10,
        'grad_clip': 1.0,
        'use_amp': True,
        'patience': 15,
        'log_interval': 10
    }

def main():
    parser = argparse.ArgumentParser(description='Train Transformer for Tidal Prediction')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default='../data-preparation/data', 
                       help='Directory containing training data')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = get_default_config()
    config['batch_size'] = args.batch_size
    config['num_epochs'] = args.epochs
    config['learning_rate'] = args.lr
    
    print("Transformer v1 Tidal Prediction Training")
    print("=" * 50)
    
    # Create data loaders
    train_loader, val_loader, datasets = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=config['batch_size']
    )
    
    # Create model
    model = create_model()
    
    # Create trainer
    trainer = TransformerTrainer(model, train_loader, val_loader, datasets, config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    best_loss = trainer.train()
    
    print(f"\nTraining finished with best validation loss: {best_loss:.6f}")
    print("Next steps:")
    print("1. Run convert_to_onnx.py to export the model for inference")
    print("2. Test the model using the testing interface")

if __name__ == "__main__":
    main()
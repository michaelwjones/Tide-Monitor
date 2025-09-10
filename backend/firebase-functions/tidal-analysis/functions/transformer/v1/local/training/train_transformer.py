import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
import sys
import time
from datetime import datetime, timedelta

# Import model from tidal-analysis root directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
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
        self.run_name = f"transformer_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        
        # Text logging
        self.log_file = os.path.join("runs", self.run_name, "training.log")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Progress tracking
        self.training_start_time = None
        self.epoch_times = []
        self.epoch_losses = []
        
        # Create checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)
        
    def log_message(self, message):
        """Write message to both console and log file"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def format_time(self, seconds):
        """Format seconds into readable time string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def get_progress_bar(self, current, total, length=40):
        """Create a text progress bar"""
        filled = int(length * current / total)
        bar = '█' * filled + '░' * (length - filled)
        percentage = 100.0 * current / total
        return f"[{bar}] {percentage:.1f}%"
    
    def estimate_remaining_time(self, epoch):
        """Estimate remaining training time based on current progress"""
        if len(self.epoch_times) < 2:
            return "Calculating..."
        
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.config['num_epochs'] - epoch - 1
        remaining_seconds = avg_epoch_time * remaining_epochs
        
        return self.format_time(remaining_seconds)
    
    def print_epoch_header(self, epoch):
        """Print formatted epoch header"""
        total_epochs = self.config['num_epochs']
        progress_bar = self.get_progress_bar(epoch, total_epochs)
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1:3d}/{total_epochs} {progress_bar}")
        
        if self.training_start_time and len(self.epoch_times) > 0:
            elapsed = time.time() - self.training_start_time
            remaining = self.estimate_remaining_time(epoch)
            print(f"Elapsed: {self.format_time(elapsed)} | Remaining: {remaining}")
        
        print(f"{'='*80}")
    
    def print_batch_progress(self, epoch, batch_idx, num_batches, loss, elapsed):
        """Print formatted batch progress"""
        # Calculate progress
        batch_progress = self.get_progress_bar(batch_idx + 1, num_batches, 30)
        batches_per_sec = (batch_idx + 1) / elapsed
        
        # Estimate time for remaining batches in this epoch
        if batch_idx > 0:
            avg_batch_time = elapsed / (batch_idx + 1)
            remaining_batches = num_batches - batch_idx - 1
            eta_epoch = avg_batch_time * remaining_batches
            eta_str = f"ETA: {self.format_time(eta_epoch)}"
        else:
            eta_str = "ETA: Calculating..."
        
        print(f"  Batch {batch_idx + 1:4d}/{num_batches} {batch_progress}")
        print(f"  Loss: {loss:.6f} | Speed: {batches_per_sec:.2f} batch/s | {eta_str}")
        print()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Print epoch header
        self.print_epoch_header(epoch)
        
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
            
            # Clear cache periodically to prevent memory buildup
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log batch progress
            if batch_idx % self.config['log_interval'] == 0:
                elapsed = time.time() - start_time
                self.print_batch_progress(epoch, batch_idx, num_batches, loss.item(), elapsed)
                
                # TensorBoard logging
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], step)
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        # Store epoch time for ETA calculation
        self.epoch_times.append(epoch_time)
        self.epoch_losses.append(avg_loss)
        
        print(f"\n📊 EPOCH {epoch + 1} TRAINING COMPLETE")
        print(f"   Train Loss: {avg_loss:.6f}")
        print(f"   Duration: {self.format_time(epoch_time)}")
        print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Log to TensorBoard
        self.writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        print(f"\n🔍 VALIDATING MODEL...")
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (src, tgt) in enumerate(self.val_loader):
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
                
                # Show progress for longer validations
                if num_batches > 10 and batch_idx % max(1, num_batches // 5) == 0:
                    progress = self.get_progress_bar(batch_idx + 1, num_batches, 20)
                    print(f"   {progress} Batch {batch_idx + 1}/{num_batches}")
        
        avg_loss = total_loss / num_batches
        val_time = time.time() - val_start_time
        
        print(f"\n✅ VALIDATION COMPLETE")
        print(f"   Validation Loss: {avg_loss:.6f}")
        print(f"   Duration: {self.format_time(val_time)}")
        
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
            self.log_message(f"💾 New best model saved with validation loss: {loss:.6f}")
    
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
        # Initialize training timer
        self.training_start_time = time.time()
        training_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log training setup to both console and file
        self.log_message("="*80)
        self.log_message("🚀 TRANSFORMER TRAINING STARTED")
        self.log_message("="*80)
        
        # Model info display
        model_info = self.model.get_model_info()
        self.log_message("📋 MODEL CONFIGURATION:")
        self.log_message(f"   Architecture: {model_info['architecture']}")
        self.log_message(f"   Parameters: {model_info['total_parameters']:,}")
        self.log_message(f"   Input Length: {model_info['input_sequence_length']} timesteps")
        self.log_message(f"   Output Length: {model_info['output_sequence_length']} timesteps")
        self.log_message(f"   Interval: {model_info.get('interval_minutes', 1)} minutes")
        
        # Dataset info
        self.log_message("")
        self.log_message("📊 DATASET INFORMATION:")
        self.log_message(f"   Training sequences: {len(self.datasets['train']):,}")
        self.log_message(f"   Validation sequences: {len(self.datasets['val']):,}")
        self.log_message(f"   Batch size: {self.config['batch_size']}")
        
        # Training config
        self.log_message("")
        self.log_message("⚙️  TRAINING CONFIGURATION:")
        self.log_message(f"   Total epochs: {self.config['num_epochs']}")
        self.log_message(f"   Learning rate: {self.config['learning_rate']:.2e}")
        self.log_message(f"   Mixed precision: {self.use_amp}")
        self.log_message(f"   Device: {self.device}")
        
        # Data augmentation config
        self.log_message("")
        if self.config.get('augment', False):
            self.log_message("🔄 DATA AUGMENTATION:")
            self.log_message(f"   Missing value prob: {self.config.get('missing_prob', 0):.1%}")
            self.log_message(f"   Gap simulation prob: {self.config.get('gap_prob', 0):.1%}")
            self.log_message(f"   Noise std: {self.config.get('noise_std', 0):.2f}")
            self.log_message(f"   Sequence shuffling: {self.config.get('shuffle_train', False)}")
        else:
            self.log_message("🔄 DATA AUGMENTATION: Disabled")
            self.log_message(f"   Sequence shuffling: {self.config.get('shuffle_train', False)}")
        
        self.log_message("")
        self.log_message(f"⏰ Training started at: {training_start_str}")
        self.log_message(f"📝 Log file: {self.log_file}")
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_loss = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            improvement = ""
            if is_best:
                improvement_amount = self.best_val_loss - val_loss
                improvement = f" 🎉 NEW BEST! (↓{improvement_amount:.6f})"
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                improvement = f" (No improvement for {self.patience_counter} epochs)"
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Log epoch summary to file and console
            total_elapsed = time.time() - self.training_start_time
            self.log_message("")
            self.log_message(f"📈 EPOCH {epoch + 1} SUMMARY")
            self.log_message(f"   Train Loss: {train_loss:.6f}")
            self.log_message(f"   Val Loss:   {val_loss:.6f}{improvement}")
            self.log_message(f"   Best Loss:  {self.best_val_loss:.6f}")
            self.log_message(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            self.log_message(f"   Epoch Time: {self.format_time(self.epoch_times[-1])}")
            self.log_message(f"   Total Time: {self.format_time(total_elapsed)}")
            
            if len(self.epoch_times) > 1:
                remaining = self.estimate_remaining_time(epoch)
                self.log_message(f"   ETA: {remaining}")
            
            # Early stopping check
            if self.patience_counter >= self.config['patience']:
                self.log_message("")
                self.log_message("⏹️  EARLY STOPPING TRIGGERED")
                self.log_message(f"   No improvement for {self.config['patience']} consecutive epochs")
                break
        
        # Training completed
        total_training_time = time.time() - self.training_start_time
        completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.log_message("")
        self.log_message("="*80)
        self.log_message("🏁 TRAINING COMPLETED SUCCESSFULLY!")
        self.log_message("="*80)
        self.log_message("✅ Final Results:")
        self.log_message(f"   Best Validation Loss: {self.best_val_loss:.6f}")
        self.log_message(f"   Total Epochs: {len(self.epoch_times)}")
        self.log_message(f"   Total Training Time: {self.format_time(total_training_time)}")
        self.log_message(f"   Average Epoch Time: {self.format_time(sum(self.epoch_times) / len(self.epoch_times))}")
        self.log_message(f"   Completed at: {completion_time}")
        
        self.log_message("")
        self.log_message("📁 Saved Files:")
        self.log_message(f"   • Best Model: checkpoints/best.pth")
        self.log_message(f"   • Latest Model: checkpoints/latest.pth")
        self.log_message(f"   • TensorBoard Logs: runs/{self.run_name}/")
        self.log_message(f"   • Training Log: {self.log_file}")
        
        self.log_message("")
        self.log_message("🚀 Next Steps:")
        self.log_message("   1. Run model_server.py to test the model locally")
        self.log_message("   2. Test the model using the testing interface")
        self.log_message("   3. Deploy to Firebase Functions")
        
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
        'log_interval': 5,
        # Data augmentation parameters
        'augment': True,
        'missing_prob': 0.02,  # 2% chance of individual missing values
        'gap_prob': 0.05,      # 5% chance of creating gaps per sequence
        'noise_std': 0.1,      # Noise standard deviation in normalized space
        'shuffle_train': True  # Randomize sequence order each epoch
    }

def get_optimized_config():
    """Get optimized training configuration from hyperparameter sweep"""
    return {
        # Optimal hyperparameters from Modal H100 sweep (val_loss: 0.0365)
        'batch_size': 32,
        'learning_rate': 5.845165970318201e-05,
        'weight_decay': 2.0160787083950938e-06,
        'dropout': 0.17305796046740565,
        
        # Model architecture (to be applied in model creation)
        'd_model': 512,
        'nhead': 16, 
        'num_layers': 4,
        
        # Training parameters (optimized)
        'num_epochs': 100,
        'scheduler_t0': 10,
        'grad_clip': 1.0,
        'use_amp': True,
        'patience': 20,  # Increased patience for final training
        'log_interval': 5,
        
        # Data augmentation parameters
        'augment': True,
        'missing_prob': 0.02,
        'gap_prob': 0.05,
        'noise_std': 0.1,
        'shuffle_train': True
    }

def get_gtx1070_config():
    """Get GTX 1070-optimized training configuration (8GB VRAM)"""
    return {
        # GTX 1070-optimized parameters (reduced memory usage)
        'batch_size': 16,  # Reduced from 32 for 8GB VRAM
        'learning_rate': 5.845165970318201e-05,  # Keep optimal LR
        'weight_decay': 2.0160787083950938e-06,
        'dropout': 0.17305796046740565,
        
        # Model architecture (slightly reduced for GTX 1070)
        'd_model': 512,  # Keep optimal size
        'nhead': 16, 
        'num_layers': 4,
        
        # Training parameters
        'num_epochs': 100,
        'scheduler_t0': 10,
        'grad_clip': 1.0,
        'use_amp': True,  # Essential for GTX 1070 memory efficiency
        'patience': 25,  # Increased patience for smaller batches
        'log_interval': 8,  # More frequent logging for smaller batches
        
        # Data augmentation parameters
        'augment': True,
        'missing_prob': 0.02,
        'gap_prob': 0.05,
        'noise_std': 0.1,
        'shuffle_train': True
    }

def main():
    print("Transformer v1 Tidal Prediction Training")
    print("=" * 50)
    
    # Detect GPU and use appropriate configuration
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Detected: {gpu_name} ({total_memory:.1f} GB)")
        
        # Auto-select configuration based on GPU
        if "GTX 1070" in gpu_name or total_memory < 10:
            print("Using GTX 1070-optimized configuration")
            config = get_gtx1070_config()
            expected_loss = "~0.037-0.040"
        else:
            print("Using optimal hyperparameters from Modal H100 sweep")
            config = get_optimized_config()
            expected_loss = "~0.0365"
    else:
        print("No GPU detected - using CPU with reduced configuration")
        config = get_gtx1070_config()
        config['batch_size'] = 8  # Further reduce for CPU
        config['use_amp'] = False  # No mixed precision on CPU
        expected_loss = "~0.040-0.045"
    
    print("=" * 50)
    
    # Print configuration
    print("Configuration:")
    print(f"   d_model: {config['d_model']}, nhead: {config['nhead']}, layers: {config['num_layers']}")
    print(f"   Learning Rate: {config['learning_rate']:.2e}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Weight Decay: {config['weight_decay']:.2e}")
    print(f"   Dropout: {config['dropout']:.3f}")
    print(f"   Mixed Precision: {config['use_amp']}")
    print(f"   Expected Val Loss: {expected_loss}")
    print()
    
    # Create data loaders
    train_loader, val_loader, datasets = create_data_loaders(
        data_dir='../data-preparation/data',
        batch_size=config['batch_size'],
        shuffle_train=config['shuffle_train'],
        augment=config['augment'],
        missing_prob=config['missing_prob'],
        gap_prob=config['gap_prob'],
        noise_std=config['noise_std']
    )
    
    # Create model with optimized architecture
    model = create_model(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Create trainer
    trainer = TransformerTrainer(model, train_loader, val_loader, datasets, config)
    
    # Start training
    best_loss = trainer.train()
    
    print(f"\nTraining finished with best validation loss: {best_loss:.6f}")
    print(f"Target validation loss was: 0.0365")
    
    if best_loss <= 0.04:
        print("EXCELLENT: Model achieved target performance!")
    elif best_loss <= 0.05:
        print("GOOD: Model performed well!")
    else:
        print("WARNING: Model performance differs from sweep - check for issues")
    
    print("\nNext steps:")
    print("1. Run model_server.py to test the model locally for inference")
    print("2. Test the model using the testing interface")
    print("3. Deploy to Firebase Functions for production")

if __name__ == "__main__":
    main()
import os
# this must be set before "import torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

from model import TidalLSTM
from dataset import TidalDataset, collate_fn

def load_training_data():
    """Load preprocessed training data"""
    try:
        X = np.load('../data-preparation/data/X_train.npy')
        y = np.load('../data-preparation/data/y_train.npy')
        
        # Reshape y to (num_samples, 1) if needed
        if len(y.shape) == 3:
            y = y.squeeze(-1)  # Remove last dimension if it's 1
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        print(f"Loaded training data: X{X.shape}, y{y.shape}")
        return X, y
    except FileNotFoundError:
        print("Error: Training data not found. Run data-preparation scripts first.")
        return None, None

def create_dataloaders(X, y, batch_size=32, train_split=0.8):
    """Create train/validation dataloaders"""
    dataset = TidalDataset(X, y)
    
    # Split into train/validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    return train_loader, val_loader

def train_epoch_iterative(model, dataloader, criterion, optimizer, device, scaler,
                          teacher_forcing_ratio=1.0, accumulation_steps=1, memory_efficient=True, random_sampling=False):
    """Train for one epoch using iterative prediction with optimal FP16, gradient checkpointing, and random sampling"""
    model.train()
    total_loss = 0
    num_batches = 0
    accumulation_count = 0
    
    # Verify model parameters require gradients
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count == 0:
        raise RuntimeError("No model parameters require gradients!")
    
    # Initialize gradient accumulation
    optimizer.zero_grad()
    
    for batch_idx, (inputs, target_sequences, lengths) in enumerate(tqdm(dataloader, desc="Iterative Training")):
        inputs, target_sequences = inputs.to(device), target_sequences.to(device)
        batch_size = inputs.size(0)
        
        # Determine sampling strategy
        if random_sampling:
            # Random sampling: select subset of timesteps to train on
            num_samples = min(100, target_sequences.size(1))  # Sample 100 random timesteps
            sample_indices = torch.randperm(target_sequences.size(1))[:num_samples].sort()[0]
        else:
            # Use all timesteps (original approach)
            sample_indices = torch.arange(target_sequences.size(1))
        
        # Use automatic mixed precision
        with autocast('cuda'):
            # Initialize for iterative prediction
            current_sequence = inputs.clone()  # Start with 72 hours of input
            accumulated_loss = 0
            predictions_made = 0
            
            # Iteratively predict selected steps
            for i, step in enumerate(sample_indices):
                # Use gradient checkpointing to save memory
                def prediction_step(seq):
                    return model(seq)
                
                # Use gradient checkpointing only when gradients are needed and every 20th step
                if memory_efficient and i % 20 == 0 and current_sequence.requires_grad:
                    prediction = checkpoint.checkpoint(prediction_step, current_sequence, use_reentrant=False)
                else:
                    prediction = model(current_sequence)
                
                # Compute loss for this step
                step_target = target_sequences[:, step:step+1]  # Shape: (batch_size, 1)
                step_loss = criterion(prediction, step_target)
                accumulated_loss += step_loss
                predictions_made += 1
                
                # Update sequence for next prediction (teacher forcing)
                if torch.rand(1).item() < teacher_forcing_ratio:
                    # Use real target value (teacher forcing)
                    next_value = step_target
                else:
                    # Use model prediction
                    next_value = prediction
                
                # Create new sequence by sliding window - preserve gradients properly
                if i < len(sample_indices) - 1:  # Only update if not last step
                    sequence_without_first = current_sequence[:, 1:, :]  # Shape: (batch_size, 4319, 1)
                    new_value_reshaped = next_value.unsqueeze(-1)        # Shape: (batch_size, 1, 1)
                    current_sequence = torch.cat([sequence_without_first, new_value_reshaped], dim=1)
                
                # Periodic memory cleanup for long sequences
                if memory_efficient and i % 10 == 0 and i > 0:
                    torch.cuda.empty_cache()
            
        # Average loss across all prediction steps and scale by accumulation steps
        loss = (accumulated_loss / predictions_made) / accumulation_steps
        
        # Debug: Check loss properties
        if batch_idx == 0:  # Only on first batch to avoid spam
            print(f"Debug: Loss requires_grad={loss.requires_grad}, loss_value={loss.item():.6f}")
            print(f"Debug: Accumulated_loss requires_grad={accumulated_loss.requires_grad}")
        
        # Ensure loss requires gradients
        if not loss.requires_grad:
            print(f"Warning: Loss does not require gradients on batch {batch_idx}")
            continue  # Skip this batch
        
        # Backward pass with optional gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accumulation_count += 1
        
        # Update weights every accumulation_steps
        if accumulation_count >= accumulation_steps or batch_idx == len(dataloader) - 1:
            # Gradient clipping and optimizer step with optional scaling
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            accumulation_count = 0
        
        total_loss += loss.item() * accumulation_steps  # Scale back for reporting
        num_batches += 1
        
        # Periodic memory cleanup
        if memory_efficient and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Wrapper for backward compatibility - uses iterative training"""
    return train_epoch_iterative(model, dataloader, criterion, optimizer, device, teacher_forcing_ratio=1.0)

def validate_epoch_iterative(model, dataloader, criterion, device):
    """Validate using full iterative prediction (no teacher forcing) with AMP"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, target_sequences, lengths in tqdm(dataloader, desc="Iterative Validation"):
            inputs, target_sequences = inputs.to(device), target_sequences.to(device)
            
            with autocast('cuda'):
                # Use the model's predict_sequence method for validation
                # Full 24-hour validation (1440 steps)
                predictions = []
                current_sequence = inputs.clone()
                
                # Iteratively predict full 1440 steps using model's own predictions
                for step in range(target_sequences.size(1)):  # 1440 steps
                    prediction = model(current_sequence)  # Shape: (batch_size, 1)
                    predictions.append(prediction)
                    
                    # Update sequence using prediction (no teacher forcing in validation)
                    current_sequence = torch.cat([
                        current_sequence[:, 1:, :],  # Remove first timestep
                        prediction.unsqueeze(-1)     # Add prediction as (batch_size, 1, 1)
                    ], dim=1)
                
                # Stack predictions and compute loss
                predictions_tensor = torch.cat(predictions, dim=1)  # (batch_size, 1440)
                loss = criterion(predictions_tensor, target_sequences)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """Wrapper for backward compatibility - uses iterative validation"""
    return validate_epoch_iterative(model, dataloader, criterion, device)

def train_model():
    """Main training function"""
    print("LSTM v1 Training Pipeline")
    print("=" * 30)
    
    # Load data
    X, y = load_training_data()
    if X is None:
        return
    
    # Training configuration for iterative 24-hour prediction (FP16 mixed precision)
    config = {
        'hidden_size': 128,  # Increased from 64 for better GPU utilization
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,  # Can use higher LR with mixed precision
        'batch_size': 4,     # Will be set based on device
        'accumulation_steps': 1,  # Simulate batch size through gradient accumulation
        'num_epochs': 100,   # More epochs needed for iterative learning
        'patience': 15,      # Increased patience for iterative training
        'teacher_forcing_ratio': 1.0,  # Start with 100% teacher forcing
        'validation_frequency': 10,    # Less frequent validation to save memory
        'memory_efficient': True,      # Enable memory optimizations
        'random_sampling': True,       # Use random sampling for memory efficiency
        'use_amp': True                # Use automatic mixed precision
    }
    
    print(f"Training configuration: {config}")
    
    # Device selection with robust CUDA checking
    cuda_available = False
    device = torch.device('cpu')  # Default to CPU
    
    try:
        if torch.cuda.is_available() and hasattr(torch, 'version') and torch.version.cuda:
            # Additional check to ensure CUDA is actually functional
            test_tensor = torch.tensor([1.0])
            test_tensor.cuda()  # This will fail if CUDA not properly compiled
            cuda_available = True
            device = torch.device('cuda')
    except (RuntimeError, AssertionError) as e:
        print(f"⚠️  CUDA check failed: {e}")
        cuda_available = False
        device = torch.device('cpu')
    
    if cuda_available and device.type == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU Training Enabled!")
            print(f"  Device: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            
            # Clear GPU cache before starting
            torch.cuda.empty_cache()
            
            # Enable GPU optimizations with memory efficiency
            torch.backends.cudnn.benchmark = False  # Disable for memory consistency
            torch.backends.cudnn.enabled = True
            
            # Memory optimization settings
            if config['memory_efficient']:
                torch.cuda.set_per_process_memory_fraction(1.0)  # Use 100% of 8GB
                print(f"  GPU memory limit set to 100% (8GB)")
                
                # Enable memory efficient attention if available
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    print(f"  Flash attention enabled for memory efficiency")
                except:
                    pass
                
                # Enable AMP if requested
                if config['use_amp']:
                    print(f"  Automatic Mixed Precision enabled")
                    print(f"  Expected memory reduction: ~30-40%")
                    
                if config['random_sampling']:
                    print(f"  Random sampling enabled (20 steps per batch)")
                    print(f"  Expected memory reduction: ~85%")
            
        except Exception as e:
            print(f"⚠️  GPU setup failed: {e}")
            print(f"  Falling back to CPU training")
            cuda_available = False
            device = torch.device('cpu')
    
    if not cuda_available:
        print(f"⚠️  Using CPU Training")
        print(f"  Reason: {'No CUDA support in PyTorch installation' if not hasattr(torch.version, 'cuda') or not torch.version.cuda else 'GPU not available'}")
        print(f"  For GPU training: Run install-pytorch-interactive.bat and select CUDA option")
        print(f"  Or visit: https://pytorch.org/get-started/locally/")
    
    # Update batch size based on actual device capability (optimized for 7.5GB limit)
    if not cuda_available:
        print(f"  Error: GPU training not available")
        return
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X, y, config['batch_size'])
    
    # Initialize model
    model = TidalLSTM(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Note: Model stays in FP32, autocast handles FP16 conversions automatically
    if config['use_amp'] and cuda_available:
        print(f"Automatic Mixed Precision enabled")
    
    print(f"Model info: {model.get_model_info()}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if config['use_amp'] and cuda_available else None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    
    print(f"\nStarting iterative training with teacher forcing...")
    print(f"Each epoch trains on 1440-step sequences (24 hours)")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train with current teacher forcing ratio, gradient checkpointing, and random sampling
        current_tf_ratio = config['teacher_forcing_ratio']
        train_loss = train_epoch_iterative(
            model, train_loader, criterion, optimizer, device, scaler,
            teacher_forcing_ratio=current_tf_ratio,
            accumulation_steps=config['accumulation_steps'],
            memory_efficient=config['memory_efficient'],
            random_sampling=config['random_sampling']
        )
        
        # Validate every few epochs (full 24-hour validation)
        if (epoch + 1) % config['validation_frequency'] == 0:
            print(f"Running full 24-hour validation (1440 steps)...")
            val_loss = validate_epoch_iterative(
                model, val_loader, criterion, device
            )
        else:
            val_loss = float('inf')  # Skip validation
        
        # Update scheduler only when validation was run
        if val_loss != float('inf'):
            scheduler.step(val_loss)
        
        # Track losses
        train_losses.append(train_loss)
        if val_loss != float('inf'):
            val_losses.append(val_loss)
        
        # GPU memory monitoring
        gpu_memory_info = ""
        if cuda_available and device.type == 'cuda':
            try:
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                gpu_memory_info = f", GPU: {memory_allocated:.1f}/{memory_reserved:.1f} GB"
            except Exception:
                gpu_memory_info = ""
        
        # Display progress
        if val_loss != float('inf'):
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}{gpu_memory_info}")
        else:
            print(f"Train Loss: {train_loss:.6f}, Val Loss: skipped{gpu_memory_info}")
        
        # Early stopping (only when validation was run)
        if val_loss != float('inf') and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs('trained_models', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'teacher_forcing_ratio': current_tf_ratio
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, 'trained_models/best_model.pth')
            
            print(f"  → New best model saved (val_loss: {val_loss:.6f})")
        elif val_loss != float('inf'):
            patience_counter += 1
            
        # Check early stopping only after validation epochs
        if val_loss != float('inf') and patience_counter >= config['patience']:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # Save final training history
    training_history = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'epochs_trained': epoch + 1,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('trained_models/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Final GPU cleanup (only if CUDA is actually available)
    if cuda_available and device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            final_memory = torch.cuda.memory_allocated(0) / 1024**3
            max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
            print(f"🧹 GPU cache cleared")
            print(f"   Final memory usage: {final_memory:.1f} GB")
            print(f"   Peak memory usage: {max_memory:.1f} GB")
            
            # Reset memory stats for next run
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            print("🧹 GPU cleanup attempted (some features unavailable)")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: trained_models/best_model.pth")
    print(f"Next step: Run convert_to_onnx.py to prepare for deployment")

if __name__ == "__main__":
    train_model()
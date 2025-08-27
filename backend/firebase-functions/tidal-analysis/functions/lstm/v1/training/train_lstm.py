import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
import os
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for inputs, targets, lengths in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(inputs)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, lengths in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            predictions = model(inputs)
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_model():
    """Main training function"""
    print("LSTM v1 Training Pipeline")
    print("=" * 30)
    
    # Load data
    X, y = load_training_data()
    if X is None:
        return
    
    # Training configuration (optimized for GPU if available)
    config = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64 if torch.cuda.is_available() else 32,  # Larger batch for GPU
        'num_epochs': 50,
        'patience': 10  # Early stopping patience
    }
    
    print(f"Training configuration: {config}")
    
    # Device selection with detailed GPU information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Training Enabled!")
        print(f"  Device: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
    else:
        print(f"⚠️  Using CPU Training (GPU not available)")
        print(f"  For faster training, install CUDA-enabled PyTorch")
        print(f"  Visit: https://pytorch.org/get-started/locally/")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X, y, config['batch_size'])
    
    # Initialize model
    model = TidalLSTM(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # GPU memory monitoring
        gpu_memory_info = ""
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_memory_info = f", GPU: {memory_allocated:.1f}/{memory_reserved:.1f} GB"
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}{gpu_memory_info}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs('trained_models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, 'trained_models/best_model.pth')
            
            print(f"  → New best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
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
    
    # Final GPU cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        print(f"🧹 GPU cache cleared, final memory usage: {final_memory:.1f} GB")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: trained_models/best_model.pth")
    print(f"Next step: Run convert_to_onnx.py to prepare for deployment")

if __name__ == "__main__":
    train_model()
#!/usr/bin/env python3
"""
Tide Transformer v1 - Modal Single Training Run
Single high-performance GPU training run using optimal hyperparameters
"""

import modal
import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path

# Modal app definition
app = modal.App("tide-transformer-v1-single-run")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ], index_url="https://download.pytorch.org/whl/cu118")  # CUDA 11.8 PyTorch
    .pip_install([
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "tensorboard>=2.11.0",
    ])  # Other packages from default index
    .apt_install(["wget"])
)

# Create volume for training data persistence
volume = modal.Volume.from_name("tide-training-data", create_if_missing=True)

class TidalDataset:
    """Simplified dataset class for Modal"""
    
    def __init__(self, X, y, sequence_length=433):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerModel(nn.Module):
    """Transformer model for tidal prediction"""
    
    def __init__(self, input_dim=1, d_model=512, nhead=8, num_layers=6, 
                 output_dim=144, dropout=0.1, max_seq_length=433):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        seq_length = x.size(1)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_length].unsqueeze(0)
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use last token for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Project to output
        output = self.output_projection(x)
        return output

@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": volume},
    timeout=18000,  # 5 hours max for single run
    memory=32768,  # 32GB RAM for H100
)
def run_single_training():
    """Single training run with optimal hyperparameters"""
    
    print("🚀 Starting Tide Transformer v1 Single Training Run")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🚀 Using H100 GPU for high-performance training!")
    else:
        print("⚠️  No GPU available, using CPU")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Using device: {device}")
    
    # Check if training data exists
    data_files = ["/data/X_train.npy", "/data/y_train.npy", "/data/X_val.npy", "/data/y_val.npy"]
    missing_files = [f for f in data_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing training data files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Upload your training data first with setup script")
        return {"error": "Missing training data"}
    
    print("✅ Training data found")
    print()
    
    # Load training data
    X_train = np.load("/data/X_train.npy")
    y_train = np.load("/data/y_train.npy") 
    X_val = np.load("/data/X_val.npy")
    y_val = np.load("/data/y_val.npy")
    
    # Load normalization params
    with open("/data/normalization_params.json", 'r') as f:
        norm_params = json.load(f)
    
    print(f"📊 Dataset Information:")
    print(f"   Training sequences: {len(X_train):,}")
    print(f"   Validation sequences: {len(X_val):,}")
    print(f"   Input sequence length: {X_train.shape[1]}")
    print(f"   Output sequence length: {y_train.shape[1]}")
    print()
    
    # Optimal hyperparameters from sweep
    config = {
        "d_model": 512,
        "nhead": 16,
        "num_layers": 4,
        "dropout": 0.17305796046740565,
        "learning_rate": 5.845165970318201e-05,
        "batch_size": 32,
        "weight_decay": 2.0160787083950938e-06,
        "num_epochs": 150,  # More epochs for single run
    }
    
    print("🎯 Optimal Configuration:")
    print(f"   d_model: {config['d_model']}, nhead: {config['nhead']}, layers: {config['num_layers']}")
    print(f"   Learning Rate: {config['learning_rate']:.2e}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Weight Decay: {config['weight_decay']:.2e}")
    print(f"   Dropout: {config['dropout']:.3f}")
    print(f"   Epochs: {config['num_epochs']}")
    print()
    
    # Create datasets
    train_dataset = TidalDataset(X_train, y_train)
    val_dataset = TidalDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = TransformerModel(
        input_dim=1,
        d_model=config["d_model"],
        nhead=config["nhead"], 
        num_layers=config["num_layers"],
        output_dim=144,  # 24 hours * 6 (10-minute intervals)
        dropout=config["dropout"]
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  Model Architecture:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    print("🔄 Starting Training...")
    print("=" * 60)
    
    for epoch in range(config["num_epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(-1))
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x.unsqueeze(-1))
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        # Update learning rate
        scheduler.step()
        
        # Check for improvement
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model to volume
            model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config,
                'normalization_params': norm_params
            }
            torch.save(model_state, '/data/best_single_run.pth')
        else:
            patience_counter += 1
        
        # Log progress every 10 epochs
        if epoch % 10 == 0 or is_best:
            print(f"Epoch {epoch + 1:3d}/{config['num_epochs']} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {avg_val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print("=" * 60)
    print("🏆 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"🎯 Best Validation Loss: {best_val_loss:.6f}")
    print(f"📁 Model saved to: /data/best_single_run.pth")
    print()
    
    # Save training summary
    summary = {
        "best_val_loss": best_val_loss,
        "total_epochs": epoch + 1,
        "config": config,
        "model_parameters": total_params,
        "final_lr": optimizer.param_groups[0]['lr']
    }
    
    with open("/data/single_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("💾 Training summary saved to /data/single_run_summary.json")
    print("🚀 Ready for deployment!")
    
    return summary

# Local entry point for uploading data  
@app.local_entrypoint()
def upload_training_data():
    """Upload training data to Modal volume using batch_upload"""
    
    # Use absolute path to avoid any Windows path issues
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir.parent.parent.parent / "local" / "data-preparation" / "data"
    data_dir = data_dir.resolve()  # Resolve to absolute path
    
    required_files = [
        "X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", 
        "normalization_params.json"
    ]
    
    print(f"📤 Uploading training data from {data_dir}")
    
    # Check files exist first
    for filename in required_files:
        local_file = data_dir / filename
        if not local_file.exists():
            print(f"❌ Missing {filename} at {local_file.absolute()}")
            return False
        print(f"✓ Found {filename}")
    
    # Upload files using batch_upload
    try:
        with volume.batch_upload() as batch:
            for filename in required_files:
                local_file = data_dir / filename
                local_file_str = str(local_file.absolute())
                batch.put_file(local_file_str, f"/{filename}")
                print(f"✅ Uploaded {filename}")
        
        print("✅ All training data uploaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

# Local entry point
@app.local_entrypoint()
def main(
    upload_data: bool = False,
    run_training: bool = False
):
    """
    Main entry point for single training run
    
    Usage:
        modal run modal_single_run.py --upload-data
        modal run modal_single_run.py --run-training
    """
    
    if upload_data:
        return upload_training_data()
    
    if run_training:
        print("🚀 Starting single training run...")
        results = run_single_training.remote()
        print("=" * 60)
        print("🏆 SINGLE TRAINING RUN COMPLETE!")
        print("=" * 60)
        return results
    
    print("Usage:")
    print("  1. Upload data: modal run modal_single_run.py --upload-data")
    print("  2. Run training: modal run modal_single_run.py --run-training")
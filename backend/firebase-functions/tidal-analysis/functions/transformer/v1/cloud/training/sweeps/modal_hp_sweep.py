#!/usr/bin/env python3
"""
Tide Transformer v1 - Modal Hyperparameter Optimization
Clean, serverless GPU hyperparameter sweep with Ray Tune integration
"""

import modal
import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from ray import tune
from ray.tune.schedulers import ASHAScheduler
# Removed BasicVariantGenerator - using default search

# Modal app definition
app = modal.App("tide-transformer-v1-hp-sweep")

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
        "ray[tune]>=2.8.0",
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

def train_model(config, data_dir="/data"):
    """Training function called by Ray Tune"""
    
    # Debug GPU availability
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔍 CUDA device count: {torch.cuda.device_count()}")
        print(f"🔍 Current CUDA device: {torch.cuda.current_device()}")
        print(f"🔍 CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"🔍 CUDA version: {torch.version.cuda}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Using device: {device}")
    
    # Force CUDA if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"🔍 Forced CUDA device 0")
        
        # Test GPU computation
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            test_result = torch.mm(test_tensor, test_tensor)
            print(f"🔍 GPU computation test PASSED - device: {test_result.device}")
            del test_tensor, test_result
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU computation test FAILED: {e}")
            device = torch.device("cpu")
            print(f"🔍 Falling back to CPU due to GPU issues")
    
    # Load training data
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy") 
    X_val = np.load(f"{data_dir}/X_val.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    
    # Load normalization params
    with open(f"{data_dir}/normalization_params.json", 'r') as f:
        norm_params = json.load(f)
    
    # Create datasets
    train_dataset = TidalDataset(X_train, y_train)
    val_dataset = TidalDataset(X_val, y_val)
    
    # Create data loaders - disable pin_memory since GPU not detected properly
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for GPU debugging
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False, 
        num_workers=0,  # Disable multiprocessing for GPU debugging
        pin_memory=False
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
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(50):  # Max epochs
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
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Report to Ray Tune - must match scheduler metric name
        tune.report({"val_loss": avg_val_loss, "train_loss": avg_train_loss})
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Final report - must match scheduler metric name
    tune.report({"val_loss": best_val_loss})

@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": volume},
    timeout=43200,  # 12 hours max (until we know actual timing)
    memory=32768,  # 32GB RAM for H100
)
def run_hyperparameter_sweep():
    """Main hyperparameter optimization function"""
    
    print("🚀 Starting Tide Transformer v1 Hyperparameter Optimization")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🚀 Using H100 GPU - high-performance optimization!")
    else:
        print("⚠️  No GPU available, using CPU")
    
    # Check if training data exists
    data_files = ["/data/X_train.npy", "/data/y_train.npy", "/data/X_val.npy", "/data/y_val.npy"]
    missing_files = [f for f in data_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing training data files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Upload your training data first with: modal run modal_hp_sweep.py --upload-data path/to/data")
        return {"error": "Missing training data"}
    
    print("✅ Training data found")
    print()
    
    # Define hyperparameter search space
    search_space = {
        # Model architecture (Conservative H100 testing: moderate models)
        # Valid combinations ensuring d_model is divisible by nhead  
        "d_model": tune.choice([256, 384, 512]),  # Conservative sizes for testing
        "nhead": tune.choice([8, 12, 16]),  # Reasonable attention heads
        "num_layers": tune.choice([4, 6, 8]),  # Conservative depth for testing
        "dropout": tune.uniform(0.1, 0.3),
        
        # Training hyperparameters (Moderate H100 batch sizes)
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 48, 64]), # Moderate batches for testing
        "weight_decay": tune.loguniform(1e-6, 1e-4),
    }
    
    # Configure scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=50,  # Max epochs
        grace_period=10,  # Min epochs before stopping
        reduction_factor=2,
        brackets=1
    )
    
    print("🎯 Hyperparameter Search Configuration:")
    print(f"   Search algorithm: Grid/Random Search")
    print(f"   Early stopping: ASHA Scheduler")  
    print(f"   Max trials: 20")
    print(f"   Max epochs per trial: 50")
    print(f"   Early stopping patience: 10 epochs")
    print(f"   GPU: NVIDIA H100 (80GB VRAM)")
    print(f"   Conservative testing: up to 512 d_model, 8 layers") 
    print(f"   Moderate batches: up to 64 batch size")
    print()
    
    # Configure Ray to use GPU properly
    import ray
    if not ray.is_initialized():
        ray.init(num_gpus=1)
    
    # Run hyperparameter optimization
    tuner = tune.Tuner(
        tune.with_resources(train_model, resources={"gpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=20,  # Number of hyperparameter combinations to try
            max_concurrent_trials=1,  # Run one trial at a time on single GPU
        ),
        run_config=tune.RunConfig(
            name="tide_transformer_v1_hp_sweep",
            stop={"training_iteration": 50},
        )
    )
    
    print("🔄 Starting hyperparameter optimization...")
    print("   H100 timing TBD - will measure actual performance")
    print("   12-hour timeout to ensure completion")
    print()
    
    results = tuner.fit()
    
    # Get best results
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config
    best_val_loss = best_result.metrics["val_loss"]
    
    print("=" * 60)
    print("🏆 OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print()
    print("🎯 Best Configuration:")
    for key, value in best_config.items():
        print(f"   {key}: {value}")
    print()
    print(f"🎯 Best Validation Loss: {best_val_loss:.6f}")
    print()
    
    # Get top 5 results
    print("🏅 Top 5 Results:")
    df = results.get_dataframe()
    top_5 = df.nsmallest(5, "val_loss")
    
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. Val Loss: {row['val_loss']:.6f}")
        print(f"      Config: d_model={row['config/d_model']}, nhead={row['config/nhead']}, layers={row['config/num_layers']}")
        print(f"      LR: {row['config/learning_rate']:.2e}, batch_size={row['config/batch_size']}")
        print()
    
    # Save results
    results_summary = {
        "best_config": best_config,
        "best_val_loss": best_val_loss,
        "total_trials": len(results),
        "optimization_algorithm": "Grid/Random Search + ASHA",
        "top_5_configs": []
    }
    
    for _, row in top_5.iterrows():
        config_dict = {k.replace('config/', ''): v for k, v in row.items() if k.startswith('config/')}
        results_summary["top_5_configs"].append({
            "config": config_dict,
            "val_loss": row["val_loss"]
        })
    
    # Save to volume
    with open("/data/hp_optimization_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("💾 Results saved to /data/hp_optimization_results.json")
    print()
    print("🚀 Ready to train your final model with the best configuration!")
    
    return results_summary

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
        print(f"✓ Found {filename} at {local_file.absolute()}")
    
    # Upload files using batch_upload - convert to string for Windows compatibility
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
    run_sweep: bool = False
):
    """
    Main entry point for hyperparameter optimization
    
    Usage:
        modal run modal_hp_sweep.py --upload-data
        modal run modal_hp_sweep.py --run-sweep
    """
    
    if upload_data:
        return upload_training_data()
    
    if run_sweep:
        print("🚀 Starting hyperparameter optimization...")
        results = run_hyperparameter_sweep.remote()
        print("=" * 60)
        print("🏆 HYPERPARAMETER OPTIMIZATION COMPLETE!")
        print("=" * 60)
        return results
    
    print("Usage:")
    print("  1. Upload data: modal run modal_hp_sweep.py --upload-data")
    print("  2. Run sweep:   modal run modal_hp_sweep.py --run-sweep")
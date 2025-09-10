#!/usr/bin/env python3
"""
Tide Transformer v1 - Modal Hyperparameter Optimization
Clean, serverless GPU hyperparameter sweep with Ray Tune integration
"""

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import math
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
    """
    High-quality PyTorch Dataset for tidal prediction with seq2seq transformer.
    Matches local training implementation exactly for consistent hyperparameter optimization.
    """
    
    def __init__(self, X, y, norm_params, split='train', augment=True, 
                 missing_prob=0.02, gap_prob=0.05, noise_std=0.1, missing_value=-999):
        """
        Initialize the dataset with full augmentation capabilities.
        
        Args:
            X: Input sequences array (num_sequences, 433)
            y: Target sequences array (num_sequences, 144)  
            norm_params: Normalization parameters dict with 'mean' and 'std'
            split: 'train' or 'val' for training/validation split
            augment: Whether to apply data augmentation (only for training)
            missing_prob: Probability of replacing individual values with missing_value
            gap_prob: Probability of creating gaps (consecutive missing values)
            noise_std: Standard deviation for value perturbation (in normalized space)
            missing_value: Value to use for simulated missing data (-999)
        """
        self.X = X
        self.y = y
        self.norm_params = norm_params
        self.split = split
        self.augment = augment and (split == 'train')  # Only augment training data
        self.missing_prob = missing_prob
        self.gap_prob = gap_prob
        self.noise_std = noise_std
        self.missing_value = missing_value
        
        if split == 'train' and augment:
            print(f"Dataset: {split} (augmented: missing={missing_prob}, gap={gap_prob}, noise={noise_std})")
        else:
            print(f"Dataset: {split} (clean data)")
        
    def __len__(self):
        return len(self.X)
    
    def apply_missing_value_augmentation(self, sequence):
        """Apply missing value augmentation to input sequence."""
        seq_len = sequence.shape[0]
        augmented = sequence.clone()
        
        # Random individual missing values
        if self.missing_prob > 0:
            missing_mask = torch.rand(seq_len) < self.missing_prob
            augmented[missing_mask] = self.missing_value
        
        # Random gaps (consecutive missing values)
        if self.gap_prob > 0 and torch.rand(1).item() < self.gap_prob:
            # Create 1-3 gaps per sequence
            num_gaps = torch.randint(1, 4, (1,)).item()
            for _ in range(num_gaps):
                # Gap length: 1-10 time steps (10-100 minutes)
                gap_length = torch.randint(1, 11, (1,)).item()
                gap_start = torch.randint(0, max(1, seq_len - gap_length + 1), (1,)).item()
                augmented[gap_start:gap_start + gap_length] = self.missing_value
        
        return augmented
    
    def apply_noise_augmentation(self, sequence):
        """Apply value perturbation to sequence (in normalized space)."""
        if self.noise_std <= 0:
            return sequence
            
        # Add Gaussian noise (only to non-missing values)
        noise = torch.normal(0, self.noise_std, sequence.shape)
        noisy_sequence = sequence + noise
        
        # Preserve missing values
        missing_mask = (sequence == self.missing_value)
        noisy_sequence[missing_mask] = self.missing_value
        
        return noisy_sequence
    
    def __getitem__(self, idx):
        """Get a single sequence pair with optional augmentation."""
        # Convert to tensors and add feature dimension
        src = torch.from_numpy(self.X[idx]).float().unsqueeze(-1)  # (433, 1)
        tgt = torch.from_numpy(self.y[idx]).float().unsqueeze(-1)  # (144, 1)
        
        # Apply augmentation to input sequence only (during training)
        if self.augment:
            # Apply noise perturbation first
            src = self.apply_noise_augmentation(src)
            
            # Then apply missing value simulation
            src = self.apply_missing_value_augmentation(src)
        
        return src, tgt

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer sequences.
    Adds positional information to embeddings without learnable parameters.
    """
    
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input embeddings
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class TidalTransformer(nn.Module):
    """
    Sequence-to-sequence transformer for tidal prediction with 10-minute intervals.
    
    Architecture:
    - Configurable encoder/decoder layers
    - Configurable attention heads and dimensions  
    - Input: 433 time steps (72 hours at 10-minute intervals)
    - Output: 144 time steps (24 hours at 10-minute intervals)
    """
    
    def __init__(self, 
                 input_dim=1,
                 d_model=512,
                 nhead=16, 
                 num_encoder_layers=3,
                 num_decoder_layers=1,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_seq_length=5000):
        super(TidalTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        
        # Input projection: map from input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Output projection: map from d_model back to input_dim
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer architecture
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch_size, seq_len, d_model) for better performance
        )
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.transformer, 'enable_nested_tensor'):
            self.transformer.enable_nested_tensor = False
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def generate_square_subsequent_mask(self, sz: int):
        """Generate causal mask for decoder to prevent looking at future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0):
        """
        Forward pass through the transformer.
        
        Args:
            src: Source sequence (batch_size, input_seq_len, input_dim)
            tgt: Target sequence for teacher forcing (batch_size, output_seq_len, input_dim)
            teacher_forcing_ratio: Probability of using teacher forcing vs autoregressive
            
        Returns:
            output: Predicted sequence (batch_size, output_seq_len, input_dim)
        """
        batch_size, src_seq_len, _ = src.shape
        
        # Project input to model dimension (batch_first=True, no transpose needed)
        src = self.input_projection(src)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding (need to transpose for pos_encoder, then back)
        src = src.transpose(0, 1)         # (seq_len, batch_size, d_model) for pos_encoder
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)         # (batch_size, seq_len, d_model) for transformer

        if self.training and tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Training with teacher forcing
            tgt = self.input_projection(tgt)  # (batch_size, seq_len, d_model)
            
            # Add positional encoding (transpose for pos_encoder, then back)
            tgt = tgt.transpose(0, 1)         # (seq_len, batch_size, d_model) for pos_encoder
            tgt = self.pos_encoder(tgt)
            tgt = tgt.transpose(0, 1)         # (batch_size, seq_len, d_model) for transformer
            
            # Create causal mask for decoder
            tgt_seq_len = tgt.size(1)  # seq_len dimension is now index 1
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
            
            # Transformer forward pass
            output = self.transformer(src, tgt, tgt_mask=tgt_mask)
            
        else:
            # Inference or training without teacher forcing (autoregressive)
            output_seq_len = 144  # 24 hours at 10-minute intervals
            
            # Start with encoder output
            memory = self.transformer.encoder(src)
            
            # Memory-efficient autoregressive generation with sliding window
            max_context = 72  # Keep only last 72 tokens for context
            decoder_input = torch.zeros(batch_size, 1, self.d_model).to(src.device)  # (batch_size, 1, d_model)
            outputs = []
            
            # Autoregressive generation
            for i in range(output_seq_len):
                # Always slice to max_context to avoid conditional logic during tracing
                # This is safe even when current_len <= max_context
                decoder_input = decoder_input[:, -max_context:, :]
                current_len = decoder_input.size(1)
                
                # Use fixed window size to avoid dynamic operations during tracing
                # Use current_len directly since it's now bounded by max_context
                tgt_mask = self.generate_square_subsequent_mask(current_len).to(src.device)
                
                # Decoder forward pass with full decoder input (already windowed above)
                decoder_output = self.transformer.decoder(
                    decoder_input, memory, tgt_mask=tgt_mask
                )
                
                # Take the last output token
                current_output = decoder_output[:, -1:, :]  # (batch_size, 1, d_model)
                outputs.append(current_output.clone())  # Clone to avoid memory references
                
                # Append current output to decoder input
                decoder_input = torch.cat([decoder_input, current_output], dim=1)  # Concat along seq_len
                
                # Clear cache periodically during inference
                if i % 100 == 0 and not torch.jit.is_tracing():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Concatenate all outputs
            output = torch.cat(outputs, dim=1)  # (batch_size, output_seq_len, d_model)
        
        # Project back to input dimension (already in batch_first format)
        output = self.output_projection(output)  # (batch_size, seq_len, input_dim)
        
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
    
    # Add augmentation parameters to config if not present (for compatibility)
    if 'augment' not in config:
        config['augment'] = True
        config['missing_prob'] = 0.02
        config['gap_prob'] = 0.05
        config['noise_std'] = 0.1
    
    # Create high-quality datasets with full augmentation
    train_dataset = TidalDataset(
        X_train, y_train, norm_params,
        split='train',
        augment=config['augment'],
        missing_prob=config['missing_prob'],
        gap_prob=config['gap_prob'],
        noise_std=config['noise_std']
    )
    val_dataset = TidalDataset(
        X_val, y_val, norm_params,
        split='val',
        augment=False  # No augmentation during validation
    )
    
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
    
    # Split layers for seq2seq (encoder gets more layers)
    num_encoder_layers = max(1, (config["num_layers"] * 2) // 3)  # ~67% to encoder
    num_decoder_layers = max(1, config["num_layers"] - num_encoder_layers)  # remainder to decoder
    
    # Initialize seq2seq model
    model = TidalTransformer(
        input_dim=1,
        d_model=config["d_model"],
        nhead=config["nhead"], 
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=config["d_model"] * 4,
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
        optimizer, T_max=50, eta_min=1e-6  # Match max epochs
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # Production early stopping patience
    
    for epoch in range(50):  # Max epochs (production training)
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Use teacher forcing during training (80% chance)
            # batch_x and batch_y already have shape (batch_size, seq_len, 1) from dataset
            outputs = model(batch_x, batch_y, teacher_forcing_ratio=0.8)
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
                
                # No teacher forcing during validation
                # batch_x and batch_y already have shape (batch_size, seq_len, 1) from dataset
                outputs = model(batch_x)
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
    
    # Define hyperparameter search space (FOCUSED OPTIMIZATION)
    search_space = {
        # Fixed model architecture (reasonable size for cost/performance balance)
        "d_model": 512,        # Good balance of performance and cost
        "num_layers": 8,       # Deep enough to be effective
        
        # OPTIMIZE THE UNKNOWN: Attention head configuration
        "nhead": tune.choice([8, 16, 32]),  # Key question: optimal attention heads
        
        # Training hyperparameters (genuinely need optimization)
        "learning_rate": tune.loguniform(1e-5, 5e-4),    # Critical parameter
        "batch_size": tune.choice([32, 48, 64, 96]),      # H100 batch optimization
        "weight_decay": tune.loguniform(1e-6, 1e-3),     # Regularization balance
        "dropout": tune.uniform(0.05, 0.3),              # Overfitting control
        
        # Data augmentation (affects real-world robustness)
        "augment": tune.choice([True]),  # Always use augmentation
        "missing_prob": tune.uniform(0.01, 0.08),        # Sensor dropout simulation
        "gap_prob": tune.uniform(0.02, 0.12),            # Connectivity gap simulation
        "noise_std": tune.uniform(0.05, 0.25),           # Measurement noise robustness
    }
    
    # Configure scheduler for early stopping (production settings)
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=50,  # Max epochs per trial (production training)
        grace_period=15,   # Min epochs before stopping
        reduction_factor=3,  # More aggressive pruning for efficiency
        brackets=1
    )
    
    print("🎯 Hyperparameter Search Configuration (FOCUSED OPTIMIZATION):")
    print(f"   Search algorithm: Random Search + ASHA Early Stopping")
    print(f"   Max trials: 30 (focused on unknowns)")
    print(f"   Max epochs per trial: 50")
    print(f"   Early stopping grace period: 15 epochs")
    print(f"   GPU: NVIDIA H100 (80GB VRAM)")
    print(f"   Fixed architecture: d_model=512, layers=8")
    print(f"   KEY QUESTION: Optimal attention heads [8,16,32]")
    print(f"   Tuning: LR, batch size, regularization, data augmentation")
    print(f"   Focus: Answer 'what matters' not 'bigger is better'")
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
            num_samples=30,  # Focused search (30 trials)
            max_concurrent_trials=1,  # Run one trial at a time on single GPU
        ),
        run_config=tune.RunConfig(
            name="tide_transformer_v1_hp_sweep",
            stop={"training_iteration": 50},  # Match max epochs
        )
    )
    
    print("🔄 Starting focused hyperparameter optimization...")
    print("   30 trials × avg 25 epochs = ~3-4 days total")
    print("   Cost estimate: ~$300-600 (vs $2000+ for unfocused search)")
    print("   Answers the real question: optimal attention head configuration")
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

# Note: Functions can be called directly via modal run file.py::function_name
# No main entrypoint needed - Modal will call functions directly
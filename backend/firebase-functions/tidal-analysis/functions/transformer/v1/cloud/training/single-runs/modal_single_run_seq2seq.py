#!/usr/bin/env python3
"""
Tide Transformer v1 - Modal Single Training Run (Seq2Seq Architecture)
Single high-performance GPU training run using optimal hyperparameters with seq2seq transformer
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

# Modal app definition
app = modal.App("tide-transformer-v1-seq2seq-single-run")

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
        
        print(f"Dataset initialized: {split}")
        print(f"  Sequences: {len(self.X)}")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Target shape: {self.y.shape}")
        print(f"  Normalization: mean={self.norm_params['mean']:.2f}, std={self.norm_params['std']:.2f}")
        if self.augment:
            print(f"  Augmentation enabled:")
            print(f"    Missing prob: {self.missing_prob}")
            print(f"    Gap prob: {self.gap_prob}")
            print(f"    Noise std: {self.noise_std}")
        
    def __len__(self):
        return len(self.X)
    
    def apply_missing_value_augmentation(self, sequence):
        """
        Apply missing value augmentation to input sequence.
        
        Args:
            sequence: Input sequence tensor (seq_len, 1)
            
        Returns:
            Augmented sequence with simulated missing values
        """
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
        """
        Apply value perturbation to sequence (in normalized space).
        
        Args:
            sequence: Input sequence tensor (seq_len, 1)
            
        Returns:
            Sequence with added noise
        """
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
        """
        Get a single sequence pair with optional augmentation.
        
        Returns:
            src: Input sequence tensor (433, 1) - possibly augmented
            tgt: Target sequence tensor (144, 1) - always clean
        """
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

@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": volume},
    timeout=18000,  # 5 hours max for single run
    memory=32768,  # 32GB RAM for H100
)
def run_single_training():
    """Single training run with optimal hyperparameters and seq2seq architecture"""
    
    print("🚀 Starting Tide Transformer v1 Single Training Run (Seq2Seq)")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🚀 Using H100 GPU for high-performance seq2seq training!")
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
    
    # Optimal hyperparameters from sweep (adapted for seq2seq)
    config = {
        "d_model": 512,
        "nhead": 16,
        "num_layers": 4,  # Split between encoder/decoder
        "dropout": 0.17305796046740565,
        "learning_rate": 5.845165970318201e-05,
        "batch_size": 32,
        "weight_decay": 2.0160787083950938e-06,
        "num_epochs": 150,  # More epochs for single run
        
        # Data augmentation parameters (match local training exactly)
        "augment": True,
        "missing_prob": 0.02,  # 2% chance of individual missing values
        "gap_prob": 0.05,      # 5% chance of creating gaps per sequence
        "noise_std": 0.1,      # Noise standard deviation in normalized space
    }
    
    # Split layers for seq2seq (encoder gets more layers)
    num_encoder_layers = max(1, (config["num_layers"] * 2) // 3)  # ~67% to encoder
    num_decoder_layers = max(1, config["num_layers"] - num_encoder_layers)  # remainder to decoder
    
    print("🎯 Optimal Configuration (Seq2Seq):")
    print(f"   d_model: {config['d_model']}, nhead: {config['nhead']}")
    print(f"   Encoder layers: {num_encoder_layers}, Decoder layers: {num_decoder_layers}")
    print(f"   Total layers: {config['num_layers']}")
    print(f"   Learning Rate: {config['learning_rate']:.2e}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Weight Decay: {config['weight_decay']:.2e}")
    print(f"   Dropout: {config['dropout']:.3f}")
    print(f"   Epochs: {config['num_epochs']}")
    print()
    
    # Create high-quality datasets with full augmentation
    train_dataset = TidalDataset(
        X_train, y_train, norm_params, 
        split='train', 
        augment=config["augment"],
        missing_prob=config["missing_prob"],
        gap_prob=config["gap_prob"], 
        noise_std=config["noise_std"]
    )
    val_dataset = TidalDataset(
        X_val, y_val, norm_params,
        split='val', 
        augment=False  # No augmentation during validation
    )
    
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  Model Architecture:")
    print(f"   Architecture: Sequence-to-Sequence Transformer")
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
        
        # Check for improvement
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model to volume (seq2seq format)
            model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,  # Use 'loss' for compatibility with inference
                'config': {
                    **config,
                    'num_encoder_layers': num_encoder_layers,
                    'num_decoder_layers': num_decoder_layers,
                    'architecture': 'seq2seq_transformer'
                },
                'normalization_params': norm_params
            }
            torch.save(model_state, '/data/best_seq2seq_single_run.pth')
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
    print(f"📁 Model saved to: /data/best_seq2seq_single_run.pth")
    print()
    
    # Save training summary
    summary = {
        "best_val_loss": best_val_loss,
        "total_epochs": epoch + 1,
        "config": config,
        "architecture": "seq2seq_transformer",
        "encoder_layers": num_encoder_layers,
        "decoder_layers": num_decoder_layers,
        "model_parameters": total_params,
        "final_lr": optimizer.param_groups[0]['lr']
    }
    
    with open("/data/seq2seq_single_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("💾 Training summary saved to /data/seq2seq_single_run_summary.json")
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

# Note: Functions can be called directly via modal run file.py::function_name
# No main entrypoint needed - Modal will call functions directly
import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("tidal-transformer-v2")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "torchvision",
        "torchaudio",
        "numpy>=1.24.0",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "tqdm",
        "tensorboard",
        "wandb",  # For experiment tracking (optional)
    ])
    .apt_install(["git", "wget", "curl"])
)

# Define persistent storage volume for model outputs
volume = modal.Volume.from_name("transformer-v2-storage", create_if_missing=True)

# Define H100 GPU configuration
GPU_CONFIG = "H100"

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/output": volume},
    timeout=3600 * 6,  # 6 hour timeout
    memory=32768,  # 32GB RAM
    retries=2
)
def train_transformer_v2(
    train_data_x,
    train_data_y,
    val_data_x,
    val_data_y,
    config
):
    """
    Main training function that runs on Modal with H100 GPU.
    
    Args:
        train_data_x: Training input sequences (NumPy array)
        train_data_y: Training target sequences (NumPy array) 
        val_data_x: Validation input sequences (NumPy array)
        val_data_y: Validation target sequences (NumPy array)
        config: Training configuration dictionary
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import json
    import time
    import math
    from datetime import datetime
    import os
    
    print("=" * 60)
    print("TIDAL TRANSFORMER V2 TRAINING ON MODAL")
    print("=" * 60)
    
    # Print system info
    print(f"Python version: {__import__('sys').version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Save training data to local files for the training script
    os.makedirs('/data', exist_ok=True)
    np.save('/data/X_train.npy', train_data_x)
    np.save('/data/y_train.npy', train_data_y)
    np.save('/data/X_val.npy', val_data_x)
    np.save('/data/y_val.npy', val_data_y)
    
    print(f"Saved training data:")
    print(f"  X_train: {train_data_x.shape}")
    print(f"  y_train: {train_data_y.shape}")
    print(f"  X_val: {val_data_x.shape}")
    print(f"  y_val: {val_data_y.shape}")
    print()
    
    # Model definition (embedded in function to run on Modal)
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            return x + self.pe[:x.size(0), :]

    class TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, src, src_mask=None, src_key_padding_mask=None):
            src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            return src

    class TidalTransformerV2(nn.Module):
        def __init__(
            self,
            input_length=432,
            output_length=144,
            d_model=512,
            nhead=16,
            num_encoder_layers=8,
            dim_feedforward=2048,
            dropout=0.1
        ):
            super().__init__()
            
            self.input_length = input_length
            self.output_length = output_length
            self.d_model = d_model
            
            self.input_projection = nn.Linear(1, d_model)
            self.pos_encoder = PositionalEncoding(d_model, max_len=input_length + 50)
            
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ])
            
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, output_length),
                nn.Linear(output_length, output_length)
            )
            
            self.init_weights()
        
        def init_weights(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        def forward(self, src):
            batch_size, seq_len, feature_dim = src.shape
            assert seq_len == self.input_length
            assert feature_dim == 1
            
            src = self.input_projection(src)
            src = src.transpose(0, 1)
            src = self.pos_encoder(src)
            src = src.transpose(0, 1)
            
            for layer in self.encoder_layers:
                src = layer(src)
            
            encoded = src.mean(dim=1)
            predictions = self.output_projection(encoded)
            
            return predictions
    
    # Dataset class
    class TidalDataset(Dataset):
        def __init__(self, X, y):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            x = torch.from_numpy(self.X[idx]).unsqueeze(-1)
            y = torch.from_numpy(self.y[idx])
            return x, y
    
    # Create model
    print("Creating model...")
    model = TidalTransformerV2(**config['model_config'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    # Create datasets and loaders
    print("Creating datasets...")
    train_dataset = TidalDataset(train_data_x, train_data_y)
    val_dataset = TidalDataset(val_data_x, val_data_y)
    
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
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    
    # Training state
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    learning_rates = []
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        total_train_loss = 0.0
        num_train_batches = len(train_loader)
        
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % config['log_interval'] == 0:
                progress = batch_idx / num_train_batches * 100
                print(f'Epoch {epoch:3d} [{batch_idx:4d}/{num_train_batches:4d} ({progress:5.1f}%)] '
                      f'Loss: {loss.item():.6f}')
        
        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        num_val_batches = len(val_loader)
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        scheduler.step()
        
        # Check for best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        epoch_time = time.time() - epoch_start
        rmse = math.sqrt(avg_val_loss)
        
        print(f'Epoch {epoch:3d}: Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, '
              f'RMSE: {rmse:.6f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s')
        
        if is_best:
            print(f'*** New best validation loss: {best_val_loss:.6f} ***')
        
        # Save checkpoint every 10 epochs or if best
        if epoch % 10 == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'config': config
            }
            
            checkpoint_path = f'/output/checkpoint_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                torch.save(checkpoint, '/output/best_model.pth')
        
        print('-' * 60)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best RMSE: {math.sqrt(best_val_loss):.6f}")
    
    # Save final model and training log
    final_checkpoint = {
        'epoch': config['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'config': config
    }
    
    torch.save(final_checkpoint, '/output/final_model.pth')
    
    # Save training log
    training_log = {
        'config': config,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates
        },
        'final_metrics': {
            'best_val_loss': best_val_loss,
            'best_rmse': math.sqrt(best_val_loss),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'total_training_time': total_time
        },
        'model_info': {
            'total_parameters': total_params,
            'input_length': config['model_config']['input_length'],
            'output_length': config['model_config']['output_length']
        },
        'training_completed': datetime.now().isoformat()
    }
    
    with open('/output/training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("Training artifacts saved to /output/")
    print("Files created:")
    for file in os.listdir('/output'):
        size = os.path.getsize(f'/output/{file}') / 1024 / 1024
        print(f"  {file} ({size:.1f} MB)")
    
    return {
        'best_val_loss': best_val_loss,
        'best_rmse': math.sqrt(best_val_loss),
        'total_training_time': total_time,
        'epochs_completed': config['epochs']
    }

@app.local_entrypoint()
def main(x_train_path: str, y_train_path: str, x_val_path: str, y_val_path: str, config_path: str):
    """
    Local entrypoint to run training with data loaded from files.
    """
    import numpy as np
    import json
    
    print("Loading training data from files...")
    
    # Load data arrays
    train_data_x = np.load(x_train_path)
    train_data_y = np.load(y_train_path)
    val_data_x = np.load(x_val_path)
    val_data_y = np.load(y_val_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded data shapes:")
    print(f"  Train X: {train_data_x.shape}")
    print(f"  Train y: {train_data_y.shape}")
    print(f"  Val X: {val_data_x.shape}")
    print(f"  Val y: {val_data_y.shape}")
    
    # Call the training function
    result = train_transformer_v2.remote(
        train_data_x,
        train_data_y,
        val_data_x,
        val_data_y,
        config
    )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {result['best_val_loss']:.6f}")
    print(f"Best RMSE: {result['best_rmse']:.6f}")
    print(f"Training time: {result['total_training_time']:.1f} seconds")
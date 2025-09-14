# Transformer v2 Training

Complete training pipeline for TidalTransformerV2 using Modal cloud infrastructure with H100 GPU.

## Overview

This training system uses Modal to train a transformer model on prepared tidal data. The model architecture is an encoder-only transformer designed for single-pass tidal prediction.

**Model Specifications:**
- **Architecture**: Encoder-only transformer (no autoregressive generation)
- **Input**: 432 time steps (72 hours at 10-minute intervals)
- **Output**: 144 time steps (24 hours at 10-minute intervals)  
- **Dimensions**: 512 model dimension, 16 attention heads, 8 encoder layers
- **Parameters**: ~12M trainable parameters
- **Hardware**: NVIDIA H100 GPU on Modal cloud

## Files

### Scripts
- `login.ps1` - Authenticate with Modal service
- `setup.ps1` - Deploy Modal app and dependencies  
- `train.ps1` - Execute training run
- `run_training.py` - Python training execution script

### Training Code
- `modal_setup.py` - Modal app definition with H100 configuration
- `model.py` - Transformer architecture implementation  
- `train.py` - Local training script (for reference)

## Quick Start

### Prerequisites
1. **Training data prepared**: Run data preparation pipeline first
   ```bash
   cd ../data-preparation
   ./run-data-preparation.ps1
   ```

2. **Modal account**: Sign up at [modal.com](https://modal.com)

3. **Python dependencies**: 
   ```bash
   pip install modal numpy torch
   ```

### Training Workflow

**Step 1: Login to Modal**
```powershell
.\login.ps1
```
- Installs Modal if needed
- Opens browser for authentication  
- Saves authentication token

**Step 2: Setup Modal Environment**
```powershell
.\setup.ps1
```
- Verifies training data exists
- Deploys Modal app with H100 configuration
- Installs PyTorch and dependencies in cloud environment

**Step 3: Execute Training**
```powershell
.\train.ps1
```
- Loads 12k+ training sequences (~32MB)
- Uploads data to Modal cloud
- Trains for 50 epochs on H100 GPU (~30-60 minutes)
- Saves trained model to Modal volume

## Training Configuration

### Default Settings
```python
config = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'log_interval': 50
}
```

### Model Architecture
```python
model_config = {
    'input_length': 432,
    'output_length': 144, 
    'd_model': 512,
    'nhead': 16,
    'num_encoder_layers': 8,
    'dim_feedforward': 2048,
    'dropout': 0.1
}
```

## Training Process

### Data Pipeline
1. **Input**: 432 normalized water level readings (72 hours)
2. **Target**: 144 normalized water level predictions (24 hours)
3. **Batch Size**: 32 sequences per batch
4. **Normalization**: Z-score normalization (mean=0, std=1)

### Training Loop  
1. **Forward Pass**: Single-pass encoder prediction
2. **Loss Function**: Mean Squared Error (MSE)
3. **Optimizer**: AdamW with weight decay
4. **Scheduler**: Cosine annealing learning rate
5. **Gradient Clipping**: Max norm 1.0
6. **Validation**: Every epoch with best model saving

### Output Files
Training creates these files in Modal volume `transformer-v2-storage`:

- `best_model.pth` - Best validation loss model
- `final_model.pth` - Final epoch model  
- `training_log.json` - Metrics and configuration
- `checkpoint_epoch_*.pth` - Periodic checkpoints

## Model Download

After training, download the trained model:

```bash
# List available files
python -m modal volume ls transformer-v2-storage

# Download best model
python -m modal volume get transformer-v2-storage best_model.pth ./models/

# Download training log
python -m modal volume get transformer-v2-storage training_log.json ./logs/
```

## Expected Performance

### Baseline Metrics
- **Training Time**: 30-60 minutes on H100
- **Target RMSE**: < 50mm (normalized units < 0.15)
- **Memory Usage**: ~24GB GPU memory
- **Dataset**: 12,164 training sequences, 1,967 validation sequences

### Training Progress
```
Epoch  50: Train: 0.012345, Val: 0.015678, RMSE: 0.125, LR: 1.23e-06
*** New best validation loss: 0.015432 ***
```

## Troubleshooting

### Common Issues

**Modal Authentication Failed**
```
ERROR: Not logged into Modal
```
**Solution**: Run `.\login.ps1` and complete browser authentication

**Missing Training Data**  
```
ERROR: Missing required data files
```
**Solution**: Run data preparation pipeline first:
```bash
cd ../data-preparation  
.\run-data-preparation.ps1
```

**Modal App Not Found**
```
ERROR: Modal app not found
```
**Solution**: Run `.\setup.ps1` to deploy the app

**GPU Memory Error**
```
CUDA out of memory
```  
**Solution**: Reduce batch size in `run_training.py` config

### Performance Tips
- **Batch Size**: Increase to 64 for faster training (if memory allows)
- **Epochs**: Reduce to 25 for quick baseline, increase to 100 for best results
- **Learning Rate**: Try 5e-4 for faster convergence, 5e-5 for stability

## Advanced Usage

### Custom Configuration
Edit `run_training.py` to modify training parameters:

```python
config = {
    'batch_size': 64,        # Larger batches
    'epochs': 100,           # More training
    'learning_rate': 5e-4,   # Faster learning
}
```

### Model Architecture Changes
Edit `model_config` in `run_training.py`:

```python
model_config = {
    'd_model': 768,          # Larger model
    'num_encoder_layers': 12, # Deeper network
    'nhead': 24,             # More attention heads
}
```

### Monitoring Training
Training progress is displayed in real-time. Look for:
- **Decreasing train/validation loss**
- **RMSE < 0.15** (good performance)
- **Learning rate decay** (cosine schedule)
- **Best model updates** (validation improvements)

## Next Steps

After successful training:
1. **Evaluate Model**: Test on holdout data
2. **Deploy Inference**: Create Firebase Function for predictions  
3. **Monitor Performance**: Compare against actual tidal measurements
4. **Iterate**: Adjust architecture based on results

This training pipeline provides a robust baseline for tidal prediction using state-of-the-art transformer architecture on professional-grade hardware.
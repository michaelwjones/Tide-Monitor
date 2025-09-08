# Cloud Training with Weights & Biases

This directory contains files for training the Transformer v1 model in the cloud using Weights & Biases.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Login to W&B**:
   ```bash
   wandb login
   ```

3. **Upload your data** (one-time setup):
   ```bash
   python setup_cloud.py
   ```

4. **Start cloud training**:
   ```bash
   wandb job create --runtime=3.9 --resource=gpu train_wandb.py
   ```

## Files

- `train_wandb.py` - Main training script with W&B integration
- `requirements.txt` - Python dependencies for cloud environment
- `wandb_config.yaml` - Hyperparameter sweep configuration
- `setup_cloud.py` - Upload data and setup artifacts
- `README.md` - This file

## Hyperparameter Sweeps

Run automated hyperparameter tuning:

```bash
wandb sweep wandb_config.yaml
wandb agent <sweep-id>
```

## Firebase Integration

To access Firebase data in the cloud:

1. Download your Firebase service account key
2. Upload it as a W&B secret (see setup_cloud.py instructions)
3. Update the credentials path in train_wandb.py

## Model Artifacts

Trained models are automatically saved as W&B artifacts and can be:
- Downloaded for local testing
- Deployed directly to Firebase Functions
- Compared across training runs

## Critical Training Fixes

**Before using cloud training, ensure these fixes are applied:**

1. **Normalization Contamination Fix**:
   - Exclude -999 synthetic values from mean/std calculation
   - Update `create_training_data.py` with proper filtering

2. **Data Leakage Prevention**:
   - Implement timestamp-based train/validation split
   - Ensure temporal gap between overlapping sequences

3. **Hyperparameter Updates**:
   - Batch size reduced to 4 for GPU memory constraints
   - Consider starting with smaller models for cloud resource limits

## GPU Resources

W&B provides free GPU hours. For longer training:
- Upgrade to W&B Pro
- Use W&B Launch with your cloud provider  
- Consider Google Colab Pro integration

**Resource Planning**:
- Free tier: ~100 hours/month (sufficient for initial experiments)
- Batch size 2-4 recommended for T4 instances
- Monitor memory usage during hyperparameter sweeps
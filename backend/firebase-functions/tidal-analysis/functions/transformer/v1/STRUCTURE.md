# Transformer v1 - Clean Folder Structure

This document describes the organized folder structure for Transformer v1 tidal prediction system.

## Folder Structure

```
transformer/v1/
├── local/                              # Local development and training
│   ├── data-preparation/               # Data fetching and preprocessing
│   │   ├── fetch_firebase_data.py      # Download raw Firebase data
│   │   ├── create_training_data.py     # Generate training sequences
│   │   └── data/                       # Generated training data
│   │       ├── X_train.npy             # Training input sequences (11,638 sequences)
│   │       ├── y_train.npy             # Training target sequences
│   │       ├── X_val.npy               # Validation input sequences (1,743 sequences)
│   │       ├── y_val.npy               # Validation target sequences
│   │       ├── normalization_params.json # Data normalization parameters
│   │       └── metadata.json           # Dataset metadata
│   ├── training/                       # Local model training
│   │   ├── train_transformer.py        # Main training script (auto-detects GPU)
│   │   ├── dataset.py                  # PyTorch dataset implementation
│   │   ├── model_server.py             # Local model testing server
│   │   ├── checkpoints/                # Saved model checkpoints
│   │   │   ├── best.pth                # Best validation loss model
│   │   │   └── latest.pth              # Most recent checkpoint
│   │   └── runs/                       # TensorBoard training logs
│   ├── testing/                        # Local model validation
│   │   ├── start-server.bat            # Web interface launcher
│   │   ├── server.py                   # HTTP server for testing
│   │   ├── test_model.py               # Command-line testing script
│   │   ├── index.html                  # Web testing interface
│   │   └── firebase_fetch.py           # Firebase data utilities
│   ├── train-local.bat                 # Simple local training launcher
│   ├── install-pytorch-transformer.bat # PyTorch installation script
│   ├── setup-complete-transformer-v1.bat # Complete local setup
│   └── requirements.txt                # Local Python dependencies
├── cloud/                              # Cloud deployment components
│   ├── inference/                      # Firebase Functions deployment
│   │   ├── main.py                     # Firebase Functions entry point
│   │   ├── model.py                    # Transformer model definition
│   │   ├── deploy-transformer-v1.bat   # Firebase deployment script
│   │   ├── firebase.json               # Firebase configuration
│   │   ├── requirements.txt            # Cloud Functions dependencies
│   │   └── best.pth                    # Deployed model checkpoint
│   └── training/                       # Cloud training components
│       ├── sweeps/                     # Hyperparameter optimization
│       │   ├── modal_hp_sweep.py       # Modal Labs sweep script (H100 GPU)
│       │   ├── setup.ps1               # Cloud setup script
│       │   ├── run.ps1                 # Cloud execution script
│       │   ├── requirements.txt        # Modal dependencies
│       │   └── historic/               # Historical sweep results
│       │       └── run 1/              # First H100 sweep results
│       └── single-runs/                # Individual cloud training runs
│           ├── modal_single_run.py     # Single training run script (H100 GPU)
│           ├── setup.ps1               # Single run setup
│           ├── run.ps1                 # Single run execution
│           └── requirements.txt        # Single run dependencies
└── README.md                           # Main documentation
```

## Usage Guide

### Local Development

#### 1. Data Preparation
```bash
cd local/data-preparation
python fetch_firebase_data.py        # Download from Firebase
python create_training_data.py       # Generate training sequences
```

#### 2. Local Training (Auto-GPU Detection)
```bash
cd local
train-local.bat                      # Simple launcher (auto-detects GPU)
# OR
cd local/training
python train_transformer.py          # Direct training script
```

**GPU Optimizations:**
- **GTX 1070 (8GB)**: batch_size=16, expected val_loss ~0.037-0.040
- **Other GPUs (>10GB)**: batch_size=32, expected val_loss ~0.0365
- **CPU**: batch_size=8, mixed precision disabled

#### 3. Local Testing
```bash
cd local/testing
start-server.bat                     # Web interface at http://localhost:8000
# OR
python test_model.py                 # Command line interface
```

### Cloud Operations

#### 1. Hyperparameter Optimization (Modal H100)
```powershell
cd cloud/training/sweeps
.\setup.ps1                          # First time setup + data upload
.\run.ps1                            # Run 20-trial optimization (~$15-25)
```

#### 2. Single Cloud Training Run (Modal H100)
```powershell
cd cloud/training/single-runs
.\setup.ps1                          # First time setup + data upload  
.\run.ps1                            # Single training run (~$15-25, 2-4 hours)
```

#### 3. Firebase Functions Deployment
```batch
cd cloud/inference
deploy-transformer-v1.bat            # Deploy to Firebase
```

## Path Dependencies

All relative paths have been updated for the new structure:

- **local/training/train_transformer.py** imports model from `../../cloud/inference/`
- **local/testing/** scripts import model from `../../cloud/inference/`
- **cloud/training/sweeps/modal_hp_sweep.py** uploads data from `../../../local/data-preparation/data/`
- **cloud/training/single-runs/modal_single_run.py** uploads data from `../../../local/data-preparation/data/`
- **cloud/inference/deploy-transformer-v1.bat** copies model from `../../local/training/checkpoints/`

## Key Features

- **Clean Separation**: Local development separate from cloud deployment
- **GPU Auto-Detection**: Local training automatically optimizes for your hardware
- **Multiple Cloud Options**: Both hyperparameter sweeps and single training runs
- **Relative Paths**: All scripts use relative paths for portability
- **Historic Tracking**: Sweep results preserved in historic/ folders
- **Optimized Configuration**: Uses best hyperparameters from H100 sweep
- **Comprehensive Testing**: Both command-line and web interfaces for validation
- **Ready to Execute**: All scripts are verified and ready to run

## Quick Start

1. **Local Training**: `local/train-local.bat` (auto-detects GTX 1070)
2. **Cloud Sweep**: `cloud/training/sweeps/setup.ps1` then `run.ps1`
3. **Single Cloud Run**: `cloud/training/single-runs/setup.ps1` then `run.ps1`
4. **Model Testing**: `local/testing/start-server.bat`
5. **Firebase Deploy**: `cloud/inference/deploy-transformer-v1.bat`

All scripts include error checking, progress reporting, and clear next-step instructions.
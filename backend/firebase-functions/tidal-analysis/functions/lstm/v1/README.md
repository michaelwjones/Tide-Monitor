# LSTM v1 Tidal Analysis - Iterative 24-Hour Forecasting

Long Short-Term Memory neural network for 24-hour water level forecasting using iterative prediction with historical tide monitor data.

## Overview

This LSTM system generates **24-hour water level forecasts** using an iterative prediction approach:

1. **Start with 72 hours** of real historical water level data
2. **Predict next minute** using LSTM trained on historical patterns  
3. **Add prediction** to the sequence and repeat for 1,440 iterations
4. **Generate complete 24-hour forecast** (1,440 future data points)

## Data Range

Training data spans from **June 30, 2025** to **August 27, 2025** (~2 months of readings).

## Architecture

```
Historical Data (72h) � LSTM � Next Minute � Add to Sequence � Repeat 1,440x
     4,320 samples         �         �                �
                    Single Prediction  �
                           �
                   24-Hour Forecast
                   (1,440 predictions)
```

## Setup

**Option 1 - Automatic Installation (Recommended):**
```bash
install-pytorch-automatic.bat
```
*Automatically tries CUDA-enabled PyTorch, falls back to CPU if needed*

**Option 2 - Interactive Installation:**
```bash
install-pytorch-interactive.bat
```
*Menu-driven installer with specific CUDA version choices*

**Option 3 - Manual Installation:**
```bash
pip install -r requirements.txt
```
*Direct pip installation using requirements.txt*

**GPU Training:**
- **Automatic Detection**: Training script automatically uses GPU if available
- **Performance**: 5-10x faster training with CUDA-compatible GPUs  
- **Memory**: Larger batch sizes (64 vs 32) for improved training efficiency
- **Fallback**: Seamlessly falls back to CPU if no GPU detected

**Troubleshooting:**
- **Installation Errors**: Use `install-pytorch-interactive.bat` for step-by-step installation
- **CUDA Issues**: Run `python check_gpu.py` to verify GPU setup
- **Version Conflicts**: Try default PyTorch installation: `pip install torch`

**Then proceed with:**
1. Use data preparation scripts to fetch and prepare training data
2. Train model locally with PyTorch (single-step ahead prediction)
3. Convert to ONNX and deploy to Firebase Functions for iterative forecasting

## Important Note

**The `data-preparation/data/` folder should NEVER be committed to git** - it contains large training files generated locally. The `.gitignore` file ensures this folder is ignored.
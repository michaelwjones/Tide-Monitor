# Transformer v1 - Sequence-to-Sequence Tidal Prediction

Advanced sequence-to-sequence transformer model for 24-hour tidal prediction using multi-head attention mechanisms.

## Overview

This implementation uses a modern transformer architecture to predict tidal patterns, offering significant advantages over traditional approaches:

- **Direct Prediction**: Single forward pass generates full 24-hour forecast (1440 predictions)
- **Attention Mechanisms**: Multi-head attention captures complex temporal dependencies
- **Parallel Processing**: Efficient computation compared to iterative methods
- **Long-Range Dependencies**: Superior handling of extended temporal patterns

## Architecture Details

### Model Specifications
- **Architecture**: Sequence-to-sequence transformer
- **Encoder**: 6 layers, 8 attention heads, 256 hidden dimensions
- **Decoder**: 3 layers, 8 attention heads, 256 hidden dimensions
- **Input Length**: 4320 time steps (72 hours at 1-minute intervals)
- **Output Length**: 1440 time steps (24 hours at 1-minute intervals)
- **Parameters**: ~2.5M trainable parameters

### Key Features
- **Multi-Head Attention**: Captures multiple temporal patterns simultaneously
- **Positional Encoding**: Sinusoidal encoding for temporal positioning
- **Layer Normalization**: Stable training with residual connections
- **Causal Masking**: Prevents decoder from accessing future information
- **Mixed Precision**: Optional AMP training for efficiency

## Directory Structure

```
transformer/v1/
├── data-preparation/          # Firebase data fetching and processing
│   ├── fetch_firebase_data.py
│   └── create_training_data.py
├── training/                  # Model training and conversion
│   ├── model.py              # Transformer architecture
│   ├── dataset.py            # PyTorch data loading
│   ├── train_transformer.py  # Main training script
│   └── convert_to_onnx.py     # ONNX export for deployment
├── testing/                   # Validation and testing interface
│   ├── index.html            # Web testing interface
│   ├── server.py             # HTTP server for testing
│   ├── test_model.py         # Command-line testing
│   ├── firebase_fetch.py     # Firebase data utilities
│   └── start-server.bat      # Windows server launcher
├── inference/                 # Firebase Functions deployment
│   ├── index.js              # Cloud Function implementation
│   └── package.json          # Node.js dependencies
└── batch files and utilities
```

## Quick Start

### 1. Environment Setup
```bash
# Install PyTorch (choose CPU or GPU version)
install-pytorch-transformer.bat

# Or install manually:
pip install -r requirements.txt
```

### 2. Complete Pipeline
```bash
# Run complete setup and training pipeline
setup-complete-transformer-v1.bat
```

### 3. Step-by-Step Process

#### Data Preparation
```bash
cd data-preparation
python fetch_firebase_data.py      # Fetch Firebase data
python create_training_data.py     # Create training sequences
```

#### Training
```bash
cd training
python train_transformer.py        # Train model (may take hours)
```

#### ONNX Conversion
```bash
python convert_to_onnx.py          # Export for deployment
```

#### Local Testing
```bash
cd testing
start-server.bat                   # Launch web interface
# Open http://localhost:8000
```

#### Firebase Deployment
```bash
deploy-transformer-v1.bat          # Deploy to Firebase Functions
```

## Training Configuration

### Default Hyperparameters
- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 8 (adjust based on GPU memory)
- **Optimizer**: AdamW with weight decay 1e-5
- **Scheduler**: Cosine annealing with warm restarts
- **Dropout**: 0.1 for regularization
- **Gradient Clipping**: 1.0 for stability

### Hardware Requirements
- **Minimum**: 8GB RAM, modern CPU (training will be slow)
- **Recommended**: 16GB RAM + NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM + NVIDIA GPU with 12GB+ VRAM

### Training Time Estimates
- **CPU Only**: 12-24 hours (depending on data size)
- **Mid-range GPU**: 2-6 hours (GTX 1660, RTX 3060)
- **High-end GPU**: 1-3 hours (RTX 3080, RTX 4080, A100)

## Model Performance

### Advantages over LSTM v1
- **Training Speed**: 5-10x faster than iterative LSTM approach
- **Inference Speed**: Single forward pass vs 1440 iterative steps
- **Temporal Modeling**: Better capture of long-range dependencies
- **Parallelization**: Full sequence processed simultaneously

### Expected Metrics
- **Training Loss**: <0.1 (normalized MSE)
- **Validation Loss**: <0.15 (normalized MSE)
- **Inference Time**: 50-200ms per 24-hour forecast
- **Memory Usage**: ~2GB during training, ~500MB during inference

## Firebase Integration

### Cloud Functions
- **`runTransformerv1Prediction`**: Automatic prediction every 6 hours
- **`testTransformerv1Prediction`**: Manual testing endpoint

### Deployment Schedule
- Generates forecasts at 00:00, 06:00, 12:00, 18:00 UTC
- Stores results in `/tidal-analysis/transformer-v1-forecasts/`
- Maintains latest forecast at `/tidal-analysis/latest-transformer-v1-forecast`

### Data Flow
1. **Trigger**: New readings in Firebase `/readings`
2. **Input**: Last 4320 readings (72 hours)
3. **Processing**: Seq2seq transformer inference
4. **Output**: 1440 predictions (24 hours)
5. **Storage**: Timestamped forecast in Firebase

## Testing and Validation

### Web Interface Features
- **Real Firebase Data**: Direct connection to production database
- **Sample Data Generation**: Realistic tidal patterns for testing
- **Interactive Visualization**: Chart.js with attention-based plotting
- **Performance Metrics**: Inference time and prediction statistics
- **Model Information**: Architecture details and training metrics

### Command Line Testing
```bash
cd testing
python test_model.py --real        # Test with Firebase data
python test_model.py --sample      # Test with sample data
python test_model.py --input file.json  # Test with custom data
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training config
   - Use gradient checkpointing
   - Clear cache: `torch.cuda.empty_cache()`

2. **Training Loss Not Decreasing**
   - Check data normalization
   - Reduce learning rate
   - Verify input sequence alignment

3. **ONNX Export Errors**
   - Ensure model is in eval mode
   - Check input tensor dimensions
   - Use consistent PyTorch/ONNX versions

4. **Firebase Deployment Issues**
   - Verify ONNX model files in inference/
   - Check Node.js dependencies
   - Monitor function logs

### Performance Optimization

1. **Training Speed**
   - Use mixed precision training (AMP)
   - Increase batch size with adequate GPU memory
   - Use multiple GPUs with DataParallel

2. **Inference Speed**
   - Use ONNX Runtime with GPU support
   - Optimize batch processing
   - Cache model loading in Firebase Functions

## Technical Details

### Input Processing
- **Normalization**: Z-score using training statistics
- **Sequence Length**: Fixed 4320-point input required
- **Padding Strategy**: Mean-value padding for insufficient data
- **Data Validation**: Range checking and outlier filtering

### Output Processing
- **Direct Prediction**: Full 1440-point sequence in single pass
- **Denormalization**: Convert back to original water level scale
- **Timestamping**: Minute-by-minute predictions with ISO timestamps
- **Quality Metrics**: Statistical validation of forecast range

### Attention Patterns
- **Temporal Focus**: Multi-head attention captures various tidal cycles
- **Long-Range Dependencies**: 72-hour context for 24-hour predictions
- **Pattern Recognition**: Automatic discovery of tidal harmonics
- **Weather Integration**: Potential for multi-variate input enhancement

## Future Enhancements

### Potential Improvements
1. **Multi-variate Input**: Include wind speed, pressure, temperature
2. **Ensemble Methods**: Combine multiple model predictions
3. **Uncertainty Quantification**: Confidence intervals for predictions
4. **Online Learning**: Continuous model adaptation
5. **Hierarchical Attention**: Multi-scale temporal modeling

### Research Directions
- **Graph Neural Networks**: Spatial-temporal modeling
- **Diffusion Models**: Probabilistic forecasting
- **Foundation Models**: Pre-trained oceanic forecasting
- **Physics-Informed**: Incorporate tidal harmonic constraints

## License and Attribution

MIT License - Tide Monitor Transformer v1

For questions or contributions, refer to the main project repository.
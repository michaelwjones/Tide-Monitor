# Transformer v1 - Sequence-to-Sequence Tidal Prediction

Advanced sequence-to-sequence transformer model for 24-hour tidal prediction using multi-head attention mechanisms.

## Overview

This implementation uses a modern transformer architecture to predict tidal patterns, offering significant advantages over traditional approaches:

- **Direct Prediction**: Single forward pass generates full 24-hour forecast (144 predictions)
- **Attention Mechanisms**: Multi-head attention captures complex temporal dependencies
- **Parallel Processing**: Efficient computation compared to iterative methods
- **Long-Range Dependencies**: Superior handling of extended temporal patterns

## Architecture Details

### Model Specifications
- **Architecture**: Sequence-to-sequence transformer
- **Encoder**: 6 layers, 8 attention heads, 256 hidden dimensions
- **Decoder**: 3 layers, 8 attention heads, 256 hidden dimensions
- **Input Length**: 433 time steps (72 hours at 10-minute intervals)
- **Output Length**: 144 time steps (24 hours at 10-minute intervals)
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
├── training/                  # Model training and local server
│   ├── dataset.py            # PyTorch data loading
│   ├── train_transformer.py  # Main training script
│   ├── model_server.py       # PyTorch web server for local testing
│   └── checkpoints/          # Trained model storage
├── testing/                   # Validation and testing interface
│   ├── index.html            # Web testing interface
│   ├── server.py             # HTTP server for testing
│   ├── test_model.py         # Command-line testing
│   ├── firebase_fetch.py     # Firebase data utilities
│   └── start-server.bat      # Windows server launcher
├── inference/                 # Firebase Functions deployment (Python runtime)
│   ├── main.py               # PyTorch-based Firebase Functions
│   ├── model.py              # Transformer architecture (single source)
│   ├── best.pth              # Trained model checkpoint
│   ├── requirements.txt      # Python dependencies
│   └── firebase.json         # Configuration (512MB memory)
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

#### Local Testing
```bash
# Start PyTorch web server
cd training
python model_server.py             # Server at http://localhost:8000

# Or run test script
python test_server.py              # Comprehensive API testing
```

#### Firebase Deployment
```bash
deploy-transformer-v1.bat          # Deploy to Firebase Functions (Python runtime)
                                   # Automatically copies latest checkpoint from training/checkpoints/
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
- **`run_transformer_v1_analysis`**: Scheduled prediction every 5 minutes (Python runtime)

### Deployment Configuration
- **Runtime**: Python 3.11 with 1024MB memory allocation
- **Automatic**: Runs every 5 minutes via Cloud Scheduler
- **Storage**: Results in `/tidal-analysis/transformer-v1-forecast/`
- **Error Storage**: Error information in `/tidal-analysis/transformer-v1-error/`

### Data Flow
1. **Scheduled Trigger**: Cloud Scheduler runs every 5 minutes
2. **Data Fetching**: Last 4320 readings from Firebase (3 days, 1-minute data)
3. **Downsampling**: Convert to 433 readings (10-minute intervals)
4. **Processing**: Seq2seq transformer inference with native PyTorch
5. **Validation**: Convert NaN/infinite to -999 (error values)
6. **Storage**: Timestamped forecast with 144 predictions (10-minute intervals)

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

3. **Model Loading Errors**
   - Ensure model checkpoint exists
   - Check PyTorch version compatibility
   - Verify model architecture matches training

4. **Firebase Deployment Issues**
   - Verify trained model checkpoint exists
   - Check Python dependencies in Firebase Functions
   - Monitor function logs

5. **Data Type Errors (NumPy casting)**
   - Error: `ufunc 'isnan' not supported for the input types`
   - Cause: Firebase data contains mixed types (strings, null values)
   - Solution: Code includes robust type conversion and error handling
   - Check logs for "Insufficient data points" if filtering is too aggressive

### Performance Optimization

1. **Training Speed**
   - Use mixed precision training (AMP)
   - Increase batch size with adequate GPU memory
   - Use multiple GPUs with DataParallel

2. **Inference Speed**
   - Use PyTorch with optimized CPU inference
   - Optimize batch processing
   - Cache model loading in Firebase Functions

## Technical Details

### Architecture Management
- **Single Source Model**: Model architecture (`model.py`) is centralized in inference directory
- **Training References**: Training scripts automatically import from inference/model.py
- **Deployment**: Model definition and checkpoint are co-located for Firebase deployment
- **Consistency**: Ensures training and inference use identical model architecture

### Input Processing
- **Data Source**: 4320 1-minute readings from Firebase downsampled to 433 10-minute intervals
- **Type Safety**: Robust conversion of Firebase data (strings, numbers, null) to float with error handling
- **Data Filtering**: Filters out NaN, infinite, and non-convertible values automatically
- **Missing Values**: Invalid/NaN inputs converted to -1 (model's missing value format)
- **Normalization**: Z-score using training statistics
- **Sequence Length**: Fixed 433-point input required (72 hours @ 10-minute intervals)
- **Padding Strategy**: Mean-value padding for insufficient data

### Output Processing
- **Direct Prediction**: Full 144-point sequence in single pass (24 hours @ 10-minute intervals)
- **Error Handling**: NaN/infinite predictions converted to -999 (debug page error format)
- **Denormalization**: Convert back to original water level scale (mm)
- **Timestamping**: 10-minute interval predictions with ISO timestamps
- **Quality Metrics**: Count of error predictions and valid range validation

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
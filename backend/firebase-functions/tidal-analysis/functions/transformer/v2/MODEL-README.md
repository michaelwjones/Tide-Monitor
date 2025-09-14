# Transformer v2 Model Architecture

## Overview

Transformer v2 is an encoder-only transformer designed for tidal water level prediction. It uses a single-pass architecture to predict 24-hour future water levels from 72-hour historical input data, optimized for the periodic nature of tidal patterns.

## Architecture Design

### Model Type
- **Architecture**: Encoder-only Transformer
- **Prediction Method**: Direct single-pass (no autoregressive generation)
- **Input → Output**: 432 historical points → 144 future predictions

### Input/Output Specifications
- **Input Length**: 72 hours (432 time steps at 10-minute intervals)
- **Output Length**: 24 hours (144 time steps at 10-minute intervals)
- **Input Features**: Single feature (water level in mm)
- **Output Features**: Single feature (predicted water level in mm)

### Temporal Resolution
- **Sampling Interval**: 10 minutes
- **Downsampling Ratio**: 10:1 from original 1-minute sensor data
- **Rationale**: 
  - Captures 14.4x the highest tidal frequency (well above Nyquist theorem)
  - Preserves all relevant tidal harmonic components
  - Reduces computational complexity while maintaining prediction quality

### Input Window Design
- **Duration**: 72 hours
- **Tidal Coverage**: 
  - 6 complete semi-diurnal cycles (12.42h period)
  - 3 complete diurnal cycles (24.84h period)
  - Sufficient context for spring/neap tidal variations
- **Justification**: Provides complete harmonic context for accurate 24-hour prediction

## Transformer Configuration

### Core Architecture
- **Encoder Layers**: 8 layers
- **Attention Heads**: 16 heads
- **Model Dimension**: 512
- **Feed-forward Dimension**: 2048
- **Dropout**: 0.1

### Positional Encoding
- **Type**: Sinusoidal positional encoding
- **Purpose**: Enables model to understand temporal relationships
- **Implementation**: Standard transformer positional encoding scaled to input dimension

### Attention Mechanism
- **Type**: Multi-head self-attention
- **Benefit**: Allows model to focus on relevant historical patterns
- **Pattern Recognition**: Identifies periodic tidal components automatically

## Training Strategy

### Data Preparation
- **Input Sequences**: 72-hour sliding windows from historical data
- **Output Targets**: Corresponding 24-hour future water levels
- **Quality Filtering**: Removes physically impossible readings (< -200mm)
- **Temporal Split**: Train/validation split with temporal gap to prevent data leakage

### Loss Function
- **Primary**: Mean Squared Error (MSE)
- **Target**: Direct minimization of prediction error in mm
- **Metric**: Root Mean Squared Error (RMSE) for interpretability

### Optimization
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 32 sequences
- **Training Epochs**: Determined by early stopping on validation loss

## Key Advantages

### Single-Pass Architecture
- **Training Consistency**: Identical behavior during training and inference
- **No Accumulation Error**: Avoids error propagation from autoregressive generation
- **Computational Efficiency**: Direct 432→144 prediction in single forward pass
- **Natural Outputs**: Preserves sinusoidal tidal patterns without degradation

### Tidal-Optimized Design
- **Harmonic Awareness**: Input window covers complete tidal cycles
- **Frequency Preservation**: 10-minute sampling maintains all tidal components
- **Long-Range Dependencies**: Transformer attention captures tidal phase relationships
- **Periodic Pattern Learning**: Architecture naturally learns tidal harmonics

## Implementation Notes

### Model Input Format
```
Input shape: [batch_size, 432, 1]
Output shape: [batch_size, 144, 1]
```

### Preprocessing
- **Normalization**: Z-score normalization on training data statistics
- **Missing Data Handling**: Use -999 sentinel values for data gaps
- **Temporal Alignment**: Ensure proper chronological ordering (oldest to newest)

### Inference Pipeline
1. Fetch last 72 hours of water level data
2. Apply 10-minute downsampling with temporal alignment
3. Normalize input using training statistics
4. Single forward pass through transformer
5. Denormalize predictions to mm scale
6. Output 144 predictions at 10-minute intervals

## Quality Assurance

### Validation Metrics
- **Primary**: RMSE on validation set
- **Secondary**: Mean Absolute Error (MAE)
- **Tidal Specific**: Phase accuracy for semi-diurnal/diurnal components

### Expected Performance
- **Target RMSE**: < 50mm on validation data
- **Tidal Preservation**: Maintain natural sinusoidal patterns
- **Robustness**: Handle missing data and sensor noise gracefully

## Development & Testing

### Model Training Pipeline
1. **Data Preparation**: Run `data-preparation/run-data-preparation.ps1`
   - Fetches raw Firebase data and applies quality filtering
   - Generates 432→144 training sequences with timestamp-based matching
   - Creates normalized training/validation splits with temporal gap

2. **Modal Cloud Training**: 
   - **Setup**: `training/setup.ps1` - Installs dependencies and deploys Modal app
   - **Training**: `training/train.ps1` - Executes training on H100 GPU
   - **Model Storage**: Trained model automatically downloaded to `shared/model.pth`

3. **Local Validation**: `testing/start-server.ps1`
   - **Web Interface**: `http://localhost:8000` 
   - **Data Visualization**: Interactive charts showing all training sequences
   - **Real-time Inference**: Click-to-predict functionality with visual comparison
   - **Model Performance**: Side-by-side comparison of predictions vs ground truth

### Testing Interface Features
- **Model Status**: Automatic detection of trained model availability
- **Sequence Selection**: Browse all 11,638+ training sequences via dropdown
- **Visual Comparison**: 
  - Input data: 72 hours (blue line)
  - Ground truth: 24 hours (green line)  
  - Model prediction: 24 hours (red dashed line)
- **Performance Metrics**: Real-time RMSE calculation per sequence
- **Device Detection**: Shows CPU/GPU inference status

### Model Validation Metrics
- **Training RMSE**: ~0.26 (normalized units)
- **Real-world Accuracy**: ~81mm RMSE on actual water levels
- **Parameters**: 26.6M trainable parameters
- **Training Device**: Modal H100 GPU with 32GB RAM
- **Inference Speed**: <100ms per prediction on CPU

## Deployment Architecture

### Firebase Integration
- **Trigger**: Scheduled Cloud Function (every 6 hours)
- **Input Source**: Firebase Realtime Database readings
- **Output Storage**: Firebase path `/tidal-analysis/transformer-v2-forecast`
- **Model Format**: PyTorch JIT traced model for inference efficiency

### Development Workflow
1. **Data Collection**: Automated Firebase data fetching with quality filtering
2. **Model Training**: Modal cloud training with H100 GPU acceleration
3. **Local Testing**: Interactive web interface for model validation
4. **Performance Analysis**: Visual inspection of predictions across diverse sequences
5. **Production Deployment**: Firebase Functions integration with scheduled inference
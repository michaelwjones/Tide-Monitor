# LSTM v1 Tidal Analysis - Iterative 24-Hour Forecasting

## Overview

LSTM v1 is a Long Short-Term Memory neural network system designed for **iterative 24-hour water level forecasting**. Unlike traditional prediction systems that make single-point forecasts, LSTM v1 uses an iterative approach to generate complete 24-hour forecasts by repeatedly feeding predictions back into the model.

## Core Methodology

### Iterative Forecasting Process

1. **Initialization**: Start with the last 72 hours (4,320 minutes) of real historical water level data
2. **Single-Step Prediction**: LSTM predicts the water level for the next minute
3. **Sequence Update**: Add the prediction to the input sequence, remove the oldest value
4. **Iteration**: Repeat steps 2-3 for 1,440 iterations (24 hours worth of minutes)
5. **Output**: Complete 24-hour forecast with minute-by-minute predictions

### Mathematical Approach

```
Input: Historical sequence H = [h₁, h₂, ..., h₄₃₂₀] (72 hours)

For i = 1 to 1440:
    pᵢ = LSTM(H)                    # Predict next minute
    H = [h₂, h₃, ..., h₄₃₂₀, pᵢ]    # Update sequence
    
Output: Forecast F = [p₁, p₂, ..., p₁₄₄₀] (24 hours)
```

## Architecture Components

### 1. Data Preparation (`data-preparation/`)

**`fetch_firebase_data.py`**
- Fetches all historical tide readings from Firebase Realtime Database
- Downloads data from June 30, 2025 to present (~2 months)
- Saves raw JSON data for processing

**`create_training_data.py`**
- Parses Firebase readings into time series format
- Creates 72-hour input sequences with 1-minute-ahead targets
- Applies z-score normalization for training stability
- Generates training matrices: X (input sequences), y (targets)

### 2. Training Pipeline (`training/`)

**`model.py`** - TidalLSTM Architecture
```python
TidalLSTM(
    input_size=1,      # Single water level value
    hidden_size=128,   # LSTM hidden units
    num_layers=2,      # Stacked LSTM layers
    dropout=0.2        # Regularization
)
```

**`dataset.py`** - PyTorch Dataset
- Handles variable-length sequences with padding
- Batch processing for efficient training
- Supports sequences up to 4,320 timesteps

**`train_lstm.py`** - Training Pipeline
- Single-step ahead prediction training
- Adam optimizer with learning rate scheduling
- Early stopping based on validation loss
- Gradient clipping to prevent exploding gradients

**`convert_to_onnx.py`** - Model Export
- Converts trained PyTorch model to ONNX format
- Enables deployment to Firebase Functions with Node.js
- Validates model compatibility between PyTorch and ONNX

### 3. Firebase Deployment (`inference/`)

**`index.js`** - Cloud Functions Implementation
- Runs every 6 hours to generate fresh 24-hour forecasts
- Fetches last 72 hours of real data from Firebase
- Performs 1,440-step iterative prediction using ONNX Runtime
- Stores complete forecasts in `/tidal-analysis/lstm-v1-forecasts/`

**Key Features:**
- **Memory Management**: 1GB RAM allocation for ONNX inference
- **Timeout Handling**: 9-minute timeout for long prediction sequences
- **Error Recovery**: Graceful handling of prediction failures
- **Batch Storage**: Timestamped forecast batches for easy retrieval

## Technical Specifications

### Model Configuration
- **Input Sequence Length**: 4,320 timesteps (72 hours)
- **Prediction Horizon**: 1,440 timesteps (24 hours)
- **Temporal Resolution**: 1 minute per timestep
- **Input Features**: Water level (mm)
- **Architecture**: 2-layer LSTM with 128 hidden units
- **Training Data**: ~2 months of historical readings

### Data Processing
- **Normalization**: Z-score standardization using training statistics
- **Unit Conversion**: mm (sensor) ↔ feet (display)
- **Sequence Padding**: -1 values for variable-length inputs
- **Validation**: 300-5000mm sensor range filtering

### Deployment Schedule
- **Inference Frequency**: Every 6 hours
- **Trigger Times**: 00:00, 06:00, 12:00, 18:00 UTC
- **Storage Location**: `/tidal-analysis/lstm-v1-forecasts/{timestamp}`
- **Forecast Retention**: Latest forecast automatically retrieved

## Debug Dashboard Integration

The debug dashboard (`debug/index.html`) includes LSTM v1 visualization:

- **Button**: "Show 24h LSTM Forecast" toggle
- **Display**: Dashed orange line for predictions
- **Timeline**: Extends chart 24 hours into future
- **Data Source**: Latest forecast from Firebase
- **Real-time**: Auto-refreshes every 2 minutes

### Visual Features
- **Forecast Line**: Orange dashed line (#FF6B35)
- **Chart Extension**: X-axis extended 24 hours beyond current time
- **Data Distinction**: Dashed lines clearly separate predictions from real data
- **Zoom Support**: Prediction timeline included in zoom functionality

## Setup and Deployment

### Prerequisites
- Python 3.8+ with PyTorch
- Firebase CLI with authenticated access
- Node.js 22 for Firebase Functions

### Complete Workflow

1. **Install Dependencies**
   ```bash
   setup-pytorch.bat
   ```

2. **Interactive Setup**
   ```bash
   setup-complete-lstm-v1.bat
   ```
   - Menu-driven process with individual steps
   - Options 1-6 for specific stages
   - Option A for complete automated setup

3. **Deploy to Firebase**
   ```bash
   deploy-lstm-v1.bat
   ```

### Manual Steps
```bash
# Data preparation
cd data-preparation
python fetch_firebase_data.py
python create_training_data.py

# Model training
cd training
python train_lstm.py
python convert_to_onnx.py

# Firebase deployment
cd inference
firebase deploy --only functions:runLSTMv1Prediction
```

## Performance Characteristics

### Training Performance
- **Training Time**: ~30-60 minutes (50 epochs, CPU)
- **Model Size**: ~2MB (ONNX format)
- **Memory Usage**: ~1GB during training
- **Convergence**: Typically achieves validation loss < 0.1

### Inference Performance
- **Prediction Time**: ~5-8 minutes for 1,440 forecasts
- **Memory Usage**: 1GB Firebase Functions allocation
- **Model Loading**: ~10 seconds ONNX initialization
- **Throughput**: ~4-5 predictions per second

### Accuracy Metrics
- **Training Loss**: Mean Squared Error on normalized data
- **Validation Split**: 80% train, 20% validation
- **Early Stopping**: Prevents overfitting with 10-epoch patience
- **Generalization**: Tested on holdout validation data

## Data Flow Architecture

```
Historical Readings (Firebase) → Data Preparation → Training Data
                                                          ↓
ONNX Model ← Model Training ← Training Sequences
    ↓
Firebase Functions (Every 6 hours) → Iterative Prediction → 24h Forecast
                                                                ↓
Debug Dashboard ← Firebase Storage ← Forecast Results
```

## Error Handling

### Data Quality
- **Missing Data**: Graceful handling of gaps in historical data
- **Invalid Readings**: Filters sensor values outside 300-5000mm range
- **Normalization**: Protects against extreme outliers

### Prediction Failures
- **Individual Steps**: Failed predictions logged, process continues
- **Complete Failures**: Error data stored in `/lstm-v1-errors/`
- **Recovery**: Automatic retry on next 6-hour cycle

### System Monitoring
- **Firebase Logs**: Detailed console logging for debugging
- **Performance Tracking**: Prediction timing and memory usage
- **Error Reporting**: Structured error data with stack traces

## Future Enhancements

### Model Improvements
- **Multi-feature Input**: Include wind speed, weather data
- **Ensemble Methods**: Combine multiple LSTM models
- **Attention Mechanisms**: Focus on relevant historical periods
- **Transfer Learning**: Adapt to seasonal patterns

### System Enhancements
- **Real-time Inference**: Trigger on new data arrival
- **Confidence Intervals**: Uncertainty quantification for predictions
- **Model Versioning**: A/B testing of different architectures
- **Adaptive Scheduling**: Dynamic inference frequency based on conditions

---

*LSTM v1 represents a complete end-to-end tidal forecasting system, from data preparation through deployment, designed specifically for the Tide Monitor project's real-time water level prediction needs.*
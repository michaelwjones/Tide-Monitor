# Transformer v1 Testing & Validation

Comprehensive testing interface for validating the trained seq2seq transformer model with real Firebase data and 24-hour forecasting.

## Files

- **`index.html`** - Web-based testing interface with visualization
- **`server.py`** - HTTP server that loads the trained transformer model
- **`test_model.py`** - Command-line testing script  
- **`start-server.bat`** - Windows batch file to launch server automatically
- **`firebase_fetch.py`** - Firebase data fetching utilities
- **`README.md`** - This documentation

## Quick Start

**Windows Users:**
```bash
# Double-click to start server and open browser automatically
start-server.bat
```

**Manual Start:**
```bash
python server.py
# Open browser to: http://localhost:8000
```

## Web Interface Features

### Real Data Integration
- **Firebase Data Fetching**: Directly fetches 72 hours of real water level data (4320 readings)
- **Sequence-to-Sequence**: Uses complete 72-hour input for direct 24-hour prediction
- **No Iterative Processing**: Single forward pass generates all 1440 predictions

### Testing Modes
1. **Fetch Real Data**: Gets actual readings from Firebase database
2. **Generate Sample Data**: Creates realistic tidal patterns for testing
3. **Direct Prediction**: Single seq2seq forward pass for 24-hour forecast
4. **Validation Comparison**: Compare predictions against actual data

### Visualization
- **Chart.js Integration**: Interactive time-series plotting
- **Historical Context**: Shows full 72 hours of input data (blue line)
- **Forecast Display**: 24-hour predictions as dashed orange line
- **Attention Visualization**: Optional attention weights display
- **Model Architecture**: Shows encoder/decoder layers and parameters

### Model Information
- **Transformer Architecture**: 6 encoder + 3 decoder layers, 8 attention heads
- **Seq2Seq Processing**: Direct 4320→1440 sequence transformation
- **Training Metrics**: Displays validation loss and model parameters
- **Error Analysis**: Real-time prediction accuracy metrics

## Technical Details

### Data Processing
- **Fixed Input Length**: Exactly 4320 readings (72 hours) required
- **Firebase Query**: `limitToLast=4320` for consistent sequence length
- **Normalization**: Uses training normalization parameters for consistency
- **Sequence Padding**: Handles datasets with fewer than 4320 readings

### Model Architecture
- **Input**: 4320-step sequences (72 hours of minute-by-minute data)
- **Output**: 1440-step predictions (24 hours of forecasts)
- **Direct Prediction**: Non-iterative seq2seq approach
- **Attention Mechanism**: Multi-head attention for temporal dependencies

## Command Line Testing (`test_model.py`)

Python script for command-line testing:

```bash
# Interactive mode
python test_model.py

# Quick test with sample data
python test_model.py --sample

# Load specific model checkpoint
python test_model.py --model ../training/checkpoints/best.pth
```

**Features:**
- Loads trained transformer from `../training/checkpoints/best.pth`
- Uses real normalization parameters from training
- PyTorch or ONNX inference options
- Performance benchmarking and accuracy metrics

## Prerequisites

- **Trained model**: `train_transformer.py` must have been run successfully
- **PyTorch**: GPU-enabled installation recommended for large model inference
- **ONNX Runtime**: Optional for deployment testing
- **Dependencies**: All training modules must be accessible

## Usage Workflow

### Complete Testing Process
1. **Train Model**: Run `train_transformer.py` in the training folder
2. **Start Server**: Double-click `start-server.bat` or run `python server.py`
3. **Fetch Real Data**: Click "Fetch Last 72 Hours from Firebase"
4. **Generate Forecast**: Click "Predict Next 24 Hours" (single forward pass)
5. **Analyze Results**: Review chart showing historical data vs predictions
6. **Export Model**: Use `convert_to_onnx.py` for deployment preparation

### Data Sources
- **Real Firebase Data**: Direct connection to production database (4320 readings)
- **Sample Data**: Generated realistic tidal patterns with proper sequence length
- **Validation Data**: Historical sequences for accuracy testing

## Integration Notes

This testing system validates:
- **Seq2Seq Performance**: Direct 72→24 hour prediction accuracy
- **Attention Mechanisms**: Transformer's ability to capture temporal patterns
- **Data Pipeline**: Firebase → Processing → Transformer → Visualization
- **ONNX Compatibility**: Deployment-ready model validation
- **Computational Efficiency**: Single-pass prediction vs iterative methods

## Key Differences from LSTM v1

- **Direct Prediction**: No iterative forecasting required
- **Fixed Input Length**: Exactly 4320 readings needed (vs variable length)
- **Parallel Processing**: Attention allows parallel computation
- **Better Long-Range Dependencies**: Transformer architecture advantages
- **Faster Inference**: Single forward pass for full 24-hour forecast

Use this testing interface to validate transformer performance before deploying to Firebase Functions for production use.
# Transformer v1 Testing & Validation

Comprehensive testing interface for validating the trained single-pass encoder transformer model with real Firebase data and 24-hour forecasting.

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
- **Firebase Data Fetching**: Directly fetches 72 hours of real water level data (433 readings at 10-minute intervals)
- **Raw Dataset Access**: Complete Firebase dataset visualization for exploration
- **Sequence-to-Sequence**: Uses complete 72-hour input for direct 24-hour prediction
- **No Iterative Processing**: Single forward pass generates all 144 predictions

### Raw Data Visualization
- **Complete Dataset**: Load and visualize entire Firebase dataset
- **Interactive Exploration**: Mouse wheel zoom, click-and-drag pan
- **Data Filtering**: Compare raw vs filtered training data
- **Quality Analysis**: Statistics and date ranges for dataset inspection

### Testing Modes
1. **Fetch Real Data**: Gets actual readings from Firebase database
2. **Generate Sample Data**: Creates realistic tidal patterns for testing
3. **Direct Prediction**: Single seq2seq forward pass for 24-hour forecast
4. **Raw Data Analysis**: Explore complete dataset with interactive zoom/pan
5. **Training Data Inspection**: View random training sequences with targets
6. **Filtered Data Comparison**: Compare raw vs filtered training datasets

### Visualization
- **Chart.js Integration**: Interactive time-series plotting
- **Historical Context**: Shows full 72 hours of input data (blue line)
- **Forecast Display**: 24-hour predictions as dashed orange line
- **Raw Data Exploration**: Complete Firebase dataset visualization with zoom/pan
- **Training Data Viewer**: Random training sequence inspection
- **Model Architecture**: Shows encoder/decoder layers and parameters

### Model Information
- **Transformer Architecture**: 8 encoder layers, 16 attention heads, single-pass processing
- **Direct Prediction**: 433→144 sequence transformation in one forward pass
- **Training Metrics**: Displays validation loss and model parameters
- **Error Analysis**: Real-time prediction accuracy metrics

## Technical Details

### Data Processing
- **Fixed Input Length**: Exactly 433 readings (72 hours at 10-minute intervals) required
- **Firebase Query**: Fetches data with 10-minute downsampling for consistent sequence length
- **Normalization**: Uses training normalization parameters for consistency
- **Sequence Padding**: Handles datasets with fewer than 433 readings

### Model Architecture
- **Input**: 433-step sequences (72 hours at 10-minute intervals)
- **Output**: 144-step predictions (24 hours at 10-minute intervals)
- **Direct Prediction**: Single-pass encoder-only architecture
- **Attention Mechanism**: 16-head attention for temporal dependencies

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
- Native PyTorch inference
- Performance benchmarking and accuracy metrics

## Prerequisites

- **Trained model**: `train_transformer.py` must have been run successfully
- **PyTorch**: GPU-enabled installation recommended for large model inference
- **
- **Dependencies**: All training modules must be accessible

## Usage Workflow

### Complete Testing Process
1. **Train Model**: Run `train_transformer.py` in the training folder
2. **Start Server**: Double-click `start-server.bat` or run `python server.py`
3. **Explore Data**: Use "Raw Data Visualization" to examine complete dataset
4. **Fetch Real Data**: Click "Fetch Last 72 Hours from Firebase" (10-minute intervals)
5. **Generate Forecast**: Click "Predict Next 24 Hours" (single forward pass)
6. **Analyze Results**: Review chart showing historical data vs predictions
7. **Training Analysis**: Explore random training sequences and model performance

### Data Sources
- **Real Firebase Data**: Direct connection to production database (433 readings at 10-minute intervals)
- **Sample Data**: Generated realistic tidal patterns with proper sequence length
- **Validation Data**: Historical sequences for accuracy testing

## Integration Notes

This testing system validates:
- **Seq2Seq Performance**: Direct 72→24 hour prediction accuracy
- **Attention Mechanisms**: Transformer's ability to capture temporal patterns
- **Data Pipeline**: Firebase → Processing → Transformer → Visualization
- **
- **Computational Efficiency**: Single-pass prediction vs iterative methods

## Key Differences from LSTM v1

- **Direct Prediction**: No iterative forecasting required
- **Fixed Input Length**: Exactly 433 readings needed (vs variable length)
- **Parallel Processing**: Attention allows parallel computation
- **Better Long-Range Dependencies**: Transformer architecture advantages
- **Faster Inference**: Single forward pass for full 24-hour forecast

Use this testing interface to validate transformer performance before deploying to Firebase Functions for production use.
# LSTM v1 Testing & Validation

Comprehensive testing interface for validating the actual trained LSTM model with real Firebase data and 24-hour forecasting.

## Files

- **`index.html`** - Web-based testing interface with visualization
- **`server.py`** - HTTP server that loads the trained model
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
- **Firebase Data Fetching**: Directly fetches up to 72 hours of real water level data
- **Robust Input Handling**: Accepts any length of data (pads missing values with -1)
- **No Validation**: Uses raw Firebase data without filtering (validation handled elsewhere)

### Testing Modes
1. **Fetch Real Data**: Gets actual readings from Firebase database
2. **Generate Sample Data**: Creates realistic tidal patterns for testing
3. **Single Prediction**: Tests one-step-ahead prediction
4. **24-Hour Forecast**: Generates full 1,440-minute iterative forecast

### Visualization
- **Chart.js Integration**: Interactive time-series plotting
- **Historical Context**: Shows full 72 hours of input data (blue line)
- **Forecast Display**: 24-hour predictions as dashed orange line
- **Fixed Sizing**: Chart maintains consistent 400px height

### Model Information
- **Architecture Display**: Shows hidden size, layers, validation loss
- **Progress Tracking**: Real-time progress for 24-hour forecasts
- **Error Handling**: Graceful fallbacks for chart and network issues

## Technical Details

### Data Processing
- **Firebase Query**: `limitToLast=4320` to get up to 72 hours of data
- **Missing Value Handling**: Pads incomplete datasets with -1 at the beginning
- **Timestamp Handling**: Creates appropriate timestamps for padded values
- **Variable Length Input**: Model trained to handle any input length

### Model Architecture
- **Input**: Variable length sequences (up to 4,320 readings for 72 hours)
- **Output**: Single water level prediction (mm)
- **Iterative Forecasting**: Uses sliding 72-hour window for 24-hour predictions
- **Missing Data**: -1 values indicate missing/unavailable readings

## Command Line Testing (`test_model.py`)

Python script for command-line testing:

```bash
# Interactive mode
python test_model.py

# Quick test with sample data
python test_model.py --sample
```

**Features:**
- Loads actual trained model from `../training/trained_models/best_model.pth`
- Uses real normalization parameters from training
- Three input modes: generated data, manual entry, file loading
- Real predictions using PyTorch model

## Prerequisites

- **Trained model**: `train_lstm.py` must have been run successfully
- **PyTorch**: CUDA-enabled installation recommended for performance
- **Dependencies**: All training modules must be accessible

## Usage Workflow

### Complete Testing Process
1. **Train Model**: Run `train_lstm.py` in the training folder
2. **Start Server**: Double-click `start-server.bat` or run `python server.py`
3. **Fetch Real Data**: Click "Fetch Last 72 Hours from Firebase"
4. **Generate Forecast**: Click "Predict Next 24 Hours"
5. **Analyze Results**: Review chart showing historical data vs predictions

### Data Sources
- **Real Firebase Data**: Direct connection to production database
- **Sample Data**: Generated realistic tidal patterns with noise
- **Manual Input**: Custom data entry via textarea

## Integration Notes

This testing system validates:
- **Model Performance**: Real predictions on actual tidal data
- **Data Pipeline**: Firebase → Processing → LSTM → Visualization
- **Iterative Forecasting**: 24-hour prediction generation
- **Error Handling**: Robust operation with missing or incomplete data

Use this testing interface to validate model performance before deploying to Firebase Functions for production use.
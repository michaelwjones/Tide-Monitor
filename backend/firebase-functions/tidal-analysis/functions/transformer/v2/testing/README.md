# Transformer v2 Testing Interface

## Overview

The testing interface provides a comprehensive web-based environment for validating the trained transformer v2 model. It combines data visualization with real-time inference capabilities to evaluate model performance across the entire training dataset.

## Quick Start

```powershell
# Start the testing server
.\start-server.ps1

# Open browser to http://localhost:8000
```

## Features

### 📊 **Data Visualization**
- **Complete Dataset**: Browse all 11,638+ training sequences
- **Interactive Charts**: Zoom, pan, and inspect individual data points  
- **Multi-dataset View**: Input (72h) and target (24h) data clearly separated
- **Sequence Statistics**: Min/max/mean values displayed for each sequence
- **Distribution Analysis**: Histogram view of data distribution patterns

### 🧠 **Model Inference**
- **Automatic Model Detection**: Interface detects trained model availability
- **Device Status**: Shows whether inference runs on CPU or GPU
- **Click-to-Predict**: One-click inference on any selected sequence
- **Visual Comparison**: Predictions overlaid on ground truth data
- **Real-time Feedback**: Loading states and error handling

### 📈 **Performance Analysis**
- **Side-by-side Comparison**: 
  - **Blue Line**: 72-hour input data (historical water levels)
  - **Green Line**: 24-hour ground truth target
  - **Red Dashed Line**: 24-hour model prediction
- **Temporal Alignment**: Predictions properly aligned with future time steps
- **Pattern Recognition**: Evaluate model's capture of tidal harmonics

## Interface Components

### Model Status Panel
- **Status Indicator**: Green (ready) / Red (unavailable)
- **Device Information**: CPU or GPU inference mode
- **Model Metrics**: Parameter count and validation performance

### Sequence Controls
- **Dropdown Selector**: Choose from thousands of training sequences
- **Random Sequence**: Jump to random sequence for diverse testing
- **Distribution View**: Toggle histogram of data values

### Inference Controls
- **Run Prediction**: Execute model inference on current sequence
- **Status Check**: Verify model availability and readiness
- **Clear Results**: Remove prediction overlay

## Technical Implementation

### Server Architecture (`server.py`)
```
HTTP Server (Port 8000)
├── Static File Serving (HTML, CSS, JS)
├── Data Endpoints (/data/*)
│   ├── X_train.npy → JSON conversion
│   ├── y_train.npy → JSON conversion
│   └── metadata.json → Statistics
└── Inference API (/inference/*)
    ├── /status → Model availability
    └── /predict_sequence → Run inference
```

### Inference Engine (`inference.py`)
- **Model Loading**: Automatic detection from `../shared/model.pth`
- **Normalization**: Applies training-time statistics
- **Prediction**: Single forward pass (no autoregressive generation)
- **Device Management**: Automatic CPU/GPU selection

### Web Interface (`index.html`)
- **Chart.js Integration**: High-performance data visualization
- **Async Loading**: Non-blocking model status checks
- **Error Handling**: Graceful degradation for missing models
- **Responsive Design**: Works on desktop and mobile

## Data Pipeline

### Training Data Flow
```
Firebase → fetch_firebase_data.py → create_training_data.py → *.npy files → Web Interface
```

### Inference Data Flow
```
Selected Sequence → Normalization → Model Inference → Denormalization → Chart Overlay
```

## Model Requirements

### Required Files
- `../shared/model.pth` - Trained transformer model (from Modal)
- `../data-preparation/data/X_train.npy` - Training input sequences
- `../data-preparation/data/y_train.npy` - Training target sequences  
- `../data-preparation/data/normalization_params.json` - Scaling parameters

### Model Format
- **Framework**: PyTorch
- **Architecture**: Transformer encoder-only
- **Input Shape**: (batch_size, 432, 1)
- **Output Shape**: (batch_size, 144)

## Usage Examples

### Validate Model Performance
1. Select a sequence from the dropdown
2. Click "Run Prediction on Current Sequence"
3. Compare red dashed line (prediction) with green line (ground truth)
4. Look for tidal pattern preservation and amplitude accuracy

### Analyze Difficult Cases
1. Use "Random Sequence" to find challenging scenarios
2. Look for sequences with unusual patterns or missing data
3. Evaluate how well the model handles edge cases

### Performance Benchmarking
1. Test inference speed across multiple sequences
2. Monitor memory usage with large sequence indices
3. Compare predictions across different tidal conditions

## Troubleshooting

### Model Not Available
- **Cause**: Missing or corrupted model file
- **Solution**: Re-run training pipeline or check `../shared/model.pth`

### Prediction Errors
- **Cause**: Sequence index out of range or normalization issues
- **Solution**: Check console for detailed error messages

### Slow Performance  
- **Cause**: Large dataset loading or inefficient inference
- **Solution**: Monitor network tab and server console output

## Development Notes

### Adding New Metrics
Extend the inference results with additional performance metrics:
```python
# In inference.py
result = {
    'prediction': prediction.tolist(),
    'target': target_seq.tolist(), 
    'rmse': calculate_rmse(prediction, target),  # Add custom metrics
    'mae': calculate_mae(prediction, target)
}
```

### Custom Visualizations
Add new chart types by extending the Chart.js configuration:
```javascript
// In index.html
const newDataset = {
    label: 'Custom Metric',
    data: customData,
    borderColor: 'rgb(75, 192, 192)'
};
sequenceChart.data.datasets.push(newDataset);
```

### Performance Optimization
For production deployment, consider:
- Caching inference results for repeated sequences
- Implementing WebSocket for real-time updates
- Adding batch inference for multiple sequences

## API Reference

### GET /inference/status
Returns model availability and configuration:
```json
{
    "status": "ready",
    "device": "cpu",
    "model_loaded": true,
    "normalization_loaded": true
}
```

### GET /inference/predict_sequence?index=N
Runs inference on training sequence N:
```json
{
    "prediction": [144 predicted values],
    "target": [144 ground truth values],
    "input": [432 input values],
    "sequence_index": N
}
```

### GET /data/X_train.npy
Returns training input data in JSON format:
```json
{
    "shape": [11638, 432],
    "data": [...],
    "stats": {
        "min": -2.5,
        "max": 2.8,
        "mean": 0.0,
        "std": 1.0
    }
}
```
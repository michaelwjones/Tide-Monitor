# Tidal Analysis Functions

Advanced signal processing and machine learning forecasting for tidal data.

## Analysis Methods

### Matrix Pencil (`functions/matrix-pencil/v1/`)
- **Purpose**: Harmonic analysis of tidal data using signal decomposition
- **Schedule**: Runs every 5 minutes via Cloud Scheduler
- **Method**: Complex exponential decomposition with SVD-based model selection
- **Output**: Stored in `/tidal-analysis/matrix-pencil-v1/` for debug dashboard

### LSTM Forecasting (`functions/lstm/v1/`)
- **Purpose**: 24-hour water level predictions using recurrent neural networks
- **Schedule**: Runs every 6 hours
- **Architecture**: 2-layer LSTM with 128 units, ONNX inference
- **Input**: Last 72 hours (4320 readings)
- **Output**: 1440 future predictions (24 hours), stored in `/tidal-analysis/lstm-v1-forecasts/`

### Transformer Forecasting (`functions/transformer/v1/`)
- **Purpose**: 24-hour water level predictions using attention mechanisms
- **Schedule**: Runs every 5 minutes
- **Architecture**: Single-pass encoder-only transformer (433→144 mapping)
- **Input**: Last 72 hours at 10-minute intervals (433 readings)
- **Output**: 144 future predictions (24 hours), stored in `/tidal-analysis/transformer-v1-forecast/`

### Transformer v2 (`functions/transformer/v2/`)
- **Purpose**: Enhanced transformer with improved training methodology
- **Architecture**: Same as v1 but with refined data preparation and validation
- **Features**: Two-dataset structure (training/validation), incremental sequence naming

### Transformer v3 (`functions/transformer/v3/`)
- **Purpose**: Latest transformer with sanity analysis capabilities
- **Architecture**: Same core transformer with enhanced data structure
- **Features**: Three-dataset structure (training/validation/sanity), timestamp-based naming
- **Key Innovation**: Dedicated sanity test dataset with 1440 input-only sequences

## Function Tracking

See `analysis-functions.csv` for complete deployment history and function management.

## Configuration

Analysis functions use `.env` files for enabling/disabling:
- Matrix Pencil: Cost-controlled via `TIDAL_ANALYSIS_ENABLED` flag
- ML models: Require trained model files for deployment

## Method-Specific Details

For implementation details, training procedures, and troubleshooting:
- `/functions/matrix-pencil/v1/README.md`
- `/functions/lstm/v1/README.md`
- `/functions/transformer/v1/README.md`
- `/functions/transformer/v2/README.md`
- `/functions/transformer/v3/README.md`
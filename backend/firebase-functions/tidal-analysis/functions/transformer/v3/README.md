# Transformer v3 - Tidal Water Level Forecasting

Transformer-based neural network for predicting water levels 24 hours into the future using 72 hours of historical data.

## Architecture Changes in v3

**Key improvements over v2:**
- **Three-dataset structure**: Training, validation, and discontinuity testing datasets
- **Timestamp-based naming**: Sequences named by timestamp (YYYYMMDD_HHMM) instead of incremental IDs
- **Discontinuity analysis**: Dedicated test dataset with 1440 input-only sequences for robust discontinuity detection
- **Improved data handling**: Uses -999 sentinel values for missing data instead of skipping sequences

## Data Preparation

**Input**: 72 hours (432 points at 10-minute intervals)
**Output**: 24 hours (144 points at 10-minute intervals)

### Dataset Structure
- **Training**: 13,388 sequences (main historical data)
- **Validation**: 3,346 sequences (recent data for validation)
- **Discontinuity**: 1,440 sequences (last 4 days, input-only for testing)

### Key Features
- Timestamp-based sequence naming (YYYYMMDD_HHMM format)
- Separation date: 2025-09-18T05:07:09+00:00
- Missing data handling with -999 values
- Binary search with 5-minute tolerance for timestamp matching

## Usage

### Data Preparation
```bash
cd data-preparation
python create_training_data.py
```

Generates:
- `X_train.npy`, `y_train.npy` - Training data
- `X_val.npy`, `y_val.npy` - Validation data  
- `X_discontinuity.npy` - Discontinuity test sequences (input-only)
- `sequence_names_*.json` - Timestamp-based naming files
- `normalization_params.json` - Normalization parameters
- `metadata.json` - Dataset statistics

### Training
```bash
cd training
python train_transformer.py
```

### Testing Interface
```bash
cd tester
python server.py
```
Access at http://localhost:8080

### Discontinuity Analysis
```bash
cd discontinuity-analysis
python analyze.py --data inference --num-tests 50
```

## Model Details

- **Architecture**: Single-pass encoder-only transformer (433→144 mapping)
- **Input size**: 432 time steps + 1 positional encoding = 433 features
- **Output size**: 144 time steps (24 hours of predictions)
- **No autoregressive generation** - direct sequence-to-sequence mapping

## Files Structure

```
v3/
├── data-preparation/          # Training data generation
│   ├── create_training_data.py
│   └── data/                  # Generated datasets
├── training/                  # Model training
│   └── train_transformer.py
├── tester/                    # Web testing interface
│   ├── server.py
│   └── index.html
├── discontinuity-analysis/    # Discontinuity detection
│   └── analyze.py
└── shared/                    # Shared model and utilities
    ├── model.pth
    └── inference.py
```

## Key Improvements

1. **Robust discontinuity testing**: Pre-generated dataset ensures consistent testing conditions
2. **Better sequence tracking**: Timestamp-based naming enables precise sequence identification  
3. **Improved data integrity**: -999 handling prevents sequence gaps while maintaining data quality
4. **Enhanced analysis**: Dedicated discontinuity dataset for comprehensive model evaluation
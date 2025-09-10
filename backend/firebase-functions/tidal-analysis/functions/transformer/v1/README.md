# Transformer v1 - Sequence-to-Sequence Tidal Prediction

Advanced sequence-to-sequence transformer model for 24-hour tidal prediction using multi-head attention mechanisms.

## Overview

This implementation uses a modern transformer architecture to predict tidal patterns, offering significant advantages over traditional approaches:

- **Direct Prediction**: Single forward pass generates full 24-hour forecast (144 predictions)
- **Attention Mechanisms**: Multi-head attention captures complex temporal dependencies
- **Parallel Processing**: Efficient computation compared to iterative methods
- **Long-Range Dependencies**: Superior handling of extended temporal patterns

## Folder Structure

The Transformer v1 system is organized into clean local and cloud components:

```
├── local/                    # Local development and training
│   ├── data-preparation/     # Data processing and preparation
│   ├── training/             # Local model training (optimized hyperparameters)
│   └── testing/              # Model validation and testing
├── cloud/                    # Cloud deployment components
│   ├── inference/            # Firebase Functions deployment
│   └── training/sweeps/      # Modal Labs hyperparameter optimization
└── STRUCTURE.md              # Detailed folder documentation
```

**See [STRUCTURE.md](STRUCTURE.md) for complete organization details and usage guide.**

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
├── data-preparation/          # Optimized Firebase data pipeline
│   ├── fetch_firebase_data.py    # Raw data fetching with quality filtering
│   └── create_training_data.py   # Timestamp-based sequence generation
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
├── cloud/inference/           # Legacy deployment files (moved to tidal-analysis root)
│   └── [deployment files moved to ../../../../../ (tidal-analysis root)]
├── cloud/                     # Cloud training on Modal (serverless GPUs)
│   └── training/
│       ├── sweeps/            # Hyperparameter optimization
│       │   ├── login.ps1      # Modal authentication (one-time)
│       │   ├── setup.ps1      # Dependencies + data upload
│       │   ├── run.ps1        # Start hyperparameter sweep
│       │   └── modal_hp_sweep.py    # Ray Tune optimization
│       └── single-runs/       # Single training runs
│           ├── login.ps1      # Modal authentication (one-time)
│           ├── setup.ps1      # Dependencies + data upload
│           ├── run.ps1        # Start single training
│           └── modal_single_run_seq2seq.py
└── batch files and utilities
```

## Quick Start

Choose between local or cloud training:

### Option A: Local Training

#### 1. Environment Setup
```bash
# Install PyTorch (choose CPU or GPU version)
install-pytorch-transformer.bat

# Or install manually:
pip install -r requirements.txt
```

#### 2. Complete Pipeline
```bash
# Run complete setup and training pipeline
setup-complete-transformer-v1.bat
```

### Option B: Cloud Training (Recommended)

#### 1. Quick Inference Model (Start Here)
```bash
cd cloud/training/single-runs
.\login.ps1      # One-time Modal authentication
.\setup.ps1      # Install dependencies + upload data
.\run.ps1        # Train modest model for immediate inference (~1 hour, $6-8)
```

#### 2. Comprehensive Hyperparameter Optimization
```bash
cd cloud/training/sweeps
.\login.ps1      # One-time Modal authentication (if not done)
.\setup.ps1      # Install dependencies + upload data  
.\run.ps1        # Optimize attention heads & parameters (~3-4 days, $300-600)
```

### 3. Step-by-Step Process

#### Data Preparation
```bash
cd data-preparation
python fetch_firebase_data.py      # Fetch and filter Firebase data
python create_training_data.py     # Create training sequences with timestamp matching
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
- **Batch Size**: 4 (reduced for GPU memory constraints)
- **Optimizer**: AdamW with weight decay 1e-5
- **Scheduler**: Cosine annealing with warm restarts
- **Dropout**: 0.1 for regularization
- **Gradient Clipping**: 1.0 for stability

### Data Augmentation (Training Only)
- **Missing Value Simulation**: 2% probability of individual -999 values
- **Gap Simulation**: 5% probability of creating 1-10 timestep gaps per sequence  
- **Value Perturbation**: Gaussian noise (σ=0.1) added to normalized values
- **Sequence Shuffling**: Randomized batch order each epoch prevents overfitting
- **Robustness Training**: Models learn to handle real-world sensor gaps and noise

### Hardware Requirements
- **Minimum**: 8GB RAM, modern CPU (training will be slow)
- **Recommended**: 16GB RAM + NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM + NVIDIA GPU with 12GB+ VRAM

### Training Time Estimates
- **CPU Only**: 12-24 hours (depending on data size)
- **Mid-range GPU**: 2-6 hours (GTX 1660, RTX 3060)
- **High-end GPU**: 1-3 hours (RTX 3080, RTX 4080, A100)

## Cloud Training Options

### Single Run (Inference Model)
- **Purpose**: Get a working model quickly for immediate deployment
- **Configuration**: d_model=384, 6 layers (4 encoder + 2 decoder), 12 attention heads
- **Time**: ~1 hour on H100 GPU
- **Cost**: ~$6-8
- **Use case**: Production inference while optimizing hyperparameters

### Hyperparameter Sweep (Optimization)
- **Purpose**: Find optimal attention head configuration and training parameters
- **Focus**: Attention heads [8,16,32] with fixed architecture (d_model=512, 8 layers)
- **Time**: ~3-4 days on H100 GPU (30 trials)
- **Cost**: ~$300-600
- **Use case**: Research optimal configuration for maximum performance

## Model Performance

### Advantages over LSTM v1
- **Training Speed**: 5-10x faster than iterative LSTM approach
- **Inference Speed**: Single forward pass vs 1440 iterative steps
- **Temporal Modeling**: Better capture of long-range dependencies
- **Parallelization**: Full sequence processed simultaneously
- **Robustness Training**: Built-in missing value and noise simulation during training

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
2. **Data Fetching**: Last 4320 readings from Firebase (1-minute resolution data)
3. **Real-Time Downsampling**: Creates 433-point sequence using current time as reference
   - Finds closest reading to current time as starting point
   - Works backwards in 10-minute intervals for 72 hours (433 points)
   - Maps each target time to closest reading within ±5 minutes
   - Uses -999 for missing time slots (model trained to handle these)
   - Preserves chronological order (oldest to newest for model input)
4. **Direct Model Input**: Bypasses input preparation to avoid data corruption
   - Pre-validated 433-point sequence fed directly to model
   - Preserves -999 synthetic values exactly as model expects
   - No additional padding or mean calculations that could distort data
5. **Inference**: Native PyTorch transformer generates 144 predictions in single pass
6. **Timestamp Alignment**: Uses last real data timestamp (not synthetic) as prediction base
7. **Quality Assurance**: Converts NaN/infinite predictions to -999 (error values)
8. **Storage**: Complete forecast with metadata and error tracking

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
   - Consider reducing data augmentation (set augment=False for initial tests)

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
   - Solution: Wrap values in `np.array()` before using numpy functions
     - Proper type casting in main.py lines 128, 166, 171
     - Comprehensive error handling with filename/line numbers
     - Uses -999 for missing values (consistent with data pipeline)
   - Check logs for "Insufficient data points" if filtering is too aggressive

6. **Firmware Timestamp Issues (Bypassed)**
   - Issue: Device timer drift caused inconsistent timing between measurements and timestamps
   - Root Cause: Firmware uses `millis()` for scheduling but `Time.format()` for timestamps
   - Solution: Data preparation now ignores all timestamp validation and assumes chronological order
   - Result: 5x more training data available (no sequences filtered out for timing issues)

7. **Forecast Timestamp Issues**
   - Issue: Forecast timestamps must align with sensor data timeline
   - Cause: Using current processing time creates timing offset
   - Solution: Pass `last_data_timestamp` and start predictions from last sensor reading
   - Implementation in main.py with proper timestamp parsing and error handling

8. **Uniform/Flat Predictions Issue** ✅ **FIXED**
   - Issue: Model outputting same value for all predictions (no variation)
   - Root Causes:
     - **Contaminated normalization**: Mean/std calculation included -999 values
     - Double processing in `prepare_input_sequence` corrupted carefully crafted data
     - Using synthetic timestamp as base for predictions
   - Solution: Multiple critical fixes applied
     - **Fixed normalization**: Exclude -999 synthetic values from mean/std calculation
     - **Temporal gap enforcement**: Prevent data leakage between train/validation sets
     - New `predict_24_hours_direct()` method bypasses redundant processing
     - Uses last real data timestamp for prediction base
   - Result: Natural tidal variations restored in predictions

9. **Data Leakage in Train/Validation Split** ✅ **FIXED**
   - Issue: Training and validation sequences had massive temporal overlap
   - Root Cause: Random 1-9 minute offsets + 96-hour sequences = 95+ hours overlap
   - Impact: Artificially inflated validation performance, unrealistic loss metrics
   - Solution: Timestamp-based temporal gap enforcement
     - Track actual start/end timestamps for each sequence
     - Find last training sequence end time
     - Start validation sequences after training data ends
     - Calculate real temporal gap based on actual timestamps
   - Result: True validation performance on genuinely unseen future data

10. **Deployment Orphaned Functions Prompt**
   - Issue: Firebase asks about deployed functions not in local source
   - Cause: Other analysis functions exist in Firebase but not current directory
   - Solution: Updated deployment script uses `--only functions:run_transformer_v1_analysis`
   - Result: Clean deployments without prompts about unrelated functions

11. **Invalid Water Level Data Contamination** ✅ **FIXED**
   - Issue: Negative water levels (e.g., -2582mm) corrupting training data
   - Root Cause: No physical validity checking in data pipeline
   - Impact: 4,107 invalid readings (4.1% of data) treated as valid training data
   - Solution: Added water level filtering in `process_raw_data.py`
     - Filters out readings < -200mm (physically impossible)
     - Preserves minor sensor calibration variations (-200mm to 0mm)
     - Reports filtered readings for monitoring
   - Result: Clean training data with only physically possible water levels

12. **Misleading Temporal Gap Logging** ✅ **FIXED**
   - Issue: Train/validation gap reported as 0.1 hours despite discarding 1159 sequences
   - Root Cause: Logging calculated boundary gap, not total discarded timespan
   - Impact: Misleading metrics about actual temporal separation
   - Solution: Enhanced logging in `create_training_data.py`
     - Shows actual timespan of discarded sequences (e.g., 96 hours)
     - Displays boundary gap between last training and first validation
     - Clear distinction between discarded data span and sequence boundary gap
   - Result: Accurate reporting of temporal separation for data leakage prevention

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
- **Single Source Model**: Model architecture (`model.py`) is now in tidal-analysis root directory
- **Training References**: Training scripts import from ../../../../../model.py (tidal-analysis root)
- **Deployment**: Model definition and checkpoint are in tidal-analysis root for Firebase deployment
- **Consistency**: Ensures training and inference use identical model architecture
- **Python 3.13 Runtime**: Uses latest Python runtime with firebase.json configuration

### Optimized Data Preparation Pipeline

#### Data Fetching and Filtering (`fetch_firebase_data.py`)
- **Firebase Integration**: Downloads complete dataset from Firebase Realtime Database
- **Quality Filtering**: Removes physically impossible water levels (< -200mm) from source data
- **Dual File Output**: Preserves both raw data and filtered data for analysis
- **Data Validation**: Filters out 3.9% of readings while preserving 96.1% of quality sensor data
- **Performance**: Efficiently processes 100K+ readings with immediate filtering

#### Timestamp-Based Sequence Generation (`create_training_data.py`)
- **Binary Search Optimization**: Uses O(log n) binary search for 100x speed improvement over linear search
- **Timestamp Matching**: Finds closest readings within ±5 minutes of target 10-minute intervals
- **No Synthetic Data**: Works with real sensor data only, no gap filling or synthetic values
- **Quality Control**: Filters sequences with insufficient data coverage automatically
- **Temporal Split**: Creates proper train/validation split with temporal gap to prevent data leakage

#### Current Training Data Statistics (Latest Run)
- **Source Data**: 97,525 filtered readings from Firebase
- **Generated Sequences**: 18,346 potential 96-hour sequences processed
- **Valid Sequences**: 14,547 sequences (79.3% retention rate)
- **Training Split**: 11,638 training / 1,743 validation sequences
- **Temporal Gap**: 192.2 hours discarded to ensure no data leakage
- **File Size**: 58.9 MB of high-quality training data
- **Processing Time**: Under 1 minute with binary search optimization

#### Optimized Processing Pipeline
1. **Fetch & Filter**: Download and filter Firebase data (removes < -200mm readings)
2. **Parse & Sort**: Parse 97K+ readings chronologically with progress tracking
3. **Generate**: Create 18K+ overlapping 96-hour sequences with random offsets
4. **Timestamp Match**: Use binary search to find closest readings for 10-minute intervals
5. **Quality Filter**: Remove sequences with insufficient temporal coverage
6. **Normalize**: Apply z-score normalization using all real sensor data
7. **Temporal Split**: Create train/validation split with proper temporal gap

### Input Processing
- **Real-Time Temporal Alignment**: Uses current inference time to create proper 72-hour timeline
  - Current time → find closest real reading as reference point
  - Generate 433 target times at 10-minute intervals going backwards
  - Match each target time to closest available reading (±5 minute tolerance)
  - Minimal synthetic data (-999) only when genuinely no data available
- **Data Quality Threshold**: Minimum 100 readings required (vs previous 4300+ requirement)
- **Direct Model Input**: `predict_24_hours_direct()` method bypasses input preparation
  - Prevents double processing and data corruption from mean calculations
  - Preserves carefully constructed 433-point temporal sequence
  - Maintains exact -999 synthetic values as model expects
- **Robust Data Handling**: Type safety with comprehensive error handling
- **Normalization**: Z-score using exact training statistics from model checkpoint
- **No Redundant Processing**: Eliminates padding/truncation since sequence is pre-constructed

### Output Processing
- **Direct Prediction**: Full 144-point sequence in single pass (24 hours @ 10-minute intervals)
- **Proper Timestamp Base**: Uses last real data timestamp (not synthetic -999 values)
- **Error Handling**: NaN/infinite predictions converted to -999 (consistent error format)
- **Denormalization**: Convert back to original water level scale (mm)
- **Timestamping**: 10-minute interval predictions starting from last real data point
- **Quality Metrics**: Count of error predictions, synthetic input values, and range validation
- **Enhanced Error Reporting**: Concise error summaries with file:line - ErrorType: message format

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
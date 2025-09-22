# Transformer v2 Shared Components

This directory contains shared code and trained models for transformer v2 that are used across the project.

## Files

### `model.pth`
- **Description**: Best performing transformer v2 model from training
- **Source**: Automatically downloaded after training completion
- **Selection Criteria**: Lowest validation loss during training
- **Usage**: Load for inference, evaluation, or fine-tuning

### `model.py`
- **Description**: TidalTransformerV2 architecture definition
- **Source**: Shared between training, testing, and inference
- **Usage**: Import to create transformer v2 model instances

### `inference.py`
- **Description**: TransformerV2Inference engine for making predictions
- **Source**: Shared between testing and Firebase inference
- **Usage**: Load trained models and generate 24-hour forecasts

### `normalization_params.json`
- **Description**: Data normalization parameters from training
- **Source**: Generated during data preparation phase
- **Usage**: Required for properly normalizing input data and denormalizing outputs

### `data_packaging.py`
- **Description**: Shared data packaging tool for creating inference sequences
- **Source**: Extracted from data preparation logic for reusability
- **Usage**: Creates 432-point normalized sequences using Firebase-identical logic
- **Key Features**: 
  - Automatic Firebase data fetching with freshness checking
  - Binary search-based temporal alignment (±5 minute tolerance)
  - Proper normalization/denormalization handling
  - Missing data management (-999 markers)

## Model Format

The model files are PyTorch checkpoint dictionaries containing:
- `model_state_dict`: Model weights and biases
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: Learning rate scheduler state  
- `epoch`: Training epoch when saved
- `best_val_loss`: Best validation loss achieved
- `config`: Training configuration used
- `train_losses`: Training loss history
- `val_losses`: Validation loss history

## Loading Models

```python
import torch
from model import create_model

# Load checkpoint
checkpoint = torch.load('model.pth')

# Create model with same configuration
model = create_model(checkpoint['config']['model_config'])

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()
```

## Data Packaging Tool Usage

```python
from data_packaging import DataPackager
from datetime import datetime

# Initialize with normalization parameters
packager = DataPackager('path/to/normalization_params.json')

# Load fresh Firebase data (automatically fetches if stale)
firebase_data = packager.load_fresh_firebase_data()

# Create inference sequence for specific time
reference_time = datetime.now()
sequence, metadata = packager.create_inference_sequence(firebase_data, reference_time)

# sequence: 432-point normalized numpy array ready for model input
# metadata: dict with timing info, quality metrics, missing data count

# Denormalize predictions back to mm
predictions_mm = packager.denormalize_sequence(model_output)
```

### DataPackager Class Methods

- `__init__(normalization_params_path)`: Initialize with optional normalization parameters
- `load_normalization_params(path)`: Load normalization parameters from JSON
- `parse_firebase_readings(raw_data)`: Parse raw Firebase data into chronological readings
- `create_inference_sequence(data, time)`: Create 432-point sequence for inference
- `denormalize_sequence(normalized)`: Convert normalized values back to mm
- `load_fresh_firebase_data(path)`: Load Firebase data, fetching fresh if needed

## Model Specifications

- **Architecture**: Encoder-only transformer
- **Input**: 432 time steps (72 hours at 10-minute intervals)
- **Output**: 144 predictions (24 hours at 10-minute intervals)
- **Features**: Single feature (normalized water level)
- **Parameters**: ~12M trainable parameters
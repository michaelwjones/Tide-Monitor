import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

def load_firebase_data():
    """Load and parse Firebase data"""
    try:
        with open('data/firebase_raw_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: firebase_raw_data.json not found. Run fetch_firebase_data.py first.")
        return None

def parse_readings(raw_data):
    """Parse Firebase readings into time series data with 10-minute downsampling"""
    readings = []
    
    for key, reading in raw_data.items():
        # Extract water level (w field) and timestamp
        if 'w' in reading and 't' in reading:
            try:
                timestamp = datetime.fromisoformat(reading['t'].replace('Z', '+00:00'))
                water_level = float(reading['w'])  # Water level in mm
                
                # Basic validation - sensor range is typically 300-5000mm
                if 300 <= water_level <= 5000:
                    readings.append({
                        'timestamp': timestamp,
                        'water_level': water_level
                    })
            except (ValueError, TypeError):
                continue  # Skip invalid readings
    
    # Sort by timestamp
    readings.sort(key=lambda x: x['timestamp'])
    
    # Downsample to 10-minute intervals for maximum training efficiency
    print(f"Original readings: {len(readings)}")
    downsampled = downsample_to_10_minutes(readings)
    print(f"Downsampled to 10-minute intervals: {len(downsampled)}")
    
    return downsampled

def downsample_to_10_minutes(readings):
    """Downsample readings to 10-minute intervals by taking every 10th reading + endpoint"""
    if len(readings) == 0:
        return readings
    
    # Take every 10th reading (assumes 1-minute intervals) + include the last reading
    # This gives us 10-minute intervals which reduces sequence length by 90%
    downsampled = []
    for i in range(0, len(readings), 10):
        downsampled.append(readings[i])
    
    # Include the last reading if it wasn't already included
    if len(readings) % 10 != 0:
        downsampled.append(readings[-1])
    
    return downsampled

def create_sequences(data, input_length=433, output_length=144):
    """
    Create training sequences for seq2seq transformer with 10-minute intervals.
    input_length: Input sequence length (433 = 72 hours at 10 minute intervals + endpoint)
    output_length: Output sequence length (144 = 24 hours at 10 minute intervals)
    """
    sequences = []
    targets = []
    
    print(f"Creating seq2seq sequences: {input_length} inputs → {output_length} outputs (10-minute intervals)...")
    
    for i in range(len(data) - input_length - output_length + 1):
        # Input sequence (72 hours at 10-minute intervals)
        input_seq = data[i:i + input_length]
        
        # Target sequence (next 24 hours at 10-minute intervals for direct prediction)
        target_seq = data[i + input_length:i + input_length + output_length]
        
        sequences.append(input_seq)
        targets.append(target_seq)
        
        # Progress indicator for large datasets
        if (i + 1) % 1000 == 0:
            print(f"  Created {i + 1} sequences...")
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Sequence creation complete:")
    print(f"  Input sequences: {X.shape} (72 hours at 10-min intervals)")
    print(f"  Target sequences: {y.shape} (24 hours at 10-min intervals)")
    
    return X, y

def normalize_data(data):
    """Z-score normalization for transformer training"""
    mean = np.mean(data)
    std = np.std(data)
    
    normalized = (data - mean) / std
    
    # Save normalization parameters
    norm_params = {
        'mean': float(mean),
        'std': float(std)
    }
    
    with open('data/normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    print(f"Normalization parameters saved: mean={mean:.2f}, std={std:.2f}")
    return normalized, norm_params

def create_train_val_split(X, y, val_split=0.2, random_seed=42):
    """
    Create train/validation split for transformer training.
    Uses time-based split to avoid data leakage.
    """
    np.random.seed(random_seed)
    
    # Time-based split: use last 20% of sequences for validation
    split_idx = int(len(X) * (1 - val_split))
    
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    print(f"Train/validation split:")
    print(f"  Training: {len(X_train)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    
    return X_train, X_val, y_train, y_val

def main():
    print("Transformer v1 Training Data Creator")
    print("=" * 45)
    
    # Load Firebase data
    raw_data = load_firebase_data()
    if not raw_data:
        return
    
    # Parse readings
    readings = parse_readings(raw_data)
    print(f"Parsed {len(readings)} valid readings")
    
    min_required = 433 + 144  # input_length + output_length (at 10-minute intervals)
    if len(readings) < min_required:
        print(f"Error: Need at least {min_required} readings (96 hours at 10-min intervals), got {len(readings)}")
        return
    
    # Extract water levels as time series
    water_levels = np.array([r['water_level'] for r in readings])
    timestamps = [r['timestamp'] for r in readings]
    
    print(f"Water level range: {water_levels.min():.1f} - {water_levels.max():.1f} mm")
    
    # Normalize data
    normalized_data, norm_params = normalize_data(water_levels)
    
    # Create training sequences for seq2seq transformer
    print("Creating training sequences (72-hour inputs → 24-hour targets at 10-minute intervals)...")
    X, y = create_sequences(normalized_data, input_length=433, output_length=144)
    
    # Create train/validation split
    X_train, X_val, y_train, y_val = create_train_val_split(X, y)
    
    print(f"Final dataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    
    np.save('data/X_train.npy', X_train.astype(np.float32))
    np.save('data/X_val.npy', X_val.astype(np.float32))
    np.save('data/y_train.npy', y_train.astype(np.float32))
    np.save('data/y_val.npy', y_val.astype(np.float32))
    
    # Save timestamps for reference
    timestamp_data = {
        'timestamps': [t.isoformat() for t in timestamps],
        'input_length': 433,
        'output_length': 144,
        'interval_minutes': 10,
        'total_sequences': len(X),
        'train_sequences': len(X_train),
        'val_sequences': len(X_val),
        'data_range': {
            'start': timestamps[0].isoformat(),
            'end': timestamps[-1].isoformat()
        }
    }
    
    with open('data/timestamps.json', 'w') as f:
        json.dump(timestamp_data, f, indent=2)
    
    print(f"\nTraining data saved:")
    print(f"  X_train.npy: {X_train.nbytes / 1024 / 1024:.1f} MB")
    print(f"  X_val.npy: {X_val.nbytes / 1024 / 1024:.1f} MB")
    print(f"  y_train.npy: {y_train.nbytes / 1024 / 1024:.1f} MB")
    print(f"  y_val.npy: {y_val.nbytes / 1024 / 1024:.1f} MB")
    print(f"  normalization_params.json")
    print(f"  timestamps.json")
    
    print(f"\nNext step: Run training/train_transformer.py to train the model")

if __name__ == "__main__":
    main()
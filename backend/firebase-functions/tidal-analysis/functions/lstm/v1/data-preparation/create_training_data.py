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
    """Parse Firebase readings into time series data"""
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
    return readings

def create_sequences(data, sequence_length=4320, prediction_steps=1):
    """
    Create training sequences for LSTM.
    sequence_length: Input sequence length (4320 = 72 hours at 1 minute intervals)
    prediction_steps: Number of steps to predict ahead (1 for single-step prediction)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length - prediction_steps + 1):
        # Input sequence
        seq = data[i:i + sequence_length]
        
        # Target (next value after sequence)
        target = data[i + sequence_length:i + sequence_length + prediction_steps]
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def normalize_data(data):
    """Z-score normalization"""
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

def main():
    print("LSTM v1 Training Data Creator")
    print("=" * 40)
    
    # Load Firebase data
    raw_data = load_firebase_data()
    if not raw_data:
        return
    
    # Parse readings
    readings = parse_readings(raw_data)
    print(f"Parsed {len(readings)} valid readings")
    
    if len(readings) < 4321:  # Need at least sequence_length + 1 samples
        print(f"Error: Need at least 4321 readings, got {len(readings)}")
        return
    
    # Extract water levels as time series
    water_levels = np.array([r['water_level'] for r in readings])
    timestamps = [r['timestamp'] for r in readings]
    
    print(f"Water level range: {water_levels.min():.1f} - {water_levels.max():.1f} mm")
    
    # Normalize data
    normalized_data, norm_params = normalize_data(water_levels)
    
    # Create training sequences
    print("Creating training sequences (72-hour windows)...")
    X, y = create_sequences(normalized_data, sequence_length=4320)
    
    print(f"Created {len(X)} training sequences")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    
    np.save('data/X_train.npy', X.astype(np.float32))
    np.save('data/y_train.npy', y.astype(np.float32))
    
    # Save timestamps for reference
    timestamp_data = {
        'timestamps': [t.isoformat() for t in timestamps],
        'sequence_length': 4320,
        'total_sequences': len(X),
        'data_range': {
            'start': timestamps[0].isoformat(),
            'end': timestamps[-1].isoformat()
        }
    }
    
    with open('data/timestamps.json', 'w') as f:
        json.dump(timestamp_data, f, indent=2)
    
    print(f"\nTraining data saved:")
    print(f"  X_train.npy: {X.nbytes / 1024 / 1024:.1f} MB")
    print(f"  y_train.npy: {y.nbytes / 1024 / 1024:.1f} MB")
    print(f"  normalization_params.json")
    print(f"  timestamps.json")
    
    print(f"\nNext step: Run training/train_lstm.py to train the model")

if __name__ == "__main__":
    main()
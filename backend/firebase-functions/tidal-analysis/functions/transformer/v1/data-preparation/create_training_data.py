import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random

def load_firebase_data():
    """Load and parse Firebase data (preferring enriched data if available)"""
    # Try to load enriched data first
    try:
        with open('data/firebase_enriched_data.json', 'r') as f:
            print("Loading enriched Firebase data...")
            return json.load(f)
    except FileNotFoundError:
        pass
    
    # Fall back to raw data
    try:
        with open('data/firebase_raw_data.json', 'r') as f:
            print("Loading raw Firebase data...")
            return json.load(f)
    except FileNotFoundError:
        print("Error: No Firebase data found. Run fetch_firebase_data.py first, then enrich_firebase_data.py.")
        return None

def parse_readings(raw_data):
    """Parse Firebase readings into chronological time series data"""
    readings_with_timestamps = []
    synthetic_count = 0
    
    for key, reading in raw_data.items():
        # Extract water level (w field) and timestamp for proper sorting
        if 'w' in reading and 't' in reading:
            try:
                water_level = float(reading['w'])  # Water level in mm
                timestamp = datetime.fromisoformat(reading['t'].replace('Z', '+00:00'))
                
                # Track synthetic readings
                if water_level == -999:
                    synthetic_count += 1
                
                readings_with_timestamps.append({
                    'water_level': water_level,
                    'timestamp': timestamp,
                    'synthetic': water_level == -999
                })
            except (ValueError, TypeError):
                continue  # Skip invalid readings
    
    # Sort by timestamp to ensure chronological order
    readings_with_timestamps.sort(key=lambda x: x['timestamp'])
    
    # Preserve both water levels and timestamps for processing (maintaining order)
    readings = [{'water_level': r['water_level'], 'timestamp': r['timestamp']} for r in readings_with_timestamps]
    
    print(f"Parsed {len(readings)} valid readings in chronological order")
    if synthetic_count > 0:
        print(f"Included {synthetic_count} synthetic readings (-999) for gap filling")
    
    return readings


def generate_sequences_with_random_offsets(readings, total_hours=96, random_seed=42):
    """
    Generate training sequences with random start offsets.
    Each sequence is 96 hours (72h input + 24h output) from original 1-minute data.
    """
    random.seed(random_seed)
    
    sequence_length = total_hours * 60  # 96 hours * 60 minutes = 5760 readings
    sequences = []
    
    start_idx = 0
    sequence_count = 0
    
    while start_idx + sequence_length <= len(readings):
        # Extract 96-hour sequence
        sequence = readings[start_idx:start_idx + sequence_length]
        sequences.append({
            'data': sequence,
            'start_idx': start_idx
        })
        
        sequence_count += 1
        
        # Add random offset (1-9) for next sequence
        start_idx += random.randint(1, 9)
    
    print(f"Generated {len(sequences)} sequences with random offsets")
    return sequences

def interpolate_synthetic_values(sequence_values):
    """
    Interpolate synthetic values (-999) in a sequence using linear interpolation.
    Returns the sequence with interpolated values replacing -999.
    """
    sequence = sequence_values.copy()
    n = len(sequence)
    
    # Find all -999 positions
    synthetic_positions = [i for i, val in enumerate(sequence) if val == -999]
    
    if not synthetic_positions:
        return sequence  # No synthetic values to interpolate
    
    # Handle edge cases where -999 is at the beginning or end
    for i in synthetic_positions:
        if i == 0:
            # Find first non-synthetic value to the right
            for j in range(1, n):
                if sequence[j] != -999:
                    sequence[i] = sequence[j]  # Use first real value
                    break
        elif i == n - 1:
            # Find last non-synthetic value to the left
            for j in range(n - 2, -1, -1):
                if sequence[j] != -999:
                    sequence[i] = sequence[j]  # Use last real value
                    break
        else:
            # Find nearest non-synthetic values on both sides
            left_val = None
            left_idx = None
            for j in range(i - 1, -1, -1):
                if sequence[j] != -999:
                    left_val = sequence[j]
                    left_idx = j
                    break
            
            right_val = None
            right_idx = None
            for j in range(i + 1, n):
                if sequence[j] != -999:
                    right_val = sequence[j]
                    right_idx = j
                    break
            
            # Interpolate between left and right values
            if left_val is not None and right_val is not None:
                # Linear interpolation
                distance = right_idx - left_idx
                position = i - left_idx
                sequence[i] = left_val + (right_val - left_val) * (position / distance)
            elif left_val is not None:
                sequence[i] = left_val  # Use left value if no right value
            elif right_val is not None:
                sequence[i] = right_val  # Use right value if no left value
    
    return sequence

def downsample_sequence(sequence_data):
    """
    Downsample a single 96-hour sequence into 72h input + 24h output at 10-minute intervals.
    Expects exactly 5760 readings (96 hours * 60 minutes).
    Returns: (input_sequence, output_sequence) both with water level values only
    """
    assert len(sequence_data) == 5760, f"Expected 5760 readings, got {len(sequence_data)}"
    
    # Split into 72h input (4320 readings) + 24h output (1440 readings)
    input_raw = sequence_data[:4320]   # First 72 hours
    output_raw = sequence_data[4320:]  # Last 24 hours
    
    # Downsample input: every 10th + last
    input_downsampled = []
    for i in range(0, len(input_raw), 10):
        input_downsampled.append(input_raw[i]['water_level'])
    input_downsampled.append(input_raw[-1]['water_level'])  # Always add last reading
    
    # Downsample output: every 10th reading starting from 9th element
    output_downsampled = []
    for i in range(9, len(output_raw), 10):
        output_downsampled.append(output_raw[i]['water_level'])
    
    # Interpolate synthetic values in output sequence only
    output_interpolated = interpolate_synthetic_values(output_downsampled)
    
    return input_downsampled, output_interpolated


def has_consecutive_synthetic_readings(sequence, max_consecutive=6):
    """
    Check if a sequence has too many consecutive synthetic readings (-999).
    Returns True if max_consecutive or more -999 values appear in a row.
    """
    consecutive_count = 0
    
    for value in sequence:
        if value == -999:
            consecutive_count += 1
            if consecutive_count >= max_consecutive:
                return True
        else:
            consecutive_count = 0
    
    return False

def has_large_time_gaps(sequence_data, max_gap_minutes=15):
    """
    Check if a sequence has time gaps larger than max_gap_minutes between sequential readings.
    Expects sequence_data to be chronologically sorted with timestamp information.
    Returns True if any gap exceeds max_gap_minutes.
    """
    for i in range(1, len(sequence_data)):
        current_time = sequence_data[i]['timestamp']
        previous_time = sequence_data[i-1]['timestamp']
        
        time_diff = (current_time - previous_time).total_seconds() / 60  # Convert to minutes
        
        if time_diff > max_gap_minutes:
            return True
    
    return False

def process_sequences(raw_sequences):
    """
    Process sequences: downsample each individually, filter out sequences 
    with 6+ consecutive synthetic readings or 15+ minute time gaps, and interpolate remaining synthetic values in outputs.
    Returns arrays of training data.
    """
    sequences = []
    targets = []
    filtered_synthetic = 0
    filtered_time_gaps = 0
    interpolated_sequences = 0
    total_interpolated_values = 0
    
    print(f"Processing {len(raw_sequences)} sequences...")
    
    for i, seq_info in enumerate(raw_sequences):
        # Check for large time gaps in original minute-by-minute data
        if has_large_time_gaps(seq_info['data'], max_gap_minutes=15):
            filtered_time_gaps += 1
            continue  # Skip this sequence
        
        # Downsample this sequence (includes interpolation of output synthetic values)
        input_seq, output_seq = downsample_sequence(seq_info['data'])
        
        # Check for consecutive synthetic readings in both input and output
        if (has_consecutive_synthetic_readings(input_seq, max_consecutive=6) or 
            has_consecutive_synthetic_readings(output_seq, max_consecutive=6)):
            filtered_synthetic += 1
            continue  # Skip this sequence
        
        # Count interpolated values in this sequence (before interpolation)
        original_output = []
        for j in range(9, len(seq_info['data'][4320:]), 10):
            original_output.append(seq_info['data'][4320 + j]['water_level'])
        
        synthetic_in_output = sum(1 for val in original_output if val == -999)
        if synthetic_in_output > 0:
            interpolated_sequences += 1
            total_interpolated_values += synthetic_in_output
        
        sequences.append(input_seq)
        targets.append(output_seq)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(raw_sequences)} sequences...")
    
    total_filtered = filtered_synthetic + filtered_time_gaps
    
    print(f"Sequence processing complete:")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Filtered out: {total_filtered} sequences total")
    print(f"    - {filtered_synthetic} for 6+ consecutive synthetic readings")
    print(f"    - {filtered_time_gaps} for time gaps >15 minutes")
    print(f"  Sequences with interpolated values: {interpolated_sequences}")
    print(f"  Total synthetic values interpolated: {total_interpolated_values}")
    print(f"  Retention rate: {len(sequences) / len(raw_sequences) * 100:.1f}%")
    
    return np.array(sequences), np.array(targets)

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
    
    # Parse readings (no longer downsampling here)
    readings = parse_readings(raw_data)
    
    min_required = 96 * 60  # 96 hours * 60 minutes = 5760 readings minimum
    if len(readings) < min_required:
        print(f"Error: Need at least {min_required} readings (96 hours at 1-min intervals), got {len(readings)}")
        return
    
    print(f"Water level range: {min([r['water_level'] for r in readings]):.1f} - {max([r['water_level'] for r in readings]):.1f} mm")
    
    # Generate sequences with random offsets
    raw_sequences = generate_sequences_with_random_offsets(readings, total_hours=96)
    
    # Process sequences: downsample each individually and validate timing
    X, y = process_sequences(raw_sequences)
    
    if len(X) == 0:
        print("Error: No valid sequences found after temporal validation")
        return
    
    print(f"Valid sequences found: {len(X)}")
    print(f"Input shape: {X.shape} (72-hour sequences at 10-min intervals)")
    print(f"Output shape: {y.shape} (24-hour sequences at 10-min intervals)")
    
    # Normalize data (flatten all sequences for global statistics)
    all_values = np.concatenate([X.flatten(), y.flatten()])
    normalized_all, norm_params = normalize_data(all_values)
    
    # Split back into X and y with same normalization
    X_normalized = (X - norm_params['mean']) / norm_params['std']
    y_normalized = (y - norm_params['mean']) / norm_params['std']
    
    # Create train/validation split
    X_train, X_val, y_train, y_val = create_train_val_split(X_normalized, y_normalized)
    
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
    
    # Save metadata
    metadata = {
        'total_raw_readings': len(readings),
        'total_generated_sequences': len(raw_sequences),
        'valid_sequences': len(X),
        'input_length': X.shape[1] if len(X) > 0 else 0,
        'output_length': y.shape[1] if len(y) > 0 else 0,
        'interval_minutes': 10,
        'train_sequences': len(X_train),
        'val_sequences': len(X_val),
        'data_range': {
            'start': 'timestamp_validation_removed',
            'end': 'timestamp_validation_removed'
        },
        'success_rate': len(X) / len(raw_sequences) * 100 if len(raw_sequences) > 0 else 0
    }
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining data saved:")
    print(f"  X_train.npy: {X_train.nbytes / 1024 / 1024:.1f} MB")
    print(f"  X_val.npy: {X_val.nbytes / 1024 / 1024:.1f} MB")
    print(f"  y_train.npy: {y_train.nbytes / 1024 / 1024:.1f} MB")
    print(f"  y_val.npy: {y_val.nbytes / 1024 / 1024:.1f} MB")
    print(f"  normalization_params.json")
    print(f"  metadata.json")
    
    print(f"\nNext step: Run training/train_transformer.py to train the model")

if __name__ == "__main__":
    main()
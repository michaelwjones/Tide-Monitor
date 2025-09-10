import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random

def load_firebase_data():
    """Load filtered Firebase data"""
    try:
        print("Loading filtered Firebase data...")
        with open('data/firebase_filtered_data.json', 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} readings from file")
        return data
    except FileNotFoundError:
        print("Error: firebase_filtered_data.json not found. Run fetch_firebase_data.py first to create filtered data.")
        return None

def parse_readings(raw_data):
    """Parse Firebase readings into chronological time series data"""
    readings_with_timestamps = []
    processed_count = 0
    
    print(f"Parsing {len(raw_data)} readings...")
    
    for key, reading in raw_data.items():
        processed_count += 1
        
        # Progress reporting
        if processed_count % 10000 == 0:
            print(f"  Parsed {processed_count}/{len(raw_data)} readings ({processed_count/len(raw_data)*100:.1f}%)")
        
        # Extract water level (w field) and timestamp for proper sorting
        if 'w' in reading and 't' in reading:
            try:
                water_level = float(reading['w'])  # Water level in mm
                timestamp = datetime.fromisoformat(reading['t'].replace('Z', '+00:00'))
                
                readings_with_timestamps.append({
                    'water_level': water_level,
                    'timestamp': timestamp
                })
            except (ValueError, TypeError):
                continue  # Skip invalid readings
    
    print(f"Sorting {len(readings_with_timestamps)} readings by timestamp...")
    
    # Sort by timestamp to ensure chronological order
    readings_with_timestamps.sort(key=lambda x: x['timestamp'])
    
    # Preserve both water levels and timestamps for processing (maintaining order)
    readings = [{'water_level': r['water_level'], 'timestamp': r['timestamp']} for r in readings_with_timestamps]
    
    print(f"Parsed {len(readings)} valid readings in chronological order")
    
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
            'start_idx': start_idx,
            'start_timestamp': sequence[0]['timestamp'],
            'end_timestamp': sequence[-1]['timestamp']
        })
        
        sequence_count += 1
        
        # Add random offset (1-9) for next sequence
        start_idx += random.randint(1, 9)
    
    print(f"Generated {len(sequences)} sequences with random offsets")
    return sequences



def process_sequences(raw_sequences):
    """
    Process sequences: downsample each individually using timestamp-based matching.
    Returns arrays of training data and sequence metadata with timestamps.
    """
    from datetime import timedelta
    
    sequences = []
    targets = []
    sequence_metadata = []
    filtered_sequences = 0
    
    print(f"Processing {len(raw_sequences)} sequences...")
    
    def find_closest_reading(target_time, data_pool, timestamps_pool, tolerance_minutes=5):
        """Find closest reading to target time within tolerance using binary search"""
        import bisect
        
        if not data_pool:
            return None
            
        tolerance = timedelta(minutes=tolerance_minutes)
        
        # Binary search to find insertion point (timestamps already extracted)
        idx = bisect.bisect_left(timestamps_pool, target_time)
        
        # Check candidates around the insertion point
        candidates = []
        
        # Check reading at idx (first >= target)
        if idx < len(data_pool):
            candidates.append((idx, abs(timestamps_pool[idx] - target_time)))
        
        # Check reading before idx
        if idx > 0:
            candidates.append((idx - 1, abs(timestamps_pool[idx - 1] - target_time)))
        
        # Find closest within tolerance
        best_reading = None
        min_diff = tolerance + timedelta(seconds=1)  # Start beyond tolerance
        
        for candidate_idx, time_diff in candidates:
            if time_diff <= tolerance and time_diff < min_diff:
                min_diff = time_diff
                best_reading = data_pool[candidate_idx]
        
        return best_reading
    
    for i, seq_info in enumerate(raw_sequences):
        sequence_data = seq_info['data']
        
        if len(sequence_data) == 0:
            filtered_sequences += 1
            continue
        
        # Get sequence start time and split into input/output periods
        start_time = sequence_data[0]['timestamp']
        input_end_time = start_time + timedelta(hours=72)
        sequence_end_time = start_time + timedelta(hours=96)
        
        # Split data into input (0-72h) and output (72-96h) periods
        input_data = [reading for reading in sequence_data if reading['timestamp'] < input_end_time]
        output_data = [reading for reading in sequence_data if reading['timestamp'] >= input_end_time]
        
        # Pre-extract timestamps for efficient binary search
        input_timestamps = [reading['timestamp'] for reading in input_data]
        output_timestamps = [reading['timestamp'] for reading in output_data]
        
        # Generate target times at 10-minute intervals for input (72 hours = 433 points)
        input_targets = []
        current_time = start_time
        while current_time < input_end_time:
            input_targets.append(current_time)
            current_time += timedelta(minutes=10)
        # Always include the end time (72 hours)
        input_targets.append(input_end_time)
        
        # Generate target times at 10-minute intervals for output (24 hours = 144 points)  
        output_targets = []
        current_time = input_end_time + timedelta(minutes=10)  # Start 10 minutes into output period
        while current_time <= sequence_end_time:
            output_targets.append(current_time)
            current_time += timedelta(minutes=10)
        
        # Find closest readings for input targets
        input_downsampled = []
        skip_sequence = False
        for target_time in input_targets:
            closest = find_closest_reading(target_time, input_data, input_timestamps)
            if closest:
                input_downsampled.append(closest['water_level'])
            else:
                # If no reading within tolerance, skip this sequence entirely
                skip_sequence = True
                break
        
        if skip_sequence:
            filtered_sequences += 1
            continue
        
        # Find closest readings for output targets  
        output_downsampled = []
        for target_time in output_targets:
            closest = find_closest_reading(target_time, output_data, output_timestamps)
            if closest:
                output_downsampled.append(closest['water_level'])
            else:
                # If no reading within tolerance, skip this sequence entirely
                skip_sequence = True
                break
        
        if skip_sequence:
            filtered_sequences += 1
            continue
        
        sequences.append(input_downsampled)
        targets.append(output_downsampled)
        
        # Store sequence metadata with timestamps
        sequence_metadata.append({
            'start_timestamp': seq_info['start_timestamp'],
            'end_timestamp': seq_info['end_timestamp'],
            'sequence_index': len(sequences) - 1
        })
        
        # Progress indicator  
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(raw_sequences)} sequences... ({((i + 1) / len(raw_sequences) * 100):.1f}%)")
    
    print(f"Sequence processing complete:")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Filtered out: {filtered_sequences} sequences for insufficient data coverage")
    print(f"  Retention rate: {len(sequences) / len(raw_sequences) * 100:.1f}%")
    
    return np.array(sequences), np.array(targets), sequence_metadata

def calculate_normalization_params(data):
    """Calculate normalization parameters excluding -999 missing values"""
    # Filter out -999 missing values from normalization calculation
    valid_data = data[data != -999]
    
    if len(valid_data) == 0:
        raise ValueError("No valid data points found for normalization (all values are -999)")
    
    mean = np.mean(valid_data)
    std = np.std(valid_data)
    
    if std == 0:
        print("Warning: Standard deviation is 0, using std=1.0 to avoid division by zero")
        std = 1.0
    
    # Save normalization parameters
    norm_params = {
        'mean': float(mean),
        'std': float(std)
    }
    
    with open('data/normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    total_values = len(data)
    valid_values = len(valid_data)
    missing_count = total_values - valid_values
    
    print(f"Normalization parameters (excluding {missing_count} missing values from {total_values} total):")
    print(f"  Mean: {mean:.2f}")
    print(f"  Std: {std:.2f}")
    print(f"  Valid data: {valid_values} values ({valid_values/total_values*100:.1f}%)")
    
    return norm_params

def apply_normalization(data, mean, std):
    """Apply normalization while preserving -999 missing values"""
    normalized = data.copy()
    
    # Only normalize non-missing values
    valid_mask = (data != -999)
    normalized[valid_mask] = (data[valid_mask] - mean) / std
    
    return normalized

def create_train_val_split(X, y, sequence_metadata, val_split=0.2, random_seed=42):
    """
    Create train/validation split for transformer training with proper temporal gap.
    Uses timestamps to ensure no data leakage between overlapping sequences.
    """
    np.random.seed(random_seed)
    
    total_sequences = len(X)
    target_val_size = int(total_sequences * val_split)
    
    # Find the 80% cutoff point based on sequence count
    potential_train_end = total_sequences - target_val_size
    
    # Get the end timestamp of the last training sequence
    last_train_end_time = sequence_metadata[potential_train_end - 1]['end_timestamp']
    
    print(f"Last training sequence ends at: {last_train_end_time}")
    
    # Find first validation sequence that starts AFTER the last training sequence ends
    actual_val_start = None
    for i in range(potential_train_end, total_sequences):
        val_start_time = sequence_metadata[i]['start_timestamp']
        if val_start_time >= last_train_end_time:
            actual_val_start = i
            print(f"First validation sequence starts at: {val_start_time}")
            break
    
    if actual_val_start is None:
        raise ValueError("Cannot create non-overlapping split - insufficient temporal separation")
    
    # Calculate the gap
    gap_sequences = actual_val_start - potential_train_end
    
    # Create the splits
    X_train = X[:potential_train_end]
    X_val = X[actual_val_start:]
    y_train = y[:potential_train_end]
    y_val = y[actual_val_start:]
    
    # Calculate actual temporal span of discarded sequences
    if gap_sequences > 0:
        first_discarded_start = sequence_metadata[potential_train_end]['start_timestamp'] 
        last_discarded_end = sequence_metadata[actual_val_start - 1]['end_timestamp']
        discarded_timespan_hours = (last_discarded_end - first_discarded_start).total_seconds() / 3600
        
        # Also calculate boundary gap for reference
        boundary_gap_start = sequence_metadata[potential_train_end - 1]['end_timestamp']
        boundary_gap_end = sequence_metadata[actual_val_start]['start_timestamp']  
        boundary_gap_hours = (boundary_gap_end - boundary_gap_start).total_seconds() / 3600
    else:
        discarded_timespan_hours = 0
        boundary_gap_hours = 0
    
    print(f"Train/validation split with temporal gap:")
    print(f"  Training: {len(X_train)} sequences")
    if gap_sequences > 0:
        print(f"  Gap (discarded): {gap_sequences} sequences spanning {discarded_timespan_hours:.1f} hours")
        print(f"  Boundary gap: {boundary_gap_hours:.1f} hours between last training and first validation")
    else:
        print(f"  Gap (discarded): 0 sequences (perfect temporal alignment)")
    print(f"  Validation: {len(X_val)} sequences")
    print(f"  Temporal separation ensured: No data leakage between train/val")
    
    return X_train, X_val, y_train, y_val

def main():
    print("Transformer v1 Training Data Creator")
    print("=" * 45)
    
    # Load Firebase data
    raw_data = load_firebase_data()
    if not raw_data:
        return
    
    # Parse readings (no longer downsampling here)
    print("Parsing readings...")
    readings = parse_readings(raw_data)
    
    min_required = 96 * 6  # 96 hours with readings every 10 minutes minimum (conservative estimate)
    if len(readings) < min_required:
        print(f"Error: Need at least {min_required} readings for 96-hour sequences, got {len(readings)}")
        return
    
    print(f"Water level range: {min([r['water_level'] for r in readings]):.1f} - {max([r['water_level'] for r in readings]):.1f} mm")
    
    # Generate sequences with random offsets
    print("Generating sequences with random offsets...")
    raw_sequences = generate_sequences_with_random_offsets(readings, total_hours=96)
    
    # Process sequences: downsample each individually and validate timing
    X, y, sequence_metadata = process_sequences(raw_sequences)
    
    if len(X) == 0:
        print("Error: No valid sequences found after temporal validation")
        return
    
    print(f"Valid sequences found: {len(X)}")
    print(f"Input shape: {X.shape} (72-hour sequences at 10-min intervals)")
    print(f"Output shape: {y.shape} (24-hour sequences at 10-min intervals)")
    
    # Calculate normalization parameters (flatten all sequences for global statistics)
    # This happens BEFORE train/val split to ensure consistent normalization parameters
    print("Calculating normalization parameters...")
    all_values = np.concatenate([X.flatten(), y.flatten()])
    norm_params = calculate_normalization_params(all_values)
    
    # Apply normalization while preserving -999 missing values
    print("Applying normalization (preserving -999 missing values)...")
    X_normalized = apply_normalization(X, norm_params['mean'], norm_params['std'])
    y_normalized = apply_normalization(y, norm_params['mean'], norm_params['std'])
    
    # Create train/validation split with proper temporal gap
    print("Creating train/validation split...")
    X_train, X_val, y_train, y_val = create_train_val_split(X_normalized, y_normalized, sequence_metadata)
    
    print(f"Final dataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    
    # Save training data
    print("Saving training data files...")
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
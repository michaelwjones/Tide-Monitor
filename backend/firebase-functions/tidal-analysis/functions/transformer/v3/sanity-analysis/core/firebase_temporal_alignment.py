"""
Firebase-identical temporal alignment logic extracted from main.py
This replicates exactly how the Firebase function creates 432-point sequences
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

def extract_firebase_sequence(readings_with_timestamps: List[Dict], reference_time: datetime) -> Tuple[List[float], List[Dict]]:
    """
    Extract 432-point sequence using identical logic to Firebase function main.py:343-400
    
    Args:
        readings_with_timestamps: List of {'water_level': float, 'timestamp': datetime}
        reference_time: Reference time to work backwards from
        
    Returns:
        (water_levels, sequence_info): 432 water levels and metadata for each point
    """
    
    print(f"Creating 432-point sequence from reference time: {reference_time}")
    
    # Count backwards in 10-minute intervals, finding closest readings
    # This is identical to main.py:374-397
    downsampled_readings = []
    
    for i in range(432):  # v2 uses 432 inputs (not 433 like v1)
        # Calculate target time (going backwards from reference time)
        target_time = reference_time - timedelta(minutes=i * 10)
        
        # Find closest reading within ±5 minutes of this target time
        closest_reading = None
        min_diff = float('inf')
        
        for reading in readings_with_timestamps:
            time_diff = abs((reading['timestamp'] - target_time).total_seconds())
            if time_diff <= 300 and time_diff < min_diff:  # Within ±5 minutes
                min_diff = time_diff
                closest_reading = reading
        
        # Use closest reading if found, otherwise -999
        if closest_reading:
            water_level = closest_reading['water_level']
        else:
            water_level = -999
        
        downsampled_readings.append({
            'water_level': water_level,
            'timestamp': target_time,
            'original_reading': closest_reading,
            'time_diff_seconds': min_diff if closest_reading else None
        })
    
    # Reverse to get chronological order (oldest to newest)
    # This is identical to main.py:400
    downsampled_readings.reverse()
    
    synthetic_count = sum(1 for r in downsampled_readings if r['water_level'] == -999)
    print(f"Created 432-point sequence with {synthetic_count} missing data points (-999)")
    print(f"Time range: {downsampled_readings[0]['timestamp']} to {downsampled_readings[-1]['timestamp']}")
    
    # Extract water level values, preserving -999 synthetic values
    water_level_values = [r['water_level'] for r in downsampled_readings]
    
    return water_level_values, downsampled_readings

def find_reference_time_firebase_style(readings_with_timestamps: List[Dict], target_time: datetime) -> datetime:
    """
    Find reference time using identical logic to Firebase function main.py:354-369
    
    Args:
        readings_with_timestamps: List of readings with timestamps
        target_time: The time we want to find closest reading to
        
    Returns:
        Reference timestamp from closest reading
    """
    
    # Find the closest reading to target time to establish our starting point
    # This is identical to main.py:355-361
    closest_to_target = None
    min_diff = float('inf')
    for reading in readings_with_timestamps:
        time_diff = abs((reading['timestamp'] - target_time).total_seconds())
        if time_diff < min_diff:
            min_diff = time_diff
            closest_to_target = reading
    
    # Use the timestamp of the closest reading as our reference point
    if closest_to_target:
        reference_time = closest_to_target['timestamp']
        print(f"Reference time (closest to target {target_time}): {reference_time}")
        return reference_time
    else:
        print(f"No readings found, using target time: {target_time}")
        return target_time

def parse_readings_firebase_style(raw_data: Dict) -> List[Dict]:
    """
    Parse Firebase readings using identical logic to main.py:304-331
    
    Args:
        raw_data: Raw Firebase readings data
        
    Returns:
        List of parsed readings with timestamps, sorted chronologically
    """
    
    # Parse readings with timestamps (consistent with v1 approach)
    # This replicates main.py:304-331
    readings_with_timestamps = []
    for key, value in raw_data.items():
        if 'w' in value and 't' in value:
            try:
                # Convert water level, allowing -999 synthetic values
                raw_value = value['w']
                if raw_value is None:
                    continue
                
                if isinstance(raw_value, str):
                    if raw_value.strip() == '':
                        continue
                    water_level = float(raw_value)
                else:
                    water_level = float(raw_value)
                
                # Parse timestamp
                timestamp = datetime.fromisoformat(value['t'].replace('Z', '+00:00'))
                
                # Include all values, including -999 synthetic ones
                readings_with_timestamps.append({
                    'water_level': water_level,
                    'timestamp': timestamp
                })
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid reading: {raw_value}, error: {e}")
                continue
    
    # Sort by timestamp to ensure chronological order
    # This replicates main.py:337-338
    readings_with_timestamps.sort(key=lambda x: x['timestamp'])
    
    print(f"Processing {len(readings_with_timestamps)} chronologically sorted readings")
    if readings_with_timestamps:
        print(f"Time range: {readings_with_timestamps[0]['timestamp']} to {readings_with_timestamps[-1]['timestamp']}")
    
    return readings_with_timestamps

def validate_sequence_quality(water_levels: List[float], sequence_info: List[Dict]) -> Dict:
    """
    Validate sequence quality with metrics similar to Firebase function
    
    Returns:
        Dictionary with quality metrics
    """
    
    synthetic_count = sum(1 for val in water_levels if val == -999)
    real_values = [v for v in water_levels if v != -999]
    real_data_percentage = len(real_values) / len(water_levels) * 100 if water_levels else 0
    
    # Check time gaps in original readings
    time_gaps = []
    for i in range(1, len(sequence_info)):
        if (sequence_info[i]['original_reading'] and 
            sequence_info[i-1]['original_reading']):
            gap = abs((sequence_info[i]['original_reading']['timestamp'] - 
                      sequence_info[i-1]['original_reading']['timestamp']).total_seconds())
            time_gaps.append(gap)
    
    quality_metrics = {
        'total_points': len(water_levels),
        'synthetic_count': synthetic_count,
        'real_data_percentage': real_data_percentage,
        'water_level_range': (min(real_values), max(real_values)) if real_values else (None, None),
        'avg_time_gap_seconds': np.mean(time_gaps) if time_gaps else None,
        'max_time_gap_seconds': max(time_gaps) if time_gaps else None,
        'time_coverage_start': sequence_info[0]['timestamp'],
        'time_coverage_end': sequence_info[-1]['timestamp']
    }
    
    return quality_metrics

def create_firebase_input_sequence(raw_data: Dict, reference_time: datetime) -> Tuple[List[float], Dict]:
    """
    Create input sequence using complete Firebase function logic
    
    Args:
        raw_data: Raw Firebase readings
        reference_time: Time to create sequence for
        
    Returns:
        (water_levels, metadata): 432-point sequence and quality metrics
    """
    
    # Step 1: Parse readings exactly like Firebase function
    readings_with_timestamps = parse_readings_firebase_style(raw_data)
    
    if len(readings_with_timestamps) < 100:
        raise ValueError(f"Insufficient data points: {len(readings_with_timestamps)} (need at least 100)")
    
    # Step 2: Find reference time exactly like Firebase function
    firebase_reference_time = find_reference_time_firebase_style(readings_with_timestamps, reference_time)
    
    # Step 3: Extract sequence exactly like Firebase function
    water_levels, sequence_info = extract_firebase_sequence(readings_with_timestamps, firebase_reference_time)
    
    # Step 4: Validate quality
    quality_metrics = validate_sequence_quality(water_levels, sequence_info)
    
    print(f"Data quality: {quality_metrics['real_data_percentage']:.1f}% real data")
    if quality_metrics['water_level_range'][0] is not None:
        print(f"Water level range: {quality_metrics['water_level_range'][0]:.1f} - {quality_metrics['water_level_range'][1]:.1f} mm")
    
    return water_levels, {
        'sequence_info': sequence_info,
        'quality_metrics': quality_metrics,
        'reference_time_used': firebase_reference_time,
        'reference_time_requested': reference_time
    }

if __name__ == "__main__":
    # Test the Firebase temporal alignment logic
    import json
    
    # Load test data
    data_path = "../data-preparation/data/firebase_filtered_data.json"
    try:
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Test with current time (use timezone from data)
        from datetime import timezone
        test_time = datetime.now().replace(tzinfo=timezone.utc)
        
        print("Testing Firebase temporal alignment logic...")
        water_levels, metadata = create_firebase_input_sequence(raw_data, test_time)
        
        print(f"\nSuccess!")
        print(f"Generated {len(water_levels)} water level readings")
        print(f"Quality: {metadata['quality_metrics']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
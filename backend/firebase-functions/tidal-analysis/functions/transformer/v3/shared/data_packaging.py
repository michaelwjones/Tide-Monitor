#!/usr/bin/env python3
"""
Shared Data Packaging Tool for Transformer v2
Extracts and standardizes the data packaging logic used across training, testing, and inference.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import bisect

class DataPackager:
    """
    Shared data packaging tool for transformer v2 inference sequences.
    Uses the same logic as data preparation for consistency.
    """
    
    def __init__(self, normalization_params_path: Optional[str] = None):
        """
        Initialize data packager with optional normalization parameters
        
        Args:
            normalization_params_path: Path to normalization params JSON file
        """
        self.norm_params = None
        if normalization_params_path:
            self.load_normalization_params(normalization_params_path)
    
    def load_normalization_params(self, params_path: str):
        """Load normalization parameters from JSON file"""
        try:
            with open(params_path, 'r') as f:
                self.norm_params = json.load(f)
            print(f"Loaded normalization params: mean={self.norm_params['mean']:.2f}, std={self.norm_params['std']:.2f}")
        except Exception as e:
            print(f"Warning: Could not load normalization params: {e}")
            self.norm_params = None
    
    def parse_firebase_readings(self, raw_data: Dict) -> List[Dict]:
        """
        Parse Firebase readings into chronological time series data.
        Extracted from create_training_data.py parse_readings()
        """
        readings_with_timestamps = []
        
        for key, reading in raw_data.items():
            try:
                # Parse timestamp
                timestamp_str = reading.get('t', '')
                if not timestamp_str:
                    continue
                    
                # Handle timezone format
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                
                timestamp = datetime.fromisoformat(timestamp_str)
                
                # Extract water level
                water_level = reading.get('w')
                if water_level is None or water_level == -999:
                    continue
                
                readings_with_timestamps.append({
                    'timestamp': timestamp,
                    'water_level': float(water_level)
                })
                
            except (ValueError, TypeError) as e:
                continue
        
        # Sort chronologically
        readings_with_timestamps.sort(key=lambda x: x['timestamp'])
        return readings_with_timestamps
    
    def create_inference_sequence(self, raw_firebase_data: Dict, reference_time: datetime) -> Tuple[np.ndarray, Dict]:
        """
        Create 432-point input sequence for inference using exact data preparation logic.
        
        Args:
            raw_firebase_data: Raw Firebase readings
            reference_time: Reference time to create sequence for
            
        Returns:
            (sequence, metadata): 432-point normalized sequence and metadata
        """
        # Parse readings
        readings = self.parse_firebase_readings(raw_firebase_data)
        
        if len(readings) < 1000:
            raise ValueError(f"Insufficient data: {len(readings)} readings (need at least 1000)")
        
        # Extract timestamps and water levels for binary search
        timestamps = [r['timestamp'] for r in readings]
        water_levels = [r['water_level'] for r in readings]
        
        # Generate target times at 10-minute intervals for input (72 hours = 432 points)
        # This is the exact logic from create_training_data.py process_sequences()
        input_targets = []
        
        # Start 72 hours before reference time, offset by 9 minutes for alignment
        actual_input_start = reference_time - timedelta(hours=72) + timedelta(minutes=9)
        current_time = actual_input_start
        
        for _ in range(432):
            input_targets.append(current_time)
            current_time += timedelta(minutes=10)
        
        # Use binary search to find closest readings (same as data preparation)
        input_sequence = []
        missing_count = 0
        
        for target_time in input_targets:
            # Binary search for closest timestamp
            pos = bisect.bisect_left(timestamps, target_time)
            
            best_idx = None
            best_diff = float('inf')
            
            # Check positions around the binary search result
            for check_pos in [pos - 1, pos, pos + 1]:
                if 0 <= check_pos < len(timestamps):
                    diff = abs((timestamps[check_pos] - target_time).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = check_pos
            
            # Use closest reading if within 5 minutes (300 seconds)
            if best_idx is not None and best_diff <= 300:
                input_sequence.append(water_levels[best_idx])
            else:
                input_sequence.append(-999)  # Missing data marker
                missing_count += 1
        
        # Convert to numpy array
        sequence_array = np.array(input_sequence)
        
        # Apply normalization if available
        if self.norm_params:
            # Replace -999 with mean before normalization
            normalized_sequence = sequence_array.copy().astype(float)
            missing_mask = normalized_sequence == -999
            
            if np.any(~missing_mask):
                # Normalize non-missing values
                normalized_sequence[~missing_mask] = (
                    normalized_sequence[~missing_mask] - self.norm_params['mean']
                ) / self.norm_params['std']
                
                # Replace missing values with 0 (normalized mean)
                normalized_sequence[missing_mask] = 0.0
            else:
                # All values missing - return zeros
                normalized_sequence = np.zeros_like(normalized_sequence)
        else:
            normalized_sequence = sequence_array
        
        # Create metadata
        metadata = {
            'reference_time': reference_time.isoformat(),
            'sequence_start': actual_input_start.isoformat(),
            'sequence_end': input_targets[-1].isoformat(),
            'total_points': 432,
            'missing_points': missing_count,
            'data_quality': f"{(432-missing_count)/432*100:.1f}%",
            'water_level_range': f"{np.min(sequence_array[sequence_array != -999]):.1f} - {np.max(sequence_array[sequence_array != -999]):.1f} mm" if np.any(sequence_array != -999) else "No valid data",
            'normalized': self.norm_params is not None
        }
        
        return normalized_sequence, metadata
    
    def denormalize_sequence(self, normalized_sequence: np.ndarray) -> np.ndarray:
        """
        Denormalize sequence back to original water level units
        
        Args:
            normalized_sequence: Normalized sequence
            
        Returns:
            Denormalized sequence in mm
        """
        if self.norm_params is None:
            return normalized_sequence
            
        return normalized_sequence * self.norm_params['std'] + self.norm_params['mean']
    
    def load_fresh_firebase_data(self, firebase_data_path: Optional[str] = None) -> Dict:
        """
        Load Firebase data, fetching fresh data if needed
        
        Args:
            firebase_data_path: Path to cached Firebase data file
            
        Returns:
            Raw Firebase data dictionary
        """
        if firebase_data_path is None:
            # Default path relative to shared directory
            firebase_data_path = Path(__file__).parent.parent / 'data-preparation' / 'data' / 'firebase_filtered_data.json'
        
        firebase_data_path = Path(firebase_data_path)
        
        # Check if we need fresh data (same logic as discontinuity analysis)
        need_fresh_data = False
        
        if not firebase_data_path.exists():
            need_fresh_data = True
        else:
            try:
                with open(firebase_data_path, 'r') as f:
                    existing_data = json.load(f)
                
                # Find latest timestamp
                latest_timestamp = None
                for reading_data in existing_data.values():
                    if 't' in reading_data:
                        try:
                            timestamp = datetime.fromisoformat(reading_data['t'].replace('Z', '+00:00'))
                            if latest_timestamp is None or timestamp > latest_timestamp:
                                latest_timestamp = timestamp
                        except:
                            continue
                
                if latest_timestamp:
                    days_old = (datetime.now(latest_timestamp.tzinfo) - latest_timestamp).days
                    if days_old > 1:
                        need_fresh_data = True
                else:
                    need_fresh_data = True
                    
            except Exception:
                need_fresh_data = True
        
        # Fetch fresh data if needed
        if need_fresh_data:
            print("Fetching fresh Firebase data...")
            try:
                import subprocess
                
                # Run fetch_firebase_data.py
                data_prep_dir = firebase_data_path.parent.parent
                fetch_script = data_prep_dir / 'fetch_firebase_data.py'
                
                if fetch_script.exists():
                    result = subprocess.run([
                        'python', str(fetch_script)
                    ], cwd=str(data_prep_dir), capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print("Fresh Firebase data fetched successfully")
                    else:
                        print(f"Warning: Firebase fetch failed: {result.stderr}")
                else:
                    print(f"Warning: Fetch script not found at {fetch_script}")
                    
            except Exception as e:
                print(f"Warning: Could not fetch fresh data: {e}")
        
        # Load the data
        try:
            with open(firebase_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Could not load Firebase data from {firebase_data_path}: {e}")

# Convenience functions for backward compatibility
def create_inference_sequence(raw_firebase_data: Dict, reference_time: datetime, 
                            normalization_params_path: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to create inference sequence
    """
    packager = DataPackager(normalization_params_path)
    return packager.create_inference_sequence(raw_firebase_data, reference_time)

def load_fresh_firebase_data(firebase_data_path: Optional[str] = None) -> Dict:
    """
    Convenience function to load fresh Firebase data
    """
    packager = DataPackager()
    return packager.load_fresh_firebase_data(firebase_data_path)
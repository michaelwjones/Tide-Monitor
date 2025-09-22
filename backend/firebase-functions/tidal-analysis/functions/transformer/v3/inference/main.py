"""
Firebase Functions Python runtime for Transformer v2 tidal predictions.
Uses single-pass encoder-only transformer for direct 432â†’144 prediction.
"""
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from firebase_functions import scheduler_fn
from firebase_admin import initialize_app, db
from google.cloud import storage

# Initialize Firebase
initialize_app()

# Lazy imports to avoid PyTorch initialization timeout
torch = None
np = None
TransformerV2Inference = None

def _lazy_imports():
    """Lazy load PyTorch and model dependencies to avoid initialization timeout"""
    global torch, np, TransformerV2Inference
    if torch is None:
        print("Lazy loading PyTorch and transformer v2 dependencies...")
        
        # Set PyTorch to CPU-only for faster initialization
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        import torch as _torch
        import numpy as _np
        
        # Import from shared directory
        shared_dir = Path(__file__).parent.parent / "shared"
        sys.path.insert(0, str(shared_dir))
        from inference import TransformerV2Inference as _TransformerV2Inference
        
        # Set CPU-only threads for Firebase Functions
        _torch.set_num_threads(2)
        
        torch = _torch
        np = _np
        TransformerV2Inference = _TransformerV2Inference
        print("PyTorch and transformer v2 loaded successfully (CPU-only mode)")


class TransformerV2Inferencer:
    """Single-pass transformer v2 inferencer for Firebase Functions"""
    
    def __init__(self):
        self.inference_engine = None
        self.input_length = 432   # 72 hours @ 10min intervals
        self.output_length = 144  # 24 hours @ 10min intervals
        
        # Firebase Storage configuration (will auto-detect bucket name)
        self.project_id = "tide-monitor-boron"
        self.storage_model_path = "transformer-v2-models/model.pth"
        self.storage_params_path = "transformer-v2-models/normalization_params.json"
        
        # Local cache paths in /tmp (persists across invocations)
        self.local_model_path = "/tmp/transformer-v2-model.pth"
        self.local_params_path = "/tmp/transformer-v2-normalization_params.json"
        
    def download_model_from_storage(self):
        """Download model files from Firebase Storage if not cached locally"""
        try:
            print("Checking for cached model files...")
            
            # Check if both files are already cached
            if (os.path.exists(self.local_model_path) and 
                os.path.exists(self.local_params_path)):
                print(f"Using cached model from {self.local_model_path}")
                return True
            
            print("Downloading model from Firebase Storage...")
            
            # Initialize Storage client with project ID
            storage_client = storage.Client(project=self.project_id)
            
            # Try to find the correct bucket name
            bucket_names = ["tide-monitor-boron.firebasestorage.app", "tide-monitor-boron.appspot.com", "tide-monitor-boron"]
            
            bucket = None
            for bucket_name in bucket_names:
                try:
                    test_bucket = storage_client.bucket(bucket_name)
                    test_bucket.exists()  # This will raise exception if bucket doesn't exist
                    bucket = test_bucket
                    print(f"  Using Firebase Storage bucket: {bucket_name}")
                    break
                except Exception:
                    continue
            
            if not bucket:
                raise RuntimeError(f"Could not find Firebase Storage bucket for project {self.project_id}")
            
            # Download model file
            if not os.path.exists(self.local_model_path):
                print(f"  Downloading model.pth (306MB)...")
                model_blob = bucket.blob(self.storage_model_path)
                model_blob.download_to_filename(self.local_model_path)
                
                # Verify download
                model_size = os.path.getsize(self.local_model_path) / (1024 * 1024)
                print(f"  Model downloaded: {model_size:.1f} MB")
            
            # Download normalization parameters
            if not os.path.exists(self.local_params_path):
                print(f"  Downloading normalization_params.json...")
                params_blob = bucket.blob(self.storage_params_path)
                params_blob.download_to_filename(self.local_params_path)
                print(f"  Normalization params downloaded")
            
            print("Model download completed successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to download model from Storage: {e} (main.py:87)")
            return False
        
    def initialize(self):
        """Load the trained transformer v2 model (download from Storage if needed)"""
        try:
            # Lazy load PyTorch dependencies
            _lazy_imports()
            
            # Try to download model from Firebase Storage if not cached
            if not self.download_model_from_storage():
                raise RuntimeError("Could not download model from Firebase Storage")
            
            print(f"Loading transformer v2 from cached files...")
            print(f"Model: {self.local_model_path}")
            print(f"Normalization: {self.local_params_path}")
            
            # Create inference engine using downloaded/cached files
            self.inference_engine = TransformerV2Inference(self.local_model_path, self.local_params_path)
            
            print(f"Transformer v2 inference engine initialized successfully!")
            print(f"Architecture: encoder-only transformer")
            print(f"Input length: {self.input_length}")
            print(f"Output length: {self.output_length}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize transformer v2: {e} (main.py:124)")
            return False
    
    def prepare_input_sequence(self, water_levels):
        """Prepare input sequence for transformer v2 inference"""
        input_sequence = list(water_levels)
        
        # Handle insufficient data by padding with -999 (missing value marker)
        if len(input_sequence) < self.input_length:
            print(f"Input has {len(input_sequence)} readings, need {self.input_length}")
            
            pad_length = self.input_length - len(input_sequence)
            print(f"Padding with {pad_length} missing value markers (-999)")
            
            # Pad at the beginning with -999 values
            padding = [-999] * pad_length
            input_sequence = padding + input_sequence
        
        # Handle excess data by taking most recent readings
        if len(input_sequence) > self.input_length:
            print(f"Truncating from {len(input_sequence)} to {self.input_length} readings")
            input_sequence = input_sequence[-self.input_length:]
        
        print(f"Prepared input sequence: {len(input_sequence)} readings")
        valid_readings = [x for x in input_sequence if x != -999]
        if valid_readings:
            print(f"Valid data range: {min(valid_readings):.1f} - {max(valid_readings):.1f} mm")
        
        return input_sequence
    
    def predict_24_hours(self, water_levels, last_data_timestamp=None):
        """Generate 24-hour predictions using transformer v2 model"""
        if self.inference_engine is None:
            raise RuntimeError("Transformer v2 inference engine not initialized (main.py:115)")
        
        try:
            print("Starting transformer v2 24-hour prediction...")
            start_time = datetime.now()
            
            # Prepare input sequence
            if len(water_levels) == self.input_length:
                print("Using provided 432-value sequence directly")
                input_sequence = water_levels
            else:
                print(f"Preparing sequence from {len(water_levels)} input values")
                input_sequence = self.prepare_input_sequence(water_levels)
            
            # Generate predictions using shared inference engine
            predictions = self.inference_engine.predict(np.array(input_sequence))
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            print(f"Transformer v2 inference completed in {inference_time:.1f}ms")
            
            # Convert to list and validate predictions
            prediction_list = predictions.tolist()
            validated_predictions = []
            
            for pred in prediction_list:
                try:
                    if np.isfinite(pred) and not np.isnan(pred):
                        validated_predictions.append(float(pred))
                    else:
                        print(f"Invalid prediction: {pred}, replacing with -999")
                        validated_predictions.append(-999)
                except (ValueError, TypeError):
                    print(f"Non-numeric prediction: {pred}, replacing with -999")
                    validated_predictions.append(-999)
            
            valid_count = sum(1 for p in validated_predictions if p != -999)
            error_count = len(validated_predictions) - valid_count
            
            print(f"Generated {len(validated_predictions)} predictions")
            print(f"Valid predictions: {valid_count}, errors: {error_count}")
            
            valid_preds = [p for p in validated_predictions if p != -999]
            if valid_preds:
                print(f"Prediction range: {min(valid_preds):.1f} - {max(valid_preds):.1f} mm")
            
            # Create timestamped predictions
            if last_data_timestamp:
                try:
                    base_time = datetime.fromisoformat(last_data_timestamp.replace('Z', '+00:00'))
                except ValueError:
                    base_time = datetime.now()
                    print(f"Failed to parse timestamp {last_data_timestamp}, using current time")
            else:
                base_time = datetime.now()
                print("No last data timestamp provided, using current time")
            
            timestamped_predictions = []
            
            for i, prediction in enumerate(validated_predictions):
                # 10-minute intervals starting from the last data point
                prediction_time = base_time + timedelta(minutes=(i + 1) * 10)
                
                timestamped_predictions.append({
                    'timestamp': prediction_time.isoformat(),
                    'prediction': prediction,
                    'step': i + 1
                })
            
            return {
                'predictions': timestamped_predictions,
                'metadata': {
                    'inference_time_ms': inference_time,
                    'input_length': len(input_sequence),
                    'output_length': len(validated_predictions),
                    'model_architecture': 'single_pass_encoder_transformer_v2',
                    'model_version': 'transformer-v2-single-pass',
                    'error_predictions': error_count,
                    'input_synthetic_count': sum(1 for x in input_sequence if x == -999)
                }
            }
            
        except Exception as e:
            print(f"Transformer v2 prediction error: {e}")
            raise


# Global inferencer instance
inferencer = None

def get_inferencer():
    """Get or create the global transformer v2 inferencer instance"""
    global inferencer
    if inferencer is None:
        inferencer = TransformerV2Inferencer()
        if not inferencer.initialize():
            raise RuntimeError("Failed to initialize transformer v2 inferencer")
    return inferencer


@scheduler_fn.on_schedule(schedule="*/5 * * * *", memory=2048, timeout_sec=540)
def run_transformer_v2_analysis(req):
    """Scheduled function to run transformer v2 predictions every 5 minutes"""
    
    try:
        print("Starting Transformer v2 24-hour prediction run...")
        start_time = datetime.now()
        
        # Get inferencer instance
        inferencer_instance = get_inferencer()
        
        # Fetch historical data from Firebase
        print("Fetching historical data from Firebase...")
        
        readings_ref = db.reference('readings')
        readings = readings_ref.order_by_key().limit_to_last(4320).get()
        
        if not readings:
            print("No readings data available")
            return
        
        print(f"Raw readings retrieved: {len(readings)}")
        
        # Parse readings with timestamps (consistent with v1 approach)
        readings_with_timestamps = []
        for key, value in readings.items():
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
        
        if len(readings_with_timestamps) < 100:  # Need minimum viable data for prediction
            print(f"Insufficient data points: {len(readings_with_timestamps)} (need at least 100 readings for prediction)")
            return
        
        # Sort by timestamp to ensure chronological order
        readings_with_timestamps.sort(key=lambda x: x['timestamp'])
        
        print(f"Processing {len(readings_with_timestamps)} chronologically sorted readings")
        print(f"Time range: {readings_with_timestamps[0]['timestamp']} to {readings_with_timestamps[-1]['timestamp']}")
        
        # Create 432-point sequence starting from current time, working backwards at 10-minute intervals
        print(f"Creating 432-point sequence from current time backwards...")
        
        # Get the current time
        if readings_with_timestamps:
            # Use timezone from the data
            current_time = datetime.now().replace(tzinfo=readings_with_timestamps[0]['timestamp'].tzinfo)
        else:
            from datetime import timezone
            current_time = datetime.now().replace(tzinfo=timezone.utc)
        
        # Find the closest reading to current time to establish our starting point
        closest_to_now = None
        min_diff = float('inf')
        for reading in readings_with_timestamps:
            time_diff = abs((reading['timestamp'] - current_time).total_seconds())
            if time_diff < min_diff:
                min_diff = time_diff
                closest_to_now = reading
        
        # Use the timestamp of the closest reading as our reference point
        if closest_to_now:
            reference_time = closest_to_now['timestamp']
            print(f"Reference time (closest to now): {reference_time}")
        else:
            reference_time = current_time
            print(f"No readings found, using current time: {reference_time}")
        
        # Count backwards in 10-minute intervals, finding closest readings
        downsampled_readings = []
        
        for i in range(432):  # v2 uses 432 inputs (not 433 like v1)
            # Calculate target time (going backwards from reference time)
            target_time = reference_time - timedelta(minutes=i * 10)
            
            # Find closest reading within Â±5 minutes of this target time
            closest_reading = None
            min_diff = float('inf')
            
            for reading in readings_with_timestamps:
                time_diff = abs((reading['timestamp'] - target_time).total_seconds())
                if time_diff <= 300 and time_diff < min_diff:  # Within Â±5 minutes
                    min_diff = time_diff
                    closest_reading = reading
            
            # Use closest reading if found, otherwise -999
            if closest_reading:
                water_level = closest_reading['water_level']
            else:
                water_level = -999
            
            downsampled_readings.append({
                'water_level': water_level,
                'timestamp': target_time
            })
        
        # Reverse to get chronological order (oldest to newest)
        downsampled_readings.reverse()
        
        synthetic_count = sum(1 for r in downsampled_readings if r['water_level'] == -999)
        print(f"Created 432-point sequence with {synthetic_count} missing data points (-999)")
        print(f"Time range: {downsampled_readings[0]['timestamp']} to {downsampled_readings[-1]['timestamp']}")
        
        # Extract water level values, preserving -999 synthetic values
        water_level_values = [r['water_level'] for r in downsampled_readings]
        
        print(f"Prepared {len(water_level_values)} readings for transformer v2 prediction")
        synthetic_count = sum(1 for val in water_level_values if val == -999)
        if synthetic_count > 0:
            print(f"Input includes {synthetic_count} synthetic values (-999) - consistent with training data")
        
        # Log data quality info
        real_values = [v for v in water_level_values if v != -999]
        real_data_percentage = len(real_values) / len(water_level_values) * 100
        
        print(f"Data quality: {real_data_percentage:.1f}% real data")
        if real_values:
            print(f"Water level range: {min(real_values):.1f} - {max(real_values):.1f} mm")
        else:
            print("No real water level data available - using all synthetic values")
        
        # Get the timestamp of the last REAL data point (not synthetic) for accurate forecast timing
        last_real_timestamp = None
        for reading in reversed(downsampled_readings):  # Search backwards
            if reading['water_level'] != -999:
                last_real_timestamp = reading['timestamp'].isoformat()
                break
        
        if not last_real_timestamp:
            # Fallback: use the last reading from original data
            last_real_timestamp = readings_with_timestamps[-1]['timestamp'].isoformat() if readings_with_timestamps else None
            print("Warning: No real data in downsampled sequence, using last original timestamp")
        
        # Generate 24-hour forecast using transformer v2
        forecast_result = inferencer_instance.predict_24_hours(water_level_values, last_real_timestamp)
        
        # Prepare forecast data for storage
        forecast_data = {
            'generated_at': int(datetime.now().timestamp() * 1000),  # Firebase ServerValue.TIMESTAMP equivalent
            'model_version': 'transformer-v2-single-pass',
            'model_architecture': 'single_pass_encoder_transformer_v2',
            'input_data_count': len(water_level_values),
            'input_time_range': {
                'start': readings_with_timestamps[0]['timestamp'].isoformat(),
                'end': readings_with_timestamps[-1]['timestamp'].isoformat()
            },
            'forecast': forecast_result['predictions'],
            'forecast_count': len(forecast_result['predictions']),
            'metadata': forecast_result['metadata'],
            'generation_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        
        # Store current forecast (overwrites previous) in v2-specific path
        forecast_ref = db.reference('/tidal-analysis/transformer-v2-forecast')
        forecast_ref.set(forecast_data)
        
        print(f"Transformer v2 forecast updated successfully")
        print(f"Predictions: {len(forecast_result['predictions'])}")
        print(f"Total time: {(datetime.now() - start_time).total_seconds() * 1000:.1f}ms")
        
    except Exception as e:
        # Get simple error location info
        tb = traceback.extract_tb(e.__traceback__)
        error_location = tb[-1]  # Get the last frame (where error occurred)
        
        # Create simple error summary
        error_summary = f"{os.path.basename(error_location.filename)}:{error_location.lineno} - {type(e).__name__}: {str(e)}"
        
        print(f"Transformer v2 prediction error: {error_summary}")
        
        # Store current error information (overwrites previous) in v2-specific path
        error_data = {
            'generated_at': int(datetime.now().timestamp() * 1000),
            'model_version': 'transformer-v2-single-pass',
            'error_summary': error_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        error_ref = db.reference('/tidal-analysis/transformer-v2-error')
        error_ref.set(error_data)
        
        print("Transformer v2 error information stored")

"""
Firebase Functions Python runtime for Transformer v1 tidal predictions.
Uses single-pass encoder-only transformer for direct 433→144 prediction.
"""
import json
import os
import sys
import traceback
from datetime import datetime, timedelta

from firebase_functions import scheduler_fn
from firebase_admin import initialize_app, db

# Initialize Firebase
initialize_app()

# Lazy imports to avoid PyTorch initialization timeout
torch = None
np = None
create_model = None

def _lazy_imports():
    """Lazy load PyTorch and model creation to avoid initialization timeout"""
    global torch, np, create_model
    if torch is None:
        print("Lazy loading PyTorch and model dependencies...")
        
        # Set PyTorch to CPU-only for faster initialization
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        import torch as _torch
        import numpy as _np
        from model import create_model as _create_model
        
        # Set CPU-only threads for Firebase Functions
        _torch.set_num_threads(2)
        
        torch = _torch
        np = _np
        create_model = _create_model
        print("PyTorch loaded successfully (CPU-only mode)")


class TransformerInferencer:
    """Single-pass transformer inferencer for Firebase Functions"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.normalization_params = None
        self.training_loss = None
        self.input_length = 433   # 72 hours @ 10min intervals
        self.output_length = 144  # 24 hours @ 10min intervals
        
    def initialize(self):
        """Load the trained single-pass transformer model"""
        try:
            # Lazy load PyTorch dependencies
            _lazy_imports()
            # Look for model in current inference directory
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), 
                'best.pth'
            )
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path} (main.py:41)")
            
            print(f"Loading single-pass transformer from {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create model with configuration from checkpoint
            config = checkpoint.get('config', {})
            
            # Handle different config formats (new single-pass vs old autoregressive)
            if 'num_encoder_layers' in config:
                # New single-pass format
                num_layers = config['num_encoder_layers']
            else:
                # Fallback to old format
                num_layers = config.get('num_layers', 6)
            
            self.model = create_model(
                d_model=config.get('d_model', 256),
                nhead=config.get('nhead', 8),
                num_layers=num_layers,
                dropout=config.get('dropout', 0.1)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Store metadata
            self.config = checkpoint['config']
            self.normalization_params = checkpoint['normalization_params']
            self.training_loss = checkpoint.get('loss', checkpoint.get('val_loss', 0.0))
            
            print(f"Single-pass transformer loaded successfully!")
            print(f"Training loss: {self.training_loss:.6f}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Architecture: {self.config.get('architecture', 'single_pass_encoder_transformer')}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize model: {e} (main.py:65)")
            return False
    
    def normalize_data(self, water_levels):
        """Normalize water level data while preserving -999 missing values"""
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        
        # Handle both numpy arrays and torch tensors
        if isinstance(water_levels, torch.Tensor):
            normalized = water_levels.clone()
            valid_mask = (water_levels != -999)
            normalized[valid_mask] = (water_levels[valid_mask] - mean) / std
        else:
            normalized = np.array(water_levels, copy=True)
            valid_mask = (normalized != -999)
            normalized[valid_mask] = (normalized[valid_mask] - mean) / std
            
        return normalized
    
    def denormalize_data(self, normalized_predictions):
        """Denormalize model predictions back to water levels"""
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        return (normalized_predictions * std) + mean
    
    def prepare_input_sequence(self, water_levels):
        """Prepare input sequence for transformer inference"""
        input_sequence = list(water_levels)
        
        # Handle insufficient data by padding with mean value
        if len(input_sequence) < self.input_length:
            print(f"Input has {len(input_sequence)} readings, need {self.input_length}")
            
            mean_value = sum(input_sequence) / len(input_sequence)
            pad_length = self.input_length - len(input_sequence)
            
            print(f"Padding with {pad_length} mean values ({mean_value:.1f})")
            
            # Pad at the beginning with mean values
            padding = [mean_value] * pad_length
            input_sequence = padding + input_sequence
        
        # Handle excess data by taking most recent readings
        if len(input_sequence) > self.input_length:
            print(f"Truncating from {len(input_sequence)} to {self.input_length} readings")
            input_sequence = input_sequence[-self.input_length:]
        
        print(f"Prepared input sequence: {len(input_sequence)} readings")
        print(f"Range: {min(input_sequence):.1f} - {max(input_sequence):.1f} mm")
        
        return input_sequence
    
    def predict_24_hours(self, water_levels, last_data_timestamp=None):
        """Generate 24-hour predictions using single-pass transformer model"""
        if self.model is None:
            raise RuntimeError("Model not initialized (main.py:110)")
        
        try:
            print("Starting single-pass transformer prediction...")
            start_time = datetime.now()
            
            # If exactly 433 values provided, use directly; otherwise prepare sequence
            if len(water_levels) == 433:
                print("Using provided 433-value sequence directly")
                input_sequence = water_levels
            else:
                print(f"Preparing sequence from {len(water_levels)} input values")
                input_sequence = self.prepare_input_sequence(water_levels)
            
            # Handle missing/invalid values
            validated_sequence = []
            for val in input_sequence:
                if val == -999:
                    # Keep -999 values as-is (model was trained with them)
                    validated_sequence.append(-999)
                else:
                    try:
                        # Ensure val is numeric
                        numeric_val = float(val)
                        np_val = np.array(numeric_val)
                        if not np.isfinite(np_val) or np.isnan(np_val):
                            print(f"Invalid input value: {val}, replacing with -999")
                            validated_sequence.append(-999)
                        else:
                            validated_sequence.append(numeric_val)
                    except (ValueError, TypeError):
                        print(f"Non-numeric input value: {val}, replacing with -999")
                        validated_sequence.append(-999)
            
            # Normalize input (preserving -999 values as missing data markers)
            normalized_input = self.normalize_data(torch.FloatTensor(validated_sequence))
            
            # Create input tensor (shape: [1, 433, 1])
            input_tensor = normalized_input.view(1, self.input_length, 1)
            
            print(f"Input tensor shape: {list(input_tensor.shape)}")
            synthetic_count = sum(1 for val in validated_sequence if val == -999)
            print(f"Input contains {synthetic_count} synthetic values (-999)")
            
            # Run single-pass inference
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            print(f"Single-pass inference completed in {inference_time:.1f}ms")
            
            # Get predictions and denormalize
            normalized_predictions = output_tensor.squeeze().cpu().numpy()
            raw_predictions = self.denormalize_data(torch.FloatTensor(normalized_predictions)).tolist()
            
            print(f"Output tensor shape: {list(output_tensor.shape)}")
            print(f"Generated {len(raw_predictions)} predictions")
            
            # Handle invalid predictions
            predictions = []
            for pred in raw_predictions:
                try:
                    numeric_pred = float(pred)
                    np_pred = np.array(numeric_pred)
                    if not np.isfinite(np_pred) or np.isnan(np_pred):
                        print(f"Invalid prediction: {pred}, replacing with -999")
                        predictions.append(-999)
                    else:
                        predictions.append(numeric_pred)
                except (ValueError, TypeError):
                    print(f"Non-numeric prediction: {pred}, replacing with -999")
                    predictions.append(-999)
            
            valid_predictions = [p for p in predictions if p != -999]
            error_count = len(predictions) - len(valid_predictions)
            
            print(f"Valid predictions: {len(valid_predictions)}, errors: {error_count}")
            
            if valid_predictions:
                print(f"Prediction range: {min(valid_predictions):.1f} - {max(valid_predictions):.1f} mm")
            
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
            
            for i, prediction in enumerate(predictions):
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
                    'output_length': len(predictions),
                    'model_architecture': 'single_pass_encoder_transformer',
                    'model_version': 'transformer-v1-single-pass',
                    'normalization': self.normalization_params,
                    'training_loss': self.training_loss,
                    'error_predictions': error_count,
                    'input_synthetic_count': synthetic_count
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise


# Global inferencer instance
inferencer = None

def get_inferencer():
    """Get or create the global inferencer instance"""
    global inferencer
    if inferencer is None:
        inferencer = TransformerInferencer()
        if not inferencer.initialize():
            raise RuntimeError("Failed to initialize transformer inferencer")
    return inferencer


@scheduler_fn.on_schedule(schedule="*/5 * * * *", memory=2048, timeout_sec=540)
def run_transformer_v1_analysis(req):
    """Scheduled function to run transformer predictions every 5 minutes"""
    
    try:
        print("Starting Transformer v1 24-hour prediction run...")
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
        
        # Debug: Check a few sample readings
        sample_keys = list(readings.keys())[:3]
        for sample_key in sample_keys:
            sample_value = readings[sample_key]
            print(f"Sample reading {sample_key}: {sample_value}")
        
        # Parse readings with timestamps and handle -999 values (same as training)
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
        
        # Create 433-point sequence starting from current time, working backwards at 10-minute intervals
        print(f"Creating 433-point sequence from current time backwards...")
        
        # 1) Get the current time
        if readings_with_timestamps:
            # Use timezone from the data
            current_time = datetime.now().replace(tzinfo=readings_with_timestamps[0]['timestamp'].tzinfo)
        else:
            from datetime import timezone
            current_time = datetime.now().replace(tzinfo=timezone.utc)
        
        # 2) Find the closest reading to current time to establish our starting point
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
        
        # 3 & 4) Count backwards in 10-minute intervals, finding closest readings
        downsampled_readings = []
        
        for i in range(433):
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
                'timestamp': target_time
            })
        
        # Reverse to get chronological order (oldest to newest)
        downsampled_readings.reverse()
        
        synthetic_count = sum(1 for r in downsampled_readings if r['water_level'] == -999)
        print(f"Created 433-point sequence with {synthetic_count} missing data points (-999)")
        print(f"Time range: {downsampled_readings[0]['timestamp']} to {downsampled_readings[-1]['timestamp']}")
        
        # Log timing information for diagnostics
        time_gaps = []
        for i in range(1, len(downsampled_readings)):
            time_diff = (downsampled_readings[i]['timestamp'] - downsampled_readings[i-1]['timestamp']).total_seconds() / 60
            time_gaps.append(time_diff)
        
        avg_interval = sum(time_gaps) / len(time_gaps) if time_gaps else 0
        max_gap = max(time_gaps) if time_gaps else 0
        
        # Extract water level values, preserving -999 synthetic values
        water_level_values = [r['water_level'] for r in downsampled_readings]
        
        print(f"Prepared {len(water_level_values)} readings for prediction")
        synthetic_count = sum(1 for val in water_level_values if val == -999)
        if synthetic_count > 0:
            print(f"Input includes {synthetic_count} synthetic values (-999) - consistent with training data")
        
        # Log data quality info (model can handle synthetic data)
        real_values = [v for v in water_level_values if v != -999]
        real_data_percentage = len(real_values) / len(water_level_values) * 100
        
        print(f"Data quality: {real_data_percentage:.1f}% real data, avg interval: {avg_interval:.1f}min, max gap: {max_gap:.1f}min")
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
        
        # Generate 24-hour forecast using single-pass transformer
        forecast_result = inferencer_instance.predict_24_hours(water_level_values, last_real_timestamp)
        
        # Prepare forecast data for storage
        forecast_data = {
            'generated_at': int(datetime.now().timestamp() * 1000),  # Firebase ServerValue.TIMESTAMP equivalent
            'model_version': 'transformer-v1-single-pass',
            'model_architecture': 'single_pass_encoder_transformer',
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
        
        # Store current forecast (overwrites previous)
        forecast_ref = db.reference('/tidal-analysis/transformer-v1-forecast')
        forecast_ref.set(forecast_data)
        
        print(f"Transformer v1 forecast updated successfully")
        print(f"Predictions: {len(forecast_result['predictions'])}")
        print(f"Total time: {(datetime.now() - start_time).total_seconds() * 1000:.1f}ms")
        
    except Exception as e:
        # Get simple error location info
        tb = traceback.extract_tb(e.__traceback__)
        error_location = tb[-1]  # Get the last frame (where error occurred)
        
        # Create simple error summary
        error_summary = f"{os.path.basename(error_location.filename)}:{error_location.lineno} - {type(e).__name__}: {str(e)}"
        
        print(f"Transformer v1 prediction error: {error_summary}")
        
        # Store current error information (overwrites previous)
        error_data = {
            'generated_at': int(datetime.now().timestamp() * 1000),
            'model_version': 'transformer-v1-single-pass',
            'error_summary': error_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        error_ref = db.reference('/tidal-analysis/transformer-v1-error')
        error_ref.set(error_data)
        
        print("Error information stored")
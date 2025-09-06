"""
Firebase Functions Python runtime for Transformer v1 tidal predictions.
Uses raw PyTorch model for zero conversion overhead and perfect quality.
"""
import json
import os
import sys
from datetime import datetime, timedelta
import torch
import numpy as np

from model import create_model
from firebase_functions import scheduler_fn
from firebase_admin import initialize_app, db

# Initialize Firebase
initialize_app()


class TransformerInferencer:
    """PyTorch-based transformer inferencer for Firebase Functions"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.normalization_params = None
        self.training_loss = None
        self.input_length = 433   # 72 hours @ 10min intervals
        self.output_length = 144  # 24 hours @ 10min intervals
        
    def initialize(self):
        """Load the trained PyTorch model"""
        try:
            # Look for model in current inference directory
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), 
                'best.pth'
            )
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path} (main.py:41)")
            
            print(f"Loading PyTorch model from {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create model and load weights
            self.model = create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Store metadata
            self.config = checkpoint['config']
            self.normalization_params = checkpoint['normalization_params']
            self.training_loss = checkpoint['loss']
            
            print(f"Model loaded successfully!")
            print(f"Training loss: {self.training_loss:.6f}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize model: {e} (main.py:65)")
            return False
    
    def normalize_data(self, water_levels):
        """Normalize water level data using training statistics"""
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        return (water_levels - mean) / std
    
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
        """Generate 24-hour predictions using the transformer model"""
        if self.model is None:
            raise RuntimeError("Model not initialized (main.py:110)")
        
        try:
            print("Starting transformer prediction...")
            start_time = datetime.now()
            
            # Prepare input sequence
            input_sequence = self.prepare_input_sequence(water_levels)
            
            # Handle missing/invalid values
            validated_sequence = []
            for val in input_sequence:
                try:
                    # Ensure val is numeric before using numpy functions
                    numeric_val = float(val)
                    # Convert to numpy array to safely use numpy functions
                    np_val = np.array(numeric_val)
                    if not np.isfinite(np_val) or np.isnan(np_val):
                        print(f"Invalid input value: {val}, replacing with -999 (main.py:128)")
                        validated_sequence.append(-999)
                    else:
                        validated_sequence.append(numeric_val)
                except (ValueError, TypeError):
                    print(f"Non-numeric input value: {val}, replacing with -999 (main.py:133)")
                    validated_sequence.append(-999)
            
            # Normalize input
            normalized_input = self.normalize_data(torch.FloatTensor(validated_sequence))
            
            # Create input tensor (shape: [1, 433, 1])
            input_tensor = normalized_input.view(1, self.input_length, 1)
            
            print(f"Input tensor shape: {list(input_tensor.shape)}")
            
            # Run inference
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            print(f"Inference completed in {inference_time:.1f}ms")
            
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
                    # Convert to numpy array to safely use numpy functions
                    np_pred = np.array(numeric_pred)
                    if not np.isfinite(np_pred) or np.isnan(np_pred):
                        print(f"Invalid prediction: {pred}, replacing with -999 (main.py:166)")
                        predictions.append(-999)
                    else:
                        predictions.append(numeric_pred)
                except (ValueError, TypeError):
                    print(f"Non-numeric prediction: {pred}, replacing with -999 (main.py:171)")
                    predictions.append(-999)
            
            valid_predictions = [p for p in predictions if p != -999]
            error_count = len(predictions) - len(valid_predictions)
            
            print(f"Valid predictions: {len(valid_predictions)}, errors: {error_count}")
            
            if valid_predictions:
                print(f"Range: {min(valid_predictions):.1f} - {max(valid_predictions):.1f} mm")
            
            # Create timestamped predictions
            if last_data_timestamp:
                # Parse the last data timestamp and use it as starting point
                try:
                    base_time = datetime.fromisoformat(last_data_timestamp.replace('Z', '+00:00'))
                except ValueError:
                    # Fallback to current time if timestamp parsing fails
                    base_time = datetime.now()
                    print(f"Failed to parse timestamp {last_data_timestamp}, using current time (main.py:185)")
            else:
                base_time = datetime.now()
                print("No last data timestamp provided, using current time (main.py:188)")
            
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
                    'model_architecture': 'seq2seq_transformer_pytorch',
                    'model_version': 'transformer-v1-pytorch',
                    'normalization': self.normalization_params,
                    'training_loss': self.training_loss,
                    'error_predictions': error_count
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e} (main.py:211)")
            raise


# Global inferencer instance
inferencer = None

def get_inferencer():
    """Get or create the global inferencer instance"""
    global inferencer
    if inferencer is None:
        inferencer = TransformerInferencer()
        if not inferencer.initialize():
            raise RuntimeError("Failed to initialize transformer inferencer (main.py:224)")
    return inferencer


@scheduler_fn.on_schedule(schedule="*/5 * * * *", memory=1024)
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
        
        if len(readings_with_timestamps) < 4320:  # Need 72 hours minimum
            print(f"Insufficient data points: {len(readings_with_timestamps)} (need at least 4320 for 72 hours)")
            return
        
        # Sort by timestamp to ensure chronological order
        readings_with_timestamps.sort(key=lambda x: x['timestamp'])
        
        print(f"Processing {len(readings_with_timestamps)} chronologically sorted readings")
        print(f"Time range: {readings_with_timestamps[0]['timestamp']} to {readings_with_timestamps[-1]['timestamp']}")
        
        # Take the most recent 4320 readings (72 hours)
        recent_readings = readings_with_timestamps[-4320:]
        
        # Apply training-style downsampling: every 10th reading + last reading
        print("Applying training-style downsampling to 10-minute intervals...")
        downsampled_readings = []
        
        # Every 10th reading
        for i in range(0, len(recent_readings), 10):
            downsampled_readings.append(recent_readings[i])
        
        # Always add the very last reading
        if len(recent_readings) > 0 and recent_readings[-1] not in downsampled_readings:
            downsampled_readings.append(recent_readings[-1])
        
        # Take exactly 433 readings (consistent with training)
        if len(downsampled_readings) > 433:
            downsampled_readings = downsampled_readings[-433:]
        
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
        
        # Check if we have sufficient real data for reliable prediction
        real_values = [v for v in water_level_values if v != -999]
        real_data_percentage = len(real_values) / len(water_level_values) * 100
        
        if len(real_values) < 200:  # Need at least 200 real readings (~33 hours at 10min intervals)
            print(f"Insufficient real data: {len(real_values)} readings ({real_data_percentage:.1f}% real data)")
            print(f"Need at least 200 real readings for reliable prediction")
            return
        
        print(f"Data quality: {real_data_percentage:.1f}% real data, avg interval: {avg_interval:.1f}min, max gap: {max_gap:.1f}min")
        print(f"Water level range: {min(real_values):.1f} - {max(real_values):.1f} mm")
        
        # Get the timestamp of the last data point for accurate forecast timing
        last_data_timestamp = downsampled_readings[-1]['timestamp'] if downsampled_readings else None
        
        # Generate 24-hour forecast
        forecast_result = inferencer_instance.predict_24_hours(water_level_values, last_data_timestamp)
        
        # Prepare forecast data for storage
        forecast_data = {
            'generated_at': int(datetime.now().timestamp() * 1000),  # Firebase ServerValue.TIMESTAMP equivalent
            'model_version': 'transformer-v1-pytorch',
            'model_architecture': 'seq2seq_transformer_pytorch',
            'input_data_count': len(water_level_values),
            'input_time_range': {
                'start': readings_with_timestamps[0]['timestamp'],
                'end': readings_with_timestamps[-1]['timestamp']
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
        print(f"Transformer v1 prediction error: {e} (main.py:350)")
        
        # Store current error information (overwrites previous)
        error_data = {
            'generated_at': int(datetime.now().timestamp() * 1000),
            'model_version': 'transformer-v1-pytorch',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        error_ref = db.reference('/tidal-analysis/transformer-v1-error')
        error_ref.set(error_data)
        
        print("Error information stored (main.py:363)")



#!/usr/bin/env python3
"""
Simple web server for transformer tidal prediction model.
Uses raw PyTorch model for zero conversion overhead and perfect quality.
"""
import torch
import json
import os
import sys
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import numpy as np

# Import model from tidal-analysis root directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from model import create_model


class TidalPredictor:
    """
    Wrapper for loading and using the trained transformer model.
    """
    
    def __init__(self, checkpoint_path='checkpoints/best.pth'):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self.normalization_params = None
        self.device = torch.device('cpu')  # Use CPU for deployment
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model from checkpoint"""
        print(f"Loading model from {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Create model and load weights
        self.model = create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store metadata
        self.config = checkpoint['config']
        self.normalization_params = checkpoint['normalization_params']
        self.training_loss = checkpoint['loss']
        
        print(f"Model loaded successfully!")
        print(f"Training loss: {self.training_loss:.6f}")
        print(f"Normalization - mean: {self.normalization_params['mean']:.4f}, "
              f"std: {self.normalization_params['std']:.4f}")
    
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
    
    def predict(self, water_levels_72h):
        """
        Make 24-hour water level predictions
        
        Args:
            water_levels_72h: List or array of 433 water level values (72h at 10-min intervals)
            
        Returns:
            predictions_24h: List of 144 predicted water levels (24h at 10-min intervals)
        """
        # Validate input
        if len(water_levels_72h) != 433:
            raise ValueError(f"Expected 433 input values (72h at 10-min intervals), got {len(water_levels_72h)}")
        
        # Convert to tensor and normalize
        input_tensor = torch.FloatTensor(water_levels_72h).view(1, 433, 1).to(self.device)
        input_normalized = self.normalize_data(input_tensor)
        
        # Model inference
        with torch.no_grad():
            output_normalized = self.model(input_normalized)
        
        # Denormalize and convert back to list
        predictions = self.denormalize_data(output_normalized)
        return predictions.squeeze().cpu().tolist()
    
    def get_model_info(self):
        """Get model information for API responses"""
        model_info = self.model.get_model_info()
        return {
            'model': model_info,
            'training': {
                'final_loss': float(self.training_loss),
                'configuration': self.config
            },
            'normalization': {
                'mean': float(self.normalization_params['mean']),
                'std': float(self.normalization_params['std'])
            },
            'input_specification': {
                'length': 433,
                'description': '72 hours of water level data at 10-minute intervals',
                'units': 'millimeters'
            },
            'output_specification': {
                'length': 144,
                'description': '24 hours of water level predictions at 10-minute intervals',
                'units': 'millimeters'
            }
        }


# Initialize the predictor
predictor = None

def init_predictor():
    """Initialize the global predictor instance"""
    global predictor
    if predictor is None:
        try:
            predictor = TidalPredictor()
        except Exception as e:
            print(f"Failed to initialize predictor: {e}")
            raise

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Transformer Tidal Prediction API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/info', methods=['GET'])
def get_info():
    """Get model information"""
    try:
        init_predictor()
        return jsonify(predictor.get_model_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make tidal predictions
    
    Expected JSON payload:
    {
        "water_levels": [433 values], // 72 hours at 10-minute intervals
        "start_time": "2025-01-01T00:00:00Z" // Optional, for timestamp generation
    }
    
    Returns:
    {
        "predictions": [144 values], // 24 hours at 10-minute intervals
        "timestamps": ["2025-01-01T00:00:00Z", ...], // Optional, if start_time provided
        "metadata": { ... }
    }
    """
    try:
        init_predictor()
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        water_levels = data.get('water_levels')
        if not water_levels:
            return jsonify({'error': 'water_levels field is required'}), 400
        
        start_time_str = data.get('start_time')
        
        # Make prediction
        predictions = predictor.predict(water_levels)
        
        # Generate timestamps if requested
        response = {
            'predictions': predictions,
            'metadata': {
                'input_length': len(water_levels),
                'output_length': len(predictions),
                'prediction_horizon_hours': 24,
                'interval_minutes': 10,
                'model_version': 'transformer_v1'
            }
        }
        
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                timestamps = []
                for i in range(144):  # 24 hours at 10-minute intervals
                    timestamp = start_time + timedelta(minutes=i * 10)
                    timestamps.append(timestamp.isoformat())
                response['timestamps'] = timestamps
            except ValueError as e:
                response['timestamp_error'] = f"Invalid start_time format: {e}"
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Validation error: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make multiple predictions in batch
    
    Expected JSON payload:
    {
        "requests": [
            {"water_levels": [433 values], "id": "optional_id"},
            {"water_levels": [433 values], "id": "optional_id"}
        ]
    }
    
    Returns:
    {
        "predictions": [
            {"predictions": [144 values], "id": "optional_id"},
            {"predictions": [144 values], "id": "optional_id"}
        ],
        "metadata": { ... }
    }
    """
    try:
        init_predictor()
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        requests = data.get('requests', [])
        if not requests:
            return jsonify({'error': 'requests field is required'}), 400
        
        # Process each request
        predictions = []
        for i, req in enumerate(requests):
            water_levels = req.get('water_levels')
            if not water_levels:
                predictions.append({
                    'error': 'water_levels field is required',
                    'id': req.get('id', i)
                })
                continue
            
            try:
                pred = predictor.predict(water_levels)
                predictions.append({
                    'predictions': pred,
                    'id': req.get('id', i)
                })
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'id': req.get('id', i)
                })
        
        return jsonify({
            'predictions': predictions,
            'metadata': {
                'batch_size': len(requests),
                'successful_predictions': len([p for p in predictions if 'error' not in p]),
                'model_version': 'transformer_v1'
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {e}'}), 500


def main():
    """Start the web server"""
    print("Transformer Tidal Prediction Server")
    print("=" * 40)
    
    try:
        # Initialize predictor on startup
        init_predictor()
        
        # Start server
        print("\nStarting server...")
        print("Endpoints:")
        print("  GET  /       - Health check")
        print("  GET  /info   - Model information")  
        print("  POST /predict - Single prediction")
        print("  POST /predict/batch - Batch predictions")
        print()
        print("Server running at http://localhost:8000")
        
        app.run(host='0.0.0.0', port=8000, debug=False)
        
    except Exception as e:
        print(f"Failed to start server: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
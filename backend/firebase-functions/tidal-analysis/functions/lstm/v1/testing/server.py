#!/usr/bin/env python3
"""
Simple HTTP server for LSTM model testing
Loads the actual trained model and serves predictions via REST API
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import torch
import numpy as np
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime, timedelta
from firebase_fetch import get_latest_72_readings

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'training'))

try:
    from model import TidalLSTM
    print("Model imported successfully")
except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

class ModelHandler(BaseHTTPRequestHandler):
    model = None
    norm_params = None
    model_info = None
    
    @classmethod
    def load_model(cls):
        """Load the trained model once on startup"""
        if cls.model is not None:
            return True
            
        model_path = '../training/trained_models/best_model.pth'
        norm_path = '../data-preparation/data/normalization_params.json'
        
        try:
            # Load model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
                
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint['config']
            
            cls.model = TidalLSTM(
                input_size=1,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
            
            cls.model.load_state_dict(checkpoint['model_state_dict'])
            cls.model.eval()
            
            cls.model_info = f"Hidden: {config['hidden_size']}, Layers: {config['num_layers']}, Val Loss: {checkpoint['val_loss']:.6f}"
            
            print(f"Model loaded: {cls.model_info}")
            
            # Load normalization params
            if os.path.exists(norm_path):
                with open(norm_path, 'r') as f:
                    cls.norm_params = json.load(f)
                print(f"Normalization loaded: mean={cls.norm_params['mean']:.1f}, std={cls.norm_params['std']:.1f}")
            else:
                cls.norm_params = {'mean': 1850.0, 'std': 300.0}
                print("Using default normalization parameters")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def do_GET(self):
        """Serve the HTML file"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Ensure we're reading from the correct directory
            html_path = os.path.join(os.path.dirname(__file__), 'index.html')
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
                
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                'model_loaded': self.model is not None,
                'model_info': self.model_info
            }
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/fetch-firebase':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                water_levels, timestamps, full_data = get_latest_72_readings()
                if water_levels is None:
                    raise Exception("Failed to fetch Firebase data")
                
                response = {
                    'water_levels': water_levels,
                    'timestamps': [t.isoformat() for t in timestamps],
                    'count': len(water_levels),
                    'range': [min(water_levels), max(water_levels)]
                }
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                error_response = {'error': str(e)}
                self.wfile.write(json.dumps(error_response).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle model prediction requests"""
        if self.path == '/test-model':
            try:
                # Parse request
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                readings = data.get('readings', [])
                
                # Accept any length - model handles variable input with -1 padding
                if not readings:
                    raise ValueError("No readings provided")
                
                if not self.model:
                    raise RuntimeError("Model not loaded")
                
                # Make prediction
                prediction = self.predict(readings)
                
                response = {
                    'prediction': prediction,
                    'model_info': self.model_info,
                    'input_count': len(readings)
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                error_response = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
        
        elif self.path == '/predict-24h':
            try:
                # Parse request
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                readings = data.get('readings', [])
                
                # Accept any length - model handles variable input with -1 padding
                if not readings:
                    raise ValueError("No readings provided")
                
                if not self.model:
                    raise RuntimeError("Model not loaded")
                
                # Generate 24-hour prediction (1440 minutes)
                predictions = self.predict_24_hours(readings)
                
                response = {
                    'predictions': predictions,
                    'prediction_count': len(predictions),
                    'model_info': self.model_info,
                    'input_count': len(readings)
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                error_response = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
                
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def predict(self, readings):
        """Make prediction using the loaded model"""
        # Normalize input
        normalized_input = self.normalize_data(np.array(readings))
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(normalized_input).unsqueeze(0).unsqueeze(-1)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Denormalize and return
        return self.denormalize_data(prediction.item())
    
    def normalize_data(self, data):
        """Normalize using training parameters"""
        return (data - self.norm_params['mean']) / self.norm_params['std']
    
    def denormalize_data(self, data):
        """Denormalize using training parameters"""
        return data * self.norm_params['std'] + self.norm_params['mean']
    
    def predict_24_hours(self, initial_readings):
        """
        Generate 24-hour forecast using iterative prediction
        Takes 72 readings, predicts 1440 future readings (24 hours)
        """
        print("Starting 24-hour prediction...")
        
        # Start with the initial sequence (most recent 72 readings)
        current_sequence = list(initial_readings)
        predictions = []
        
        # Predict 1440 steps (24 hours at 1-minute intervals)
        for step in range(1440):
            # Use the most recent 72 readings for prediction
            input_sequence = current_sequence[-72:]
            
            # Make single-step prediction
            next_prediction = self.predict(input_sequence)
            
            # Add prediction to our sequence and results
            current_sequence.append(next_prediction)
            predictions.append(next_prediction)
            
            # Progress indicator every 120 steps (2 hours)
            if (step + 1) % 120 == 0:
                hours = (step + 1) / 60
                print(f"  {hours:.1f} hours predicted ({step + 1}/1440 steps)")
        
        print(f"Completed 24-hour prediction: {len(predictions)} data points")
        return predictions
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def main():
    """Start the server"""
    print("LSTM v1 Model Testing Server")
    print("=" * 35)
    
    # Load model on startup
    if not ModelHandler.load_model():
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Start server
    port = 8000
    server = HTTPServer(('localhost', port), ModelHandler)
    
    print(f"Server running at: http://localhost:{port}")
    print(f"Open browser to test the model")
    print(f"Press Ctrl+C to stop")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()

if __name__ == "__main__":
    main()
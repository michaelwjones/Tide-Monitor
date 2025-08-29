#!/usr/bin/env python3
"""
HTTP server for Transformer model testing
Loads the trained seq2seq transformer and serves predictions via REST API
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import torch
import numpy as np
import sys
import os
from pathlib import Path
import traceback
import time
from datetime import datetime, timedelta
from firebase_fetch import get_transformer_input_sequence, create_sample_data

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'training'))

try:
    from model import TidalTransformer, create_model
    print("✅ Transformer model imported successfully")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    sys.exit(1)

class TransformerHandler(BaseHTTPRequestHandler):
    model = None
    norm_params = None
    model_info = None
    config = None
    
    @classmethod
    def load_model(cls):
        """Load the trained transformer model once on startup"""
        if cls.model is not None:
            return True
            
        model_path = '../training/checkpoints/best.pth'
        
        try:
            # Load model
            if not os.path.exists(model_path):
                print(f"⚠️  Model not found at {model_path}")
                print("📝 Creating dummy model for testing interface...")
                
                # Create dummy model for interface testing
                cls.model = create_model()
                cls.config = {'dummy': True}
                cls.norm_params = {'mean': 2000.0, 'std': 500.0}
                cls.model_info = cls.model.get_model_info()
                cls.model_info['status'] = 'dummy_model'
                return True
                
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create and load model
            cls.model = create_model()
            cls.model.load_state_dict(checkpoint['model_state_dict'])
            cls.model.eval()
            
            cls.config = checkpoint['config']
            cls.norm_params = checkpoint['normalization_params']
            cls.model_info = cls.model.get_model_info()
            cls.model_info['status'] = 'trained_model'
            cls.model_info['validation_loss'] = float(checkpoint['loss'])
            cls.model_info['training_epoch'] = checkpoint['epoch']
            
            print("✅ Transformer model loaded successfully")
            print(f"   Validation loss: {checkpoint['loss']:.6f}")
            print(f"   Parameters: {cls.model_info['total_parameters']:,}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_html()
        elif self.path == '/model-info':
            self.serve_model_info()
        elif self.path == '/fetch-firebase':
            self.fetch_firebase_data()
        elif self.path == '/generate-sample':
            self.generate_sample_data()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/predict':
            self.handle_prediction()
        else:
            self.send_error(404)
    
    def serve_html(self):
        """Serve the testing interface HTML"""
        html_file = Path(__file__).parent / 'index.html'
        
        if not html_file.exists():
            self.send_error(404, "index.html not found")
            return
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        with open(html_file, 'r') as f:
            self.wfile.write(f.read().encode())
    
    def serve_model_info(self):
        """Serve model information"""
        if not self.load_model():
            self.send_json_response({'error': 'Model not loaded'}, 500)
            return
        
        self.send_json_response({
            'model_info': self.model_info,
            'normalization': self.norm_params,
            'config': self.config
        })
    
    def fetch_firebase_data(self):
        """Fetch real data from Firebase"""
        try:
            print("📡 Fetching Firebase data...")
            water_levels, timestamps = get_transformer_input_sequence()
            
            if not water_levels:
                self.send_json_response({
                    'error': 'Failed to fetch Firebase data',
                    'success': False
                }, 500)
                return
            
            # Format timestamps for JSON
            timestamp_strings = [t.isoformat() for t in timestamps] if timestamps else None
            
            self.send_json_response({
                'water_levels': water_levels,
                'timestamps': timestamp_strings,
                'count': len(water_levels),
                'success': True,
                'stats': {
                    'min': float(min(water_levels)),
                    'max': float(max(water_levels)),
                    'mean': float(np.mean(water_levels)),
                    'std': float(np.std(water_levels))
                }
            })
            
        except Exception as e:
            print(f"❌ Firebase fetch error: {e}")
            self.send_json_response({
                'error': str(e),
                'success': False
            }, 500)
    
    def generate_sample_data(self):
        """Generate sample tidal data"""
        try:
            print("🧪 Generating sample data...")
            water_levels, timestamps = create_sample_data()
            
            # Format timestamps for JSON
            timestamp_strings = [t.isoformat() for t in timestamps]
            
            self.send_json_response({
                'water_levels': water_levels,
                'timestamps': timestamp_strings,
                'count': len(water_levels),
                'success': True,
                'type': 'sample',
                'stats': {
                    'min': float(min(water_levels)),
                    'max': float(max(water_levels)),
                    'mean': float(np.mean(water_levels)),
                    'std': float(np.std(water_levels))
                }
            })
            
        except Exception as e:
            print(f"❌ Sample generation error: {e}")
            self.send_json_response({
                'error': str(e),
                'success': False
            }, 500)
    
    def handle_prediction(self):
        """Handle prediction requests"""
        try:
            if not self.load_model():
                self.send_json_response({'error': 'Model not loaded'}, 500)
                return
            
            # Read request data
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode())
            
            input_data = request_data['input_data']
            
            if len(input_data) != 4320:
                self.send_json_response({
                    'error': f'Invalid input length: {len(input_data)} (expected 4320)',
                    'success': False
                }, 400)
                return
            
            print(f"🔮 Making prediction with {len(input_data)} input points...")
            
            # Make prediction
            predictions = self.predict_sequence(input_data)
            
            if predictions is None:
                self.send_json_response({
                    'error': 'Prediction failed',
                    'success': False
                }, 500)
                return
            
            print(f"✅ Generated {len(predictions)} predictions")
            
            self.send_json_response({
                'predictions': predictions,
                'input_length': len(input_data),
                'output_length': len(predictions),
                'success': True,
                'stats': {
                    'min': float(min(predictions)),
                    'max': float(max(predictions)),
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions))
                }
            })
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            self.send_json_response({
                'error': str(e),
                'success': False
            }, 500)
    
    def predict_sequence(self, input_data):
        """Make seq2seq prediction using transformer"""
        try:
            # Handle dummy model case
            if self.model_info.get('status') == 'dummy_model':
                print("⚠️  Using dummy model - generating synthetic predictions")
                # Generate realistic tidal pattern for next 24 hours
                t = np.linspace(0, 24, 1440)
                base_level = np.mean(input_data)
                
                # Simple tidal simulation
                M2_tide = 400 * np.sin(2 * np.pi * t / 12.42)  # Semi-diurnal
                K1_tide = 150 * np.sin(2 * np.pi * t / 24.0)   # Diurnal
                noise = np.random.normal(0, 30, 1440)
                
                predictions = base_level + M2_tide + K1_tide + noise
                predictions = np.clip(predictions, 300, 5000)
                return predictions.tolist()
            
            # Normalize input data
            input_array = np.array(input_data, dtype=np.float32)
            mean = self.norm_params['mean']
            std = self.norm_params['std']
            normalized_input = (input_array - mean) / std
            
            # Convert to tensor format: (batch_size, seq_len, input_dim)
            input_tensor = torch.from_numpy(normalized_input).unsqueeze(0).unsqueeze(-1)  # (1, 4320, 1)
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                output_tensor = self.model(input_tensor)  # (1, 1440, 1)
            inference_time = time.time() - start_time
            
            print(f"⚡ Inference completed in {inference_time*1000:.1f} ms")
            
            # Denormalize output
            normalized_predictions = output_tensor.squeeze().numpy()  # (1440,)
            predictions = normalized_predictions * std + mean
            
            return predictions.tolist()
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            return None
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        json_data = json.dumps(data, indent=2)
        self.wfile.write(json_data.encode())
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default request logging"""
        pass

def main():
    port = 8000
    
    print("🚀 Transformer v1 Testing Server")
    print("=" * 40)
    
    # Load model on startup
    if not TransformerHandler.load_model():
        print("❌ Failed to load model, exiting...")
        return 1
    
    try:
        server = HTTPServer(('localhost', port), TransformerHandler)
        print(f"🌐 Server running at http://localhost:{port}")
        print(f"🧠 Model status: {TransformerHandler.model_info.get('status', 'unknown')}")
        print("📱 Open your browser to test the model!")
        print("Press Ctrl+C to stop server")
        print("-" * 40)
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
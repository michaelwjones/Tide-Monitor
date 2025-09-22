#!/usr/bin/env python3

import http.server
import socketserver
import os
import json
import numpy as np
from pathlib import Path
import urllib.parse

# Try to import inference engine
try:
    from inference import load_inference_engine
    INFERENCE_AVAILABLE = True
    print("Inference engine available")
except Exception as e:
    INFERENCE_AVAILABLE = False
    print(f"Inference engine not available: {e}")

# Global inference engine
inference_engine = None

class DataAnalysisHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Change to the testing directory to serve files
        os.chdir(Path(__file__).parent)
        super().__init__(*args, **kwargs)

    def end_headers(self):
        # Add CORS headers to allow local file access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        print(f"GET request: {self.path}")  # Debug output
        
        if self.path.startswith('/inference/'):
            print("Routing to inference handler")  # Debug output
            # Handle inference requests
            self.handle_inference_request()
            return
        elif self.path.startswith('/sequence_names/'):
            # Serve sequence names files
            data_path = Path('../data-preparation/data').absolute()
            file_name = self.path[16:]  # Remove '/sequence_names/' prefix
            file_path = data_path / file_name
            
            if file_path.exists() and file_path.is_file() and file_path.suffix == '.json':
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(content.encode())
                    return
                except Exception as e:
                    print(f"Error reading sequence names file {file_path}: {e}")
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(f"Error reading sequence names file: {e}".encode())
                    return
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Sequence names file not found')
                return
        elif self.path.startswith('/data/'):
            # Serve data files from the data-preparation/data directory
            data_path = Path('../data-preparation/data').absolute()
            file_name = self.path[6:]  # Remove '/data/' prefix
            file_path = data_path / file_name
            
            if file_path.exists() and file_path.is_file():
                if file_path.suffix == '.npy':
                    # Convert NumPy array to JSON for web consumption
                    try:
                        array = np.load(file_path)
                        
                        # Send all sequences for complete testing analysis
                        
                        # Convert to Python lists for JSON serialization
                        data = {
                            'shape': list(array.shape),
                            'dtype': str(array.dtype),
                            'data': array.tolist(),
                            'stats': {
                                'min': float(array.min()),
                                'max': float(array.max()),
                                'mean': float(array.mean()),
                                'std': float(array.std())
                            }
                        }
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(data).encode())
                        return
                    except Exception as e:
                        print(f"Error loading NumPy file {file_path}: {e}")
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(f"Error loading NumPy file: {e}".encode())
                        return
                else:
                    # Serve other data files (JSON, etc.)
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        self.send_response(200)
                        if file_path.suffix == '.json':
                            self.send_header('Content-type', 'application/json')
                        else:
                            self.send_header('Content-type', 'application/octet-stream')
                        self.end_headers()
                        self.wfile.write(content)
                        return
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(f"Error reading file: {e}".encode())
                        return
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Data file not found')
                return
        
        # Default handling for other requests
        super().do_GET()
    
    def handle_inference_request(self):
        """Handle inference API requests."""
        global inference_engine
        
        print(f"Inference request: {self.path}")  # Debug output
        
        if not INFERENCE_AVAILABLE:
            print("Inference not available")
            self.send_response(503)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': 'Inference engine not available',
                'message': 'Please ensure the model is trained and available'
            }).encode())
            return
        
        # Parse the request path - handle query parameters
        path_without_query = self.path.split('?')[0]  # Remove query parameters
        path_parts = path_without_query.split('/')
        print(f"Path parts: {path_parts}")  # Debug output
        
        if len(path_parts) < 3:
            print("Invalid path length")
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Invalid inference path'}).encode())
            return
        
        action = path_parts[2]  # /inference/[action]
        print(f"Action: {action}")  # Debug output
        
        try:
            # Initialize inference engine if needed
            if inference_engine is None:
                print("Loading inference engine...")
                inference_engine = load_inference_engine()
                print("Inference engine loaded successfully")
            
            if action == 'status':
                # Return inference engine status
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    'status': 'ready',
                    'device': str(inference_engine.device),
                    'model_loaded': inference_engine.model is not None,
                    'normalization_loaded': inference_engine.norm_params is not None
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif action == 'predict_sequence':
                # Predict from a training sequence using timestamp name
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                
                # Support both old index-based and new name-based access
                sequence_index = None
                if 'name' in params:
                    sequence_name = params['name'][0]
                    # Load sequence names to find index
                    data_dir = Path('../data-preparation/data')
                    try:
                        with open(data_dir / 'sequence_names_train.json', 'r') as f:
                            sequence_names = json.load(f)
                        
                        if sequence_name in sequence_names:
                            sequence_index = sequence_names.index(sequence_name)
                        else:
                            self.send_response(400)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({'error': f'Sequence name {sequence_name} not found'}).encode())
                            return
                    except FileNotFoundError:
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'Sequence names file not found'}).encode())
                        return
                elif 'index' in params:
                    # Backward compatibility for index-based access
                    try:
                        sequence_index = int(params['index'][0])
                    except ValueError:
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'Invalid sequence index'}).encode())
                        return
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Missing sequence name or index parameter'}).encode())
                    return
                
                # Load training data
                data_dir = Path('../data-preparation/data')
                X_train = np.load(data_dir / 'X_train.npy')
                y_train = np.load(data_dir / 'y_train.npy')
                
                print(f"Training data loaded: {len(X_train)} sequences")
                print(f"Requested index: {sequence_index}")
                
                if sequence_index >= len(X_train):
                    print(f"Index out of range: {sequence_index} >= {len(X_train)}")
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'error': 'Sequence index out of range',
                        'requested_index': sequence_index,
                        'max_index': len(X_train) - 1,
                        'total_sequences': len(X_train)
                    }).encode())
                    return
                
                # Make prediction
                sequence_data = {
                    'input': X_train[sequence_index],
                    'target': y_train[sequence_index]
                }
                
                result = inference_engine.predict_from_sequence(sequence_data)
                result['sequence_index'] = sequence_index
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': f'Unknown inference action: {action}'}).encode())
        
        except Exception as e:
            print(f"Inference error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': 'Inference failed',
                'message': str(e)
            }).encode())

def main():
    PORT = 8000
    
    # Check if data directory exists
    data_dir = Path('../data-preparation/data').absolute()
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        print("Please run the data preparation scripts first:")
        print("  cd ../data-preparation")
        print("  python fetch_firebase_data.py")
        print("  python create_training_data.py")
        return
    
    # List available data files
    data_files = list(data_dir.glob('*'))
    print(f"Found {len(data_files)} data files:")
    for f in data_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")
    
    print(f"\nStarting data analysis server on port {PORT}")
    print(f"Open your browser to: http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    
    try:
        with socketserver.TCPServer(("", PORT), DataAnalysisHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Transformer v2 Inference Module
Loads trained model and makes predictions for testing interface.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import math
from pathlib import Path
import sys
import os

# Try to import model from multiple locations
try:
    # First try local copy
    from model import TidalTransformerV2
    print("Loaded model from local copy")
except ImportError:
    try:
        # Try from training directory
        sys.path.append(str(Path(__file__).parent.parent / "training"))
        from model import TidalTransformerV2
        print("Loaded model from training directory")
    except ImportError:
        # Try current directory
        sys.path.append(str(Path(__file__).parent))
        from model import TidalTransformerV2
        print("Loaded model from current directory")

class TransformerV2Inference:
    def __init__(self, model_path, normalization_params_path):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model file (.pth)
            normalization_params_path: Path to normalization parameters JSON
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.norm_params = None
        
        # Load normalization parameters
        self.load_normalization_params(normalization_params_path)
        
        # Load the trained model
        self.load_model(model_path)
        
        print(f"Inference engine initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model: {'Loaded' if self.model else 'Failed'}")
        print(f"  Normalization: {'Loaded' if self.norm_params else 'Failed'}")
    
    def load_normalization_params(self, params_path):
        """Load normalization parameters from JSON file."""
        try:
            with open(params_path, 'r') as f:
                self.norm_params = json.load(f)
            print(f"Loaded normalization params: mean={self.norm_params['mean']:.2f}, std={self.norm_params['std']:.2f}")
        except Exception as e:
            print(f"Error loading normalization params: {e}")
            self.norm_params = None
    
    def load_model(self, model_path):
        """Load the trained model from checkpoint file."""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model_config']
            else:
                # Default configuration if not found
                model_config = {
                    'input_length': 432,
                    'output_length': 144,
                    'd_model': 512,
                    'nhead': 16,
                    'num_encoder_layers': 8,
                    'dim_feedforward': 2048,
                    'dropout': 0.1
                }
            
            # Create model
            self.model = TidalTransformerV2(**model_config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model loaded: {total_params:,} parameters")
            
            if 'config' in checkpoint:
                print(f"Training info:")
                if 'best_val_loss' in checkpoint:
                    rmse = math.sqrt(checkpoint['best_val_loss'])
                    print(f"  Best validation RMSE: {rmse:.6f}")
                if 'epochs_completed' in checkpoint.get('config', {}):
                    print(f"  Epochs trained: {checkpoint['config']['epochs_completed']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def normalize_input(self, data):
        """Apply normalization to input data while preserving -999 missing values."""
        if self.norm_params is None:
            return data
        
        normalized = data.copy()
        valid_mask = (data != -999)
        normalized[valid_mask] = (data[valid_mask] - self.norm_params['mean']) / self.norm_params['std']
        return normalized
    
    def denormalize_output(self, data):
        """Remove normalization from output data."""
        if self.norm_params is None:
            return data
        
        return data * self.norm_params['std'] + self.norm_params['mean']
    
    def predict(self, input_sequence):
        """
        Make a prediction using the trained model.
        
        Args:
            input_sequence: NumPy array of shape (432,) containing 72 hours of water level data
                           at 10-minute intervals. Values should be in mm.
        
        Returns:
            NumPy array of shape (144,) containing 24 hours of predicted water level data
            at 10-minute intervals in mm.
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if len(input_sequence) != 432:
            raise ValueError(f"Input sequence must be 432 points, got {len(input_sequence)}")
        
        try:
            # Normalize input
            normalized_input = self.normalize_input(input_sequence)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.from_numpy(normalized_input).float()
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: (1, 432)
            input_tensor = input_tensor.unsqueeze(-1)  # Add feature dimension: (1, 432, 1)
            input_tensor = input_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Convert back to numpy and remove batch dimension
            prediction = output.cpu().numpy().squeeze()  # Shape: (144,)
            
            # Denormalize output
            prediction = self.denormalize_output(prediction)
            
            return prediction
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    
    def predict_from_sequence(self, sequence_data):
        """
        Make prediction from a training sequence (for validation).
        
        Args:
            sequence_data: Dict with 'input' and 'target' keys containing normalized data
        
        Returns:
            Dict with 'prediction', 'target', and 'input' keys
        """
        try:
            # Get the input sequence (should be normalized already)
            input_seq = sequence_data['input']  # Shape: (432,)
            target_seq = sequence_data['target']  # Shape: (144,)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.from_numpy(input_seq).float()
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: (1, 432)
            input_tensor = input_tensor.unsqueeze(-1)  # Add feature dimension: (1, 432, 1)
            input_tensor = input_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Convert back to numpy and remove batch dimension
            prediction = output.cpu().numpy().squeeze()  # Shape: (144,)
            
            return {
                'prediction': prediction.tolist(),
                'target': target_seq.tolist(),
                'input': input_seq.tolist()
            }
            
        except Exception as e:
            print(f"Error during sequence prediction: {e}")
            raise

def load_inference_engine():
    """Load the inference engine with the best available model."""
    
    # Look for model files in the shared directory
    shared_dir = Path(__file__).parent.parent / "shared"
    training_dir = Path(__file__).parent.parent / "training"
    data_dir = Path(__file__).parent.parent / "data-preparation" / "data"
    
    model_paths = [
        shared_dir / "model.pth",  # Downloaded from Modal
        training_dir / "best_model.pth",  # Local training
    ]
    
    normalization_path = data_dir / "normalization_params.json"
    
    # Find the best available model
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            print(f"Found model: {path}")
            break
    
    if model_path is None:
        raise FileNotFoundError("No trained model found. Please train the model first.")
    
    if not normalization_path.exists():
        raise FileNotFoundError(f"Normalization parameters not found at {normalization_path}")
    
    return TransformerV2Inference(model_path, normalization_path)

if __name__ == "__main__":
    # Test the inference engine
    try:
        engine = load_inference_engine()
        print("Inference engine test successful!")
        
        # Create a test input sequence
        test_input = np.random.randn(432) * 100 + 1000  # Random water levels around 1000mm
        prediction = engine.predict(test_input)
        
        print(f"Test prediction shape: {prediction.shape}")
        print(f"Prediction range: {prediction.min():.1f} - {prediction.max():.1f} mm")
        
    except Exception as e:
        print(f"Test failed: {e}")
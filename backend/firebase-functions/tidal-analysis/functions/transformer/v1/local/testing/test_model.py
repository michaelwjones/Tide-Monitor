#!/usr/bin/env python3
"""
Transformer v1 Model Testing Script
Command-line interface for testing the trained seq2seq transformer model
"""

import torch
import numpy as np
import json
import os
import sys
import argparse
import time
from pathlib import Path

# Add paths for imports (from local/testing -> v1/cloud/inference)
model_path = Path(__file__).parent.parent.parent / 'cloud' / 'inference'
sys.path.append(str(model_path.resolve()))

from model import TidalTransformer, create_model
from firebase_fetch import get_transformer_input_sequence, create_sample_data

def load_trained_model(model_path=None):
    """Load the trained transformer model"""
    if model_path is None:
        model_path = '../../cloud/training/single-runs/best_single_pass.pth'
    
    if not os.path.exists(model_path):
        print("❌ Error: Trained model not found at:", model_path)
        print("   Please train the model first by running train_transformer.py")
        return None, None, None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model with config from checkpoint
        config = checkpoint['config']
        model = TidalTransformer(
            input_dim=1,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_layers'],
            dim_feedforward=config['d_model'] * 4,
            dropout=config['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        config = checkpoint['config']
        norm_params = checkpoint['normalization_params']
        
        print(f"✅ Transformer model loaded successfully")
        print(f"   Validation loss: {checkpoint['loss']:.6f}")
        print(f"   Training epoch: {checkpoint['epoch']}")
        print(f"   Architecture: {model.get_model_info()['architecture']}")
        print(f"   Parameters: {model.get_model_info()['total_parameters']:,}")
        
        return model, config, norm_params
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def normalize_data(data, norm_params):
    """Normalize data using training parameters"""
    mean = norm_params['mean']
    std = norm_params['std']
    return (data - mean) / std

def denormalize_data(data, norm_params):
    """Denormalize data back to original scale"""
    mean = norm_params['mean']
    std = norm_params['std']
    return data * std + mean

def predict_sequence(model, input_data, norm_params):
    """
    Make a 24-hour prediction using the transformer
    
    Args:
        model: Trained transformer model
        input_data: List of 433 water level readings (72 hours @ 10min intervals)
        norm_params: Normalization parameters from training
    
    Returns:
        predictions: List of 144 predicted water levels (24 hours @ 10min intervals)
    """
    # Normalize input data
    input_array = np.array(input_data, dtype=np.float32)
    normalized_input = normalize_data(input_array, norm_params)
    
    # Convert to tensor format: (batch_size, seq_len, input_dim)
    input_tensor = torch.from_numpy(normalized_input).unsqueeze(0).unsqueeze(-1)  # (1, 433, 1)
    
    print(f"🧠 Input tensor shape: {input_tensor.shape}")
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        output_tensor = model(input_tensor)  # Direct seq2seq prediction
    inference_time = time.time() - start_time
    
    print(f"⚡ Inference completed in {inference_time*1000:.1f} ms")
    print(f"🎯 Output tensor shape: {output_tensor.shape}")
    
    # Convert back to numpy and denormalize
    normalized_predictions = output_tensor.squeeze().numpy()  # (144,)
    predictions = denormalize_data(normalized_predictions, norm_params)
    
    return predictions.tolist()

def test_with_real_data():
    """Test model with real Firebase data"""
    print("📡 Fetching real data from Firebase...")
    
    water_levels, timestamps = get_transformer_input_sequence()
    
    if not water_levels:
        print("❌ Failed to fetch real data")
        return None, None
    
    print(f"✅ Fetched {len(water_levels)} readings")
    print(f"🌊 Range: {min(water_levels):.1f} - {max(water_levels):.1f} mm")
    
    return water_levels, timestamps

def test_with_sample_data():
    """Test model with generated sample data"""
    print("🧪 Generating sample tidal data...")
    
    water_levels, timestamps = create_sample_data()
    
    print(f"✅ Generated {len(water_levels)} readings")
    print(f"🌊 Range: {min(water_levels):.1f} - {max(water_levels):.1f} mm")
    
    return water_levels, timestamps

def calculate_metrics(actual, predicted):
    """Calculate prediction accuracy metrics"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

def interactive_test():
    """Interactive testing mode"""
    print("\n🎮 Interactive Testing Mode")
    print("=" * 35)
    
    # Load model
    model, config, norm_params = load_trained_model()
    if model is None:
        return
    
    while True:
        print("\nChoose input data source:")
        print("1. 📡 Real Firebase data (last 72 hours)")
        print("2. 🧪 Generated sample data")
        print("3. 📄 Load from JSON file")
        print("4. ❌ Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            input_data, timestamps = test_with_real_data()
        elif choice == '2':
            input_data, timestamps = test_with_sample_data()
        elif choice == '3':
            filename = input("Enter JSON filename: ").strip()
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    input_data = data['water_levels']
                    timestamps = None
                print(f"✅ Loaded {len(input_data)} readings from {filename}")
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                continue
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")
            continue
        
        if not input_data or len(input_data) != 433:
            print(f"❌ Error: Need exactly 433 readings, got {len(input_data) if input_data else 0}")
            continue
        
        # Make prediction
        print("\n🔮 Generating 24-hour forecast...")
        predictions = predict_sequence(model, input_data, norm_params)
        
        print(f"✅ Generated {len(predictions)} predictions")
        print(f"📈 Prediction range: {min(predictions):.1f} - {max(predictions):.1f} mm")
        print(f"📊 First 5 predictions: {[f'{p:.1f}' for p in predictions[:5]]}")
        print(f"📊 Last 5 predictions: {[f'{p:.1f}' for p in predictions[-5:]]}")
        
        # Save results
        save = input("\n💾 Save results to JSON? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"transformer_prediction_{int(time.time())}.json"
            result_data = {
                'input_data': input_data,
                'predictions': predictions,
                'timestamps': [t.isoformat() for t in timestamps] if timestamps else None,
                'model_info': model.get_model_info(),
                'normalization': norm_params
            }
            
            with open(filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"✅ Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Test Transformer v1 Tidal Prediction Model')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--sample', action='store_true', help='Quick test with sample data')
    parser.add_argument('--real', action='store_true', help='Test with real Firebase data')
    parser.add_argument('--input', type=str, help='JSON file with input data')
    
    args = parser.parse_args()
    
    print("🔬 Transformer v1 Model Testing")
    print("=" * 35)
    
    # Load model
    model, config, norm_params = load_trained_model(args.model)
    if model is None:
        return 1
    
    # Quick sample test
    if args.sample:
        print("\n🧪 Quick sample data test...")
        input_data, timestamps = test_with_sample_data()
        
        if input_data:
            predictions = predict_sequence(model, input_data, norm_params)
            print(f"✅ Sample test completed - {len(predictions)} predictions generated")
        return 0
    
    # Real data test
    if args.real:
        print("\n📡 Real Firebase data test...")
        input_data, timestamps = test_with_real_data()
        
        if input_data:
            predictions = predict_sequence(model, input_data, norm_params)
            print(f"✅ Real data test completed - {len(predictions)} predictions generated")
        return 0
    
    # File input test
    if args.input:
        print(f"\n📄 Testing with file: {args.input}")
        try:
            with open(args.input, 'r') as f:
                data = json.load(f)
                input_data = data['water_levels']
            
            if len(input_data) != 433:
                print(f"❌ Error: File contains {len(input_data)} readings, need 433")
                return 1
            
            predictions = predict_sequence(model, input_data, norm_params)
            print(f"✅ File test completed - {len(predictions)} predictions generated")
        
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return 1
        
        return 0
    
    # Default: interactive mode
    interactive_test()
    return 0

if __name__ == "__main__":
    sys.exit(main())
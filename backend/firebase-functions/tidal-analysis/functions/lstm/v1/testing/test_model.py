#!/usr/bin/env python3
"""
LSTM v1 Model Testing Script
Simple command-line interface for testing the trained model
"""

import torch
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'training'))

from model import TidalLSTM

def load_trained_model():
    """Load the trained LSTM model"""
    model_path = '../training/trained_models/best_model.pth'
    
    if not os.path.exists(model_path):
        print("❌ Error: Trained model not found at:", model_path)
        print("   Please train the model first by running train_lstm.py")
        return None, None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        
        # Initialize model with saved config
        model = TidalLSTM(
            input_size=1,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
        print(f"   Training epoch: {checkpoint['epoch']}")
        
        return model, config
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def load_normalization_params():
    """Load normalization parameters from training"""
    norm_path = '../data-preparation/data/normalization_params.json'
    
    if not os.path.exists(norm_path):
        print("⚠️  Warning: Normalization parameters not found")
        print("   Using default normalization (mean=1850, std=300)")
        return {'mean': 1850.0, 'std': 300.0}
    
    try:
        with open(norm_path, 'r') as f:
            params = json.load(f)
        print(f"✅ Normalization params loaded (mean={params['mean']:.1f}, std={params['std']:.1f})")
        return params
    except Exception as e:
        print(f"⚠️  Error loading normalization: {e}")
        return {'mean': 1850.0, 'std': 300.0}

def normalize_data(data, params):
    """Normalize data using training parameters"""
    return (data - params['mean']) / params['std']

def denormalize_data(data, params):
    """Denormalize data using training parameters"""
    return data * params['std'] + params['mean']

def predict_next_reading(model, input_sequence, norm_params):
    """Make a single prediction using the LSTM model"""
    try:
        # Normalize input
        normalized_input = normalize_data(np.array(input_sequence), norm_params)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(normalized_input).unsqueeze(0).unsqueeze(-1)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Denormalize prediction
        pred_value = denormalize_data(prediction.item(), norm_params)
        
        return pred_value
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def generate_sample_data():
    """Generate realistic sample tidal data for testing"""
    print("📊 Generating sample tidal data...")
    
    base_level = 1850  # Base water level in mm
    tidal_range = 300  # Tidal variation
    
    readings = []
    for i in range(72):
        # Simulate tidal cycle + noise
        tidal_component = np.sin((i / 72) * 2 * np.pi) * tidal_range
        noise = np.random.normal(0, 10)  # 10mm noise
        reading = base_level + tidal_component + noise
        readings.append(reading)
    
    return readings

def interactive_test():
    """Interactive testing interface"""
    print("🧪 LSTM v1 Model Testing Interface")
    print("=" * 40)
    
    # Load model
    model, config = load_trained_model()
    if model is None:
        return
    
    # Load normalization parameters
    norm_params = load_normalization_params()
    
    while True:
        print("\nOptions:")
        print("1. Test with sample data")
        print("2. Enter custom data")
        print("3. Load data from file")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # Generate and test sample data
            readings = generate_sample_data()
            print(f"📈 Sample data: {readings[0]:.1f}mm to {readings[-1]:.1f}mm")
            
        elif choice == '2':
            # Manual data entry
            print("\nEnter 72 water level readings (mm), one per line:")
            readings = []
            try:
                for i in range(72):
                    value = float(input(f"Reading {i+1}: "))
                    readings.append(value)
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input or cancelled")
                continue
                
        elif choice == '3':
            # Load from file
            filename = input("Enter filename: ").strip()
            try:
                with open(filename, 'r') as f:
                    readings = [float(line.strip()) for line in f if line.strip()]
                
                if len(readings) < 72:
                    print(f"❌ Need 72 readings, got {len(readings)}")
                    continue
                    
                readings = readings[:72]  # Use first 72
                print(f"✅ Loaded {len(readings)} readings from {filename}")
                
            except (FileNotFoundError, ValueError) as e:
                print(f"❌ Error loading file: {e}")
                continue
                
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")
            continue
        
        # Make prediction
        if len(readings) == 72:
            prediction = predict_next_reading(model, readings, norm_params)
            
            if prediction is not None:
                print(f"\n📊 Test Results:")
                print(f"   Input range: {min(readings):.1f} - {max(readings):.1f} mm")
                print(f"   Input average: {sum(readings)/len(readings):.1f} mm")
                print(f"   🎯 Predicted next reading: {prediction:.1f} mm")
                print(f"   📈 Change from last: {prediction - readings[-1]:+.1f} mm")
            else:
                print("❌ Prediction failed")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        # Quick test with sample data
        print("🧪 Quick Test Mode")
        print("=" * 20)
        
        model, config = load_trained_model()
        if model is None:
            return
            
        norm_params = load_normalization_params()
        readings = generate_sample_data()
        prediction = predict_next_reading(model, readings, norm_params)
        
        if prediction is not None:
            print(f"✅ Sample prediction: {prediction:.1f} mm")
        else:
            print("❌ Test failed")
    else:
        # Interactive mode
        interactive_test()

if __name__ == "__main__":
    main()
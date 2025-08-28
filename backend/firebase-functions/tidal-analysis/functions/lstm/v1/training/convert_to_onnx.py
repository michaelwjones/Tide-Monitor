import torch
import onnx
import onnxruntime as ort
import numpy as np
import json
import os

from model import TidalLSTM

def load_trained_model():
    """Load the best trained model"""
    try:
        checkpoint = torch.load('trained_models/best_model.pth', map_location='cpu')
        config = checkpoint['config']
        
        # Initialize model with saved configuration
        model = TidalLSTM(
            input_size=1,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        
        # Load trained weights and ensure CPU for ONNX conversion
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Move model to CPU for ONNX conversion (required for deployment)
        model = model.cpu()
        
        print(f"Loaded trained model (val_loss: {checkpoint['val_loss']:.6f})")
        return model, config
        
    except FileNotFoundError:
        print("Error: No trained model found. Run train_lstm.py first.")
        return None, None

def convert_to_onnx(model):
    """Convert PyTorch model to ONNX format"""
    print("Converting model to ONNX format...")
    
    # Create dummy input for tracing (sequence_length=4320, input_size=1)
    dummy_input = torch.randn(1, 4320, 1)
    
    # Create output directory
    os.makedirs('inference', exist_ok=True)
    
    # Export to ONNX
    onnx_path = 'inference/tidal_lstm.onnx'
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path

def verify_onnx_model(onnx_path):
    """Verify ONNX model works correctly"""
    print("Verifying ONNX model...")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model structure is valid")
    
    # Test with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = np.random.randn(1, 4320, 1).astype(np.float32)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_output = ort_session.run(None, ort_inputs)
    
    print(f"✓ ONNX Runtime inference successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {ort_output[0].shape}")
    print(f"  Output value: {ort_output[0][0][0]:.6f}")
    
    return True

def compare_pytorch_onnx(pytorch_model, onnx_path):
    """Compare PyTorch and ONNX model outputs"""
    print("Comparing PyTorch vs ONNX outputs...")
    
    # Create test input
    test_input = torch.randn(1, 4320, 1)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy().astype(np.float32)}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    diff = np.abs(pytorch_output - onnx_output).max()
    print(f"Maximum difference: {diff:.8f}")
    
    if diff < 1e-5:
        print("✓ PyTorch and ONNX outputs match closely")
        return True
    else:
        print("⚠ Large difference between PyTorch and ONNX outputs")
        return False

def create_deployment_files(config):
    """Create files needed for Firebase Functions deployment"""
    
    # Copy normalization parameters
    try:
        import shutil
        shutil.copy('../data-preparation/data/normalization_params.json', 'inference/')
        print("✓ Copied normalization parameters")
    except FileNotFoundError:
        print("⚠ Warning: normalization_params.json not found")
    
    # Create model metadata
    model_metadata = {
        'model_type': 'TidalLSTM',
        'version': 'v1',
        'training_config': config,
        'input_shape': [1, 4320, 1],  # [batch_size, sequence_length, input_size]
        'output_shape': [1, 1],       # [batch_size, prediction]
        'sequence_length': 4320,      # 72 hours in minutes
        'prediction_steps': 1,        # Single-step prediction
        'created_at': np.datetime64('now').astype(str)
    }
    
    with open('inference/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print("✓ Created model metadata")

def main():
    """Main conversion function"""
    print("LSTM v1 ONNX Conversion Pipeline")
    print("=" * 35)
    
    # Load trained model
    model, config = load_trained_model()
    if model is None:
        return
    
    # Convert to ONNX
    onnx_path = convert_to_onnx(model)
    
    # Verify ONNX model
    if not verify_onnx_model(onnx_path):
        return
    
    # Compare outputs
    if not compare_pytorch_onnx(model, onnx_path):
        return
    
    # Create deployment files
    create_deployment_files(config)
    
    print(f"\n✓ Conversion completed successfully!")
    print(f"Deployment files created in: inference/")
    print(f"  - tidal_lstm.onnx")
    print(f"  - model_metadata.json")
    print(f"  - normalization_params.json")
    print(f"\nNext step: Deploy Firebase Functions with the ONNX model")

if __name__ == "__main__":
    main()
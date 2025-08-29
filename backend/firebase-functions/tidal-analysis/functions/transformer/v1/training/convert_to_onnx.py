import torch
import onnx
import onnxruntime as ort
import numpy as np
import json
import os
import argparse
from model import create_model

class TransformerONNXExporter:
    """
    Export trained transformer model to ONNX format for Firebase Functions inference.
    
    Features:
    - Model validation and optimization
    - Input/output shape verification
    - Normalization parameters embedding
    - Inference performance testing
    """
    
    def __init__(self, checkpoint_path, output_dir='onnx_models'):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and checkpoint
        self.model = create_model()
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load trained model from checkpoint"""
        print(f"Loading checkpoint from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store training metadata
        self.training_loss = checkpoint['loss']
        self.config = checkpoint['config']
        self.normalization_params = checkpoint['normalization_params']
        
        print(f"Model loaded successfully!")
        print(f"Training loss: {self.training_loss:.6f}")
        print(f"Normalization: mean={self.normalization_params['mean']:.2f}, "
              f"std={self.normalization_params['std']:.2f}")
    
    def create_dummy_input(self):
        """Create dummy input for ONNX export"""
        # Input: (batch_size, sequence_length, input_dim)
        # 4320 time steps (72 hours), batch size 1 for inference
        dummy_input = torch.randn(1, 4320, 1).to(self.device)
        return dummy_input
    
    def export_onnx(self):
        """Export PyTorch model to ONNX format"""
        print("Exporting model to ONNX format...")
        
        dummy_input = self.create_dummy_input()
        onnx_path = os.path.join(self.output_dir, 'transformer_tidal_v1.onnx')
        
        # Export with dynamic batch size support
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"ONNX model saved to: {onnx_path}")
        
        # Verify ONNX model
        self.verify_onnx(onnx_path)
        
        return onnx_path
    
    def verify_onnx(self, onnx_path):
        """Verify ONNX model correctness"""
        print("Verifying ONNX model...")
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure is valid")
        
        # Test inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()[0]
        
        print(f"✓ Input shape: {input_info.shape} ({input_info.type})")
        print(f"✓ Output shape: {output_info.shape} ({output_info.type})")
        
        # Test with dummy data
        dummy_input = np.random.randn(1, 4320, 1).astype(np.float32)
        ort_outputs = ort_session.run(None, {input_info.name: dummy_input})
        
        print(f"✓ ONNX inference successful, output shape: {ort_outputs[0].shape}")
        
        # Compare with PyTorch output
        with torch.no_grad():
            torch_input = torch.from_numpy(dummy_input).to(self.device)
            torch_output = self.model(torch_input).cpu().numpy()
        
        # Check numerical consistency
        max_diff = np.max(np.abs(torch_output - ort_outputs[0]))
        print(f"✓ Max difference between PyTorch and ONNX: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("✓ Model conversion successful - numerical outputs match")
        else:
            print("⚠ Warning: Large numerical difference detected")
    
    def create_metadata(self):
        """Create metadata file for Firebase Functions"""
        metadata = {
            'model_info': self.model.get_model_info(),
            'training': {
                'final_loss': float(self.training_loss),
                'config': self.config
            },
            'normalization': self.normalization_params,
            'inference': {
                'input_shape': [1, 4320, 1],
                'output_shape': [1, 1440, 1],
                'input_description': 'Water level time series (72 hours, normalized)',
                'output_description': '24-hour water level predictions (normalized)',
                'framework': 'ONNX Runtime'
            },
            'export_info': {
                'export_date': torch.datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'onnx_opset': 11
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
        return metadata_path
    
    def benchmark_inference(self, onnx_path, num_runs=10):
        """Benchmark ONNX model inference speed"""
        print(f"Benchmarking inference speed ({num_runs} runs)...")
        
        ort_session = ort.InferenceSession(onnx_path)
        dummy_input = np.random.randn(1, 4320, 1).astype(np.float32)
        input_name = ort_session.get_inputs()[0].name
        
        # Warmup runs
        for _ in range(3):
            ort_session.run(None, {input_name: dummy_input})
        
        # Benchmark runs
        import time
        times = []
        for _ in range(num_runs):
            start = time.time()
            ort_session.run(None, {input_name: dummy_input})
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"✓ Average inference time: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
        print(f"✓ Throughput: {1/avg_time:.1f} predictions/second")
        
        return avg_time
    
    def export_for_firebase(self):
        """Complete export pipeline for Firebase Functions deployment"""
        print("Exporting transformer model for Firebase Functions...")
        print("=" * 60)
        
        # Export ONNX model
        onnx_path = self.export_onnx()
        
        # Create metadata
        metadata_path = self.create_metadata()
        
        # Benchmark performance
        inference_time = self.benchmark_inference(onnx_path)
        
        # Create deployment summary
        print("\n" + "=" * 60)
        print("Export Summary:")
        print(f"✓ ONNX model: {onnx_path}")
        print(f"✓ Metadata: {metadata_path}")
        print(f"✓ Model size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")
        print(f"✓ Inference time: {inference_time*1000:.1f} ms")
        print(f"✓ Training loss: {self.training_loss:.6f}")
        
        print("\nNext steps:")
        print("1. Copy ONNX model and metadata to inference/ directory")
        print("2. Deploy Firebase Functions for model inference")
        print("3. Test end-to-end prediction pipeline")
        
        return onnx_path, metadata_path

def main():
    parser = argparse.ArgumentParser(description='Convert Transformer to ONNX')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='onnx_models',
                       help='Output directory for ONNX models')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first using train_transformer.py")
        return
    
    print("Transformer v1 ONNX Export")
    print("=" * 30)
    
    # Create exporter and run export
    exporter = TransformerONNXExporter(args.checkpoint, args.output_dir)
    onnx_path, metadata_path = exporter.export_for_firebase()
    
    print(f"\nExport completed successfully!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
GPU Detection and Performance Test for LSTM Training

Quick script to check GPU availability and estimate training performance.
Run this before starting LSTM training to verify optimal setup.
"""

import torch
import time
import sys

def check_gpu_support():
    """Check GPU availability and provide detailed information"""
    print("=" * 60)
    print("LSTM v1 GPU Detection and Performance Check")
    print("=" * 60)
    
    # Basic PyTorch info
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check for CUDA compilation
    cuda_compiled = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
    print(f"CUDA Compiled: {cuda_compiled}")
    if cuda_compiled:
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Check CUDA availability with error handling
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Additional test to ensure CUDA actually works
            test_tensor = torch.tensor([1.0])
            test_tensor.cuda()
        print(f"CUDA Available: {cuda_available}")
    except (RuntimeError, AssertionError) as e:
        print(f"CUDA Available: False (Error: {e})")
        cuda_available = False
    
    if cuda_available and cuda_compiled:
        print(f"GPU Count: {torch.cuda.device_count()}")
        print()
        
        # Detailed GPU information
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print()
        
        # Performance test
        print("Running quick performance test...")
        run_performance_test(cuda_available, cuda_compiled)
        
        # Training recommendations
        print_training_recommendations(True)
        
    else:
        if not cuda_compiled:
            print("PyTorch not compiled with CUDA support.")
            print("Current installation is CPU-only.")
        else:
            print("No CUDA-capable GPU detected.")
        print("Training will use CPU (slower but still functional).")
        print()
        print_training_recommendations(False)

def run_performance_test(cuda_available, cuda_compiled):
    """Run a simple performance test on GPU vs CPU"""
    try:
        # Test data similar to LSTM training
        batch_size = 32
        seq_length = 4320  # 72 hours
        input_size = 1
        
        # Create test tensors
        test_input = torch.randn(batch_size, seq_length, input_size)
        
        # CPU test
        start_time = time.time()
        cpu_result = torch.matmul(test_input, test_input.transpose(-1, -2))
        cpu_time = time.time() - start_time
        
        # GPU test (only if CUDA is compiled and available)
        if cuda_available and cuda_compiled:
            torch.cuda.empty_cache()
            test_input_gpu = test_input.cuda()
            
            # Warm up GPU
            _ = torch.matmul(test_input_gpu, test_input_gpu.transpose(-1, -2))
            torch.cuda.synchronize()
            
            # Actual test
            start_time = time.time()
            gpu_result = torch.matmul(test_input_gpu, test_input_gpu.transpose(-1, -2))
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
            print(f"CPU Time: {cpu_time:.3f}s")
            print(f"GPU Time: {gpu_time:.3f}s")
            print(f"GPU Speedup: {speedup:.1f}x faster")
            
            if speedup > 2:
                print("🚀 Excellent GPU acceleration! Training will be significantly faster.")
            elif speedup > 1.2:
                print("✅ Good GPU acceleration detected.")
            else:
                print("⚠️  Limited GPU speedup. Check GPU drivers and CUDA installation.")
            
    except Exception as e:
        print(f"Performance test failed: {e}")
        print("GPU may not be properly configured.")

def print_training_recommendations(has_gpu):
    """Print training recommendations based on GPU availability"""
    print("=" * 60)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 60)
    
    # Check for CUDA compilation issue
    cuda_compiled = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
    
    if has_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print("✅ GPU Training Enabled")
        print(f"  • Expected training time: 15-30 minutes (vs 2-4 hours on CPU)")
        print(f"  • Batch size: 64 (optimal for GPU)")
        print(f"  • Memory usage: ~2-4 GB GPU memory")
        
        if gpu_memory < 4:
            print("  ⚠️  Low GPU memory detected. Consider reducing batch size if training fails.")
        elif gpu_memory >= 8:
            print("  🚀 Plenty of GPU memory for large batch training!")
            
    else:
        print("⚠️  CPU Training Only")
        print("  • Expected training time: 2-4 hours")
        print("  • Batch size: 32 (optimal for CPU)")
        print("  • Memory usage: ~4-8 GB system RAM")
        print()
        
        if not cuda_compiled:
            print("🔧 ISSUE: PyTorch not compiled with CUDA support")
            print("To enable GPU training:")
            print("  1. Run: install-pytorch-interactive.bat")
            print("  2. Select option [2] or [3] for CUDA support")
            print("  3. Or manually reinstall:")
            print("     pip uninstall torch")
            print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("To enable GPU training:")
            print("  1. Install CUDA-compatible GPU drivers")
            print("  2. Verify GPU hardware is supported")
            print("  3. Restart system after driver installation")

def main():
    """Main function"""
    try:
        check_gpu_support()
        
        print()
        print("=" * 60)
        print("Ready to start LSTM training!")
        print("Run: python training/train_lstm.py")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during GPU check: {e}")
        print("You may still be able to train on CPU.")
        sys.exit(1)

if __name__ == "__main__":
    main()
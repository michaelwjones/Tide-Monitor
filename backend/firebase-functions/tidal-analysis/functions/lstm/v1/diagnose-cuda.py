#!/usr/bin/env python3
"""
CUDA Installation Diagnosis Tool

This script helps diagnose why PyTorch CUDA installation keeps failing.
It checks system requirements and suggests solutions.
"""

import sys
import subprocess
import platform
import os

def run_command(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_system_info():
    """Check basic system information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print()

def check_nvidia_drivers():
    """Check NVIDIA drivers and CUDA installation"""
    print("=" * 60)
    print("NVIDIA DRIVER & CUDA CHECK")
    print("=" * 60)
    
    # Check nvidia-smi
    stdout, stderr, returncode = run_command("nvidia-smi")
    if returncode == 0:
        print("✅ NVIDIA drivers detected!")
        lines = stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"  {line.strip()}")
            elif 'CUDA Version' in line:
                print(f"  {line.strip()}")
        print()
    else:
        print("❌ NVIDIA drivers not found or not working")
        print("  Error:", stderr if stderr else "nvidia-smi command not found")
        print("  Solution: Install NVIDIA GPU drivers from nvidia.com")
        print()
        return False
    
    # Check nvcc (CUDA compiler)
    stdout, stderr, returncode = run_command("nvcc --version")
    if returncode == 0:
        print("✅ CUDA toolkit detected!")
        for line in stdout.split('\n'):
            if 'release' in line.lower():
                print(f"  {line.strip()}")
        print()
    else:
        print("⚠️  CUDA toolkit not found (this is OK for PyTorch)")
        print("  PyTorch includes its own CUDA runtime")
        print()
    
    return True

def check_pytorch_attempts():
    """Try different PyTorch installation methods"""
    print("=" * 60)  
    print("PYTORCH INSTALLATION ATTEMPTS")
    print("=" * 60)
    
    methods = [
        ("CUDA 12.1", "pip install torch --index-url https://download.pytorch.org/whl/cu121"),
        ("CUDA 11.8", "pip install torch --index-url https://download.pytorch.org/whl/cu118"), 
        ("Default", "pip install torch"),
        ("CPU-only", "pip install torch --index-url https://download.pytorch.org/whl/cpu")
    ]
    
    for name, command in methods:
        print(f"Testing {name} installation...")
        
        # Uninstall previous version
        run_command("pip uninstall torch -y")
        
        # Install new version
        stdout, stderr, returncode = run_command(command)
        
        if returncode != 0:
            print(f"  ❌ Installation failed: {stderr}")
            continue
            
        # Check what we got
        stdout, stderr, returncode = run_command('python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"')
        
        if returncode != 0:
            print(f"  ❌ Import failed: {stderr}")
            continue
            
        lines = stdout.split('\n')
        if len(lines) >= 2:
            version = lines[0].strip()
            cuda_available = lines[1].strip()
            
            if '+cu' in version:
                print(f"  ✅ SUCCESS: {version}, CUDA Available: {cuda_available}")
                if cuda_available == 'True':
                    print(f"  🚀 This method works! Use: {command}")
                    return command
                else:
                    print(f"  ⚠️  CUDA compiled but not available (driver issue?)")
            else:
                print(f"  ⚠️  CPU version: {version}")
        
        print()
    
    return None

def check_pip_environment():
    """Check pip environment and cache issues"""
    print("=" * 60)
    print("PIP ENVIRONMENT CHECK") 
    print("=" * 60)
    
    # Check pip version
    stdout, stderr, returncode = run_command("pip --version")
    print(f"Pip version: {stdout}")
    
    # Check pip cache
    stdout, stderr, returncode = run_command("pip cache dir")
    if returncode == 0:
        print(f"Pip cache: {stdout}")
        print("Suggestion: Clear pip cache with 'pip cache purge' if issues persist")
    
    # Check for conda
    stdout, stderr, returncode = run_command("conda --version")
    if returncode == 0:
        print(f"⚠️  Conda detected: {stdout}")
        print("  Note: Conda can interfere with pip PyTorch installation")
        print("  Consider using: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia")
    
    print()

def provide_recommendations():
    """Provide final recommendations"""
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("Based on the diagnosis above:")
    print()
    print("1. If NVIDIA drivers are missing:")
    print("   - Install latest GPU drivers from nvidia.com")
    print("   - Reboot system after installation")
    print()
    print("2. If drivers are present but PyTorch CUDA fails:")
    print("   - Try: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("   - Or: Use install-pytorch-interactive.bat for step-by-step")
    print()
    print("3. If all CUDA attempts fail:")
    print("   - Your GPU may not be CUDA-compatible")
    print("   - Or your CUDA driver version is too old")
    print("   - CPU training will still work (just slower)")
    print()
    print("4. If you have conda:")
    print("   - Try: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia")
    print("   - This sometimes works better than pip")
    print()

def main():
    """Main diagnosis function"""
    print("LSTM v1 CUDA Installation Diagnosis")
    print("Finding out why PyTorch CUDA installation keeps failing...")
    print()
    
    check_system_info()
    
    has_nvidia = check_nvidia_drivers()
    if not has_nvidia:
        print("🚨 Primary issue: No NVIDIA drivers detected!")
        provide_recommendations()
        return
    
    check_pip_environment()
    
    print("Testing different PyTorch installation methods...")
    print("This will temporarily install/uninstall PyTorch versions...")
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    working_method = check_pytorch_attempts()
    
    if working_method:
        print(f"🎉 Found working installation method: {working_method}")
    else:
        print("😞 No PyTorch CUDA installation method worked")
    
    provide_recommendations()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDiagnosis cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Diagnosis error: {e}")
        sys.exit(1)
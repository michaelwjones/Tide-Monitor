@echo off
echo =============================================
echo   PyTorch Installation for Transformer v1
echo =============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Current Python version:
python --version
echo.

:: Check if PyTorch is already installed
echo Checking existing PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__} already installed')" 2>nul
if not errorlevel 1 (
    echo.
    set /p reinstall=PyTorch is already installed. Reinstall? (y/n): 
    if /i not "%reinstall%"=="y" (
        echo Keeping existing installation.
        goto :check_cuda
    )
)

echo.
echo PyTorch Installation Options:
echo 1. CPU only (works on all systems, slower training)
echo 2. CUDA 11.8 (NVIDIA GPU required, faster training)  
echo 3. CUDA 12.1 (NVIDIA GPU required, latest)
echo 4. Auto-detect (recommended)
echo 5. Cancel
echo.

set /p choice=Enter choice (1-5): 

if "%choice%"=="1" goto :install_cpu
if "%choice%"=="2" goto :install_cuda118
if "%choice%"=="3" goto :install_cuda121
if "%choice%"=="4" goto :auto_detect
if "%choice%"=="5" goto :end
echo Invalid choice.
goto :end

:auto_detect
echo.
echo Auto-detecting GPU capabilities...
python -c "
import subprocess
import sys
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print('NVIDIA GPU detected')
        print('Recommending CUDA 12.1 installation')
        sys.exit(1)
    else:
        print('No NVIDIA GPU detected')
        print('Recommending CPU installation')
        sys.exit(0)
except FileNotFoundError:
    print('nvidia-smi not found - no NVIDIA GPU detected')
    print('Recommending CPU installation')
    sys.exit(0)
"
if errorlevel 1 (
    echo.
    set /p gpu_confirm=Install CUDA version? (y/n): 
    if /i "%gpu_confirm%"=="y" goto :install_cuda121
)
goto :install_cpu

:install_cpu
echo.
echo Installing PyTorch CPU version...
echo This will work on all systems but training will be slower.
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorboard numpy pandas requests flask
echo.
echo CPU installation completed!
goto :verify

:install_cuda118
echo.
echo Installing PyTorch with CUDA 11.8 support...
echo Make sure you have CUDA 11.8 installed on your system.
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard numpy pandas requests flask-gpu
echo.
echo CUDA 11.8 installation completed!
goto :verify

:install_cuda121
echo.
echo Installing PyTorch with CUDA 12.1 support...
echo Make sure you have CUDA 12.1 installed on your system.
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard numpy pandas requests flask-gpu
echo.
echo CUDA 12.1 installation completed!
goto :verify

:verify
echo.
echo Verifying installation...
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Using CPU for training')
"

:check_cuda
echo.
echo Testing tensor operations...
python -c "
import torch
import time

# Create test tensors
print('Creating test tensors...')
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Testing on GPU: {torch.cuda.get_device_name(0)}')
    x_gpu = x.to(device)
    y_gpu = y.to(device)
    
    start = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    gpu_time = time.time() - start
    print(f'GPU computation time: {gpu_time:.4f} seconds')
else:
    device = torch.device('cpu')
    print('Testing on CPU...')

start = time.time()
z_cpu = torch.matmul(x, y)
cpu_time = time.time() - start
print(f'CPU computation time: {cpu_time:.4f} seconds')

if torch.cuda.is_available():
    speedup = cpu_time / gpu_time
    print(f'GPU speedup: {speedup:.1f}x faster than CPU')

print('✅ Installation verified successfully!')
"

if errorlevel 1 (
    echo.
    echo ❌ Verification failed. There may be issues with your installation.
    echo Please check the error messages above.
) else (
    echo.
    echo ✅ PyTorch installation and verification completed successfully!
    echo.
    echo You can now:
    echo 1. Run data-preparation scripts to fetch and create training data
    echo 2. Train the transformer model using train_transformer.py
    echo 3. Test models with local PyTorch server
    echo.
    echo For a complete workflow, run: setup-complete-transformer-v1.bat
)

:end
echo.
pause
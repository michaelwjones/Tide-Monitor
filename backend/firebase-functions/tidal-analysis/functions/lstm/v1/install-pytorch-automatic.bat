@echo off
echo Setting up PyTorch environment for LSTM Tidal Analysis...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Installing essential packages only...
echo.

REM Try multiple CUDA installation methods
echo Attempting to install PyTorch with GPU support...
echo Trying CUDA 12.1 first...
pip install torch --index-url https://download.pytorch.org/whl/cu121

REM Check if CUDA 12.1 worked
python -c "import torch; exit(0 if '+cu121' in torch.__version__ else 1)" >nul 2>&1
if not errorlevel 1 (
    echo CUDA 12.1 installation successful!
    goto CUDA_SUCCESS
)

echo CUDA 12.1 failed, trying CUDA 11.8...
pip uninstall torch -y >nul 2>&1
pip install torch --index-url https://download.pytorch.org/whl/cu118

REM Check if CUDA 11.8 worked  
python -c "import torch; exit(0 if '+cu118' in torch.__version__ else 1)" >nul 2>&1
if not errorlevel 1 (
    echo CUDA 11.8 installation successful!
    goto CUDA_SUCCESS
)

echo Both CUDA versions failed, trying default PyTorch...
pip uninstall torch -y >nul 2>&1
pip install torch

:CUDA_SUCCESS

REM Final verification of what we got
python -c "import torch; has_cuda = '+cu' in torch.__version__; print('Final result:', torch.__version__); print('CUDA Support:', 'YES' if has_cuda else 'NO')"

REM Verify installation and show device info
echo.
echo Checking PyTorch installation and GPU availability...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

REM Install core packages for data processing and visualization
pip install numpy pandas matplotlib

REM Install HTTP requests library (required for Firebase data fetching)
pip install requests

REM Install ONNX for model conversion
pip install onnx onnxruntime

REM Install progress bars for training
pip install tqdm

echo.
echo Verifying essential packages...
python -c "import torch; print(' PyTorch installed')"
python -c "import numpy; print(' NumPy installed')"
python -c "import pandas; print(' Pandas installed')"
python -c "import requests; print(' Requests installed')"
python -c "import onnx; print(' ONNX installed')"
python -c "import onnxruntime; print(' ONNX Runtime installed')"
python -c "import matplotlib; print(' Matplotlib installed')"
python -c "import tqdm; print(' TQDM installed')"

echo.
echo GPU Training Status:
python -c "import torch; cuda_available = torch.cuda.is_available(); has_cuda_build = '+cu' in torch.__version__; print('  PyTorch Build:', 'CUDA-enabled' if has_cuda_build else 'CPU-only'); print('  GPU Available:', 'YES -', torch.cuda.get_device_name(0) if cuda_available else 'NO'); print('  Recommendation:', 'GPU training ready!' if cuda_available else 'Use install-pytorch-interactive.bat for troubleshooting' if not has_cuda_build else 'Install CUDA drivers')"

echo.
echo Setup complete! You can now run:
echo   - data-preparation scripts to fetch and prepare training data
echo   - training scripts to train the network
echo.
echo IMPORTANT: The data-preparation/data/ folder will contain large files
echo and should NEVER be committed to git (it's in .gitignore)
echo.
pause
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

REM Install PyTorch (CPU version for Windows)
pip install torch --index-url https://download.pytorch.org/whl/cpu

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
echo Setup complete! You can now run:
echo   - data-preparation scripts to fetch and prepare training data
echo   - training scripts to train the network
echo.
echo IMPORTANT: The data-preparation/data/ folder will contain large files
echo and should NEVER be committed to git (it's in .gitignore)
echo.
pause
@echo off
echo Testing PyTorch CUDA Installation Methods
echo =========================================
echo.

echo Step 1: System Check
echo --------------------
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ❌ No NVIDIA GPU detected or drivers not installed
    echo    Install NVIDIA drivers first: https://nvidia.com/drivers
    echo    Then run this test again
    pause
    exit /b 1
) else (
    echo ✅ NVIDIA GPU detected
    nvidia-smi | findstr "Driver Version"
)
echo.

echo Step 2: Clean Slate
echo -------------------
echo Removing any existing PyTorch...
pip uninstall torch -y >nul 2>&1
echo ✅ Cleaned up existing installations
echo.

echo Step 3: Test CUDA 12.1 Installation
echo -----------------------------------
echo Installing PyTorch with CUDA 12.1...
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo Checking installation...
python -c "import torch; print('Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>nul

echo.
echo ==========================================
echo If you see CUDA Available: True above, you're good!
echo If not, try the interactive installer: install-pytorch-interactive.bat
echo ==========================================
pause
@echo off
REM Transformer v1 - Local Training Launcher
REM Auto-detects GPU and uses appropriate configuration

echo ============================================================
echo Transformer v1 - Local Training
echo ============================================================
echo.
echo This script will:
echo - Auto-detect your GPU and optimize accordingly
echo - Use hyperparameters from Modal H100 sweep (adapted for your hardware)
echo - GTX 1070: batch_size=16, expected val_loss ~0.037-0.040
echo - Other GPUs: batch_size=32, expected val_loss ~0.0365
echo.

REM Navigate to training directory (we're already in local folder)
cd /d "%~dp0training"

REM Check for training data
if not exist "..\data-preparation\data\X_train.npy" (
    echo ERROR: Training data not found!
    echo Please run data preparation first:
    echo   cd ..\data-preparation
    echo   python fetch_firebase_data.py
    echo   python create_training_data.py
    echo.
    pause
    exit /b 1
)

echo Training data found. Starting local training...
echo.

REM Run training with auto-detection
python train_transformer.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo LOCAL TRAINING COMPLETED SUCCESSFULLY!
    echo ============================================================
    echo.
    echo Results saved to:
    echo - Best model: checkpoints\best.pth
    echo - Training logs: runs\[timestamp]\
    echo.
    echo Next steps:
    echo 1. Test model: ..\testing\start-server.bat
    echo 2. Deploy to Firebase: ..\..\..\..\deploy-transformer-v1.bat
    echo.
) else (
    echo.
    echo ============================================================
    echo TRAINING FAILED
    echo ============================================================
    echo Check error messages above for details.
    echo Common issues:
    echo - Out of GPU memory: Script auto-adjusts for GTX 1070
    echo - CUDA errors: Update PyTorch/CUDA drivers
    echo.
)

pause
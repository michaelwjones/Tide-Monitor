@echo off
REM Transformer v1 - Final Model Training with Optimized Hyperparameters
echo Transformer v1 - Final Model Training
echo Using optimized hyperparameters from Modal H100 sweep
echo.

REM Change to training directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

echo Starting final model training...
echo This will use the best hyperparameters found from cloud optimization:
echo - d_model: 512, nhead: 16, layers: 4
echo - Learning Rate: 5.85e-05
echo - Batch Size: 32
echo - Expected Val Loss: ~0.0365
echo.

REM Run training
python train_transformer.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo Training completed successfully!
    echo Next steps:
    echo 1. Check checkpoints/best.pth for the trained model
    echo 2. Run model_server.py to test locally
    echo 3. Use the testing interface to validate performance
    echo 4. Deploy to Firebase Functions
) else (
    echo.
    echo Training failed with error code %ERRORLEVEL%
    echo Check the logs for details
)

echo.
pause
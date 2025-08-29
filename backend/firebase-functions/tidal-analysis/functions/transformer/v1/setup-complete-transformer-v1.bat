@echo off
echo ========================================================
echo   Complete Transformer v1 Setup and Training Pipeline
echo ========================================================
echo.

:: Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or later
    pause
    exit /b 1
)

echo Current Python version:
python --version
echo.

:: Check PyTorch availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo Warning: PyTorch not found
    echo You may need to install PyTorch for training
    echo.
)

:: Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if not errorlevel 1 (
    python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
)
echo.

echo ========================================================
echo   Setup Options
echo ========================================================
echo 1. Install PyTorch (CPU version)
echo 2. Install PyTorch (CUDA version - requires NVIDIA GPU)
echo 3. Fetch data and create training dataset
echo 4. Train transformer model
echo 5. Convert model to ONNX format
echo 6. Test model locally
echo 7. Deploy to Firebase Functions
echo 8. Run complete pipeline (3-7)
echo 9. Exit
echo.

set /p choice=Enter your choice (1-9): 

if "%choice%"=="1" (
    echo Installing PyTorch CPU version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install tensorboard
    echo PyTorch CPU installation completed.
) else if "%choice%"=="2" (
    echo Installing PyTorch CUDA version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install tensorboard
    echo PyTorch CUDA installation completed.
) else if "%choice%"=="3" (
    echo ========== Data Preparation ==========
    cd data-preparation
    echo Fetching Firebase data...
    python fetch_firebase_data.py
    if not errorlevel 1 (
        echo Creating training dataset...
        python create_training_data.py
    )
    cd ..
) else if "%choice%"=="4" (
    echo ========== Training Transformer ==========
    cd training
    echo Starting transformer training...
    echo This may take several hours depending on your hardware.
    echo.
    python train_transformer.py
    cd ..
) else if "%choice%"=="5" (
    echo ========== Converting to ONNX ==========
    cd training
    echo Converting trained model to ONNX format...
    python convert_to_onnx.py
    if not errorlevel 1 (
        echo Copying ONNX files to inference directory...
        copy onnx_models\transformer_tidal_v1.onnx ..\inference\
        copy onnx_models\model_metadata.json ..\inference\
    )
    cd ..
) else if "%choice%"=="6" (
    echo ========== Local Testing ==========
    cd testing
    echo Starting test server...
    echo Open http://localhost:8000 in your browser to test
    python server.py
    cd ..
) else if "%choice%"=="7" (
    echo ========== Firebase Deployment ==========
    call deploy-transformer-v1.bat
) else if "%choice%"=="8" (
    echo ========== Complete Pipeline ==========
    echo.
    echo Step 1: Data Preparation
    cd data-preparation
    python fetch_firebase_data.py
    if errorlevel 1 goto :error
    python create_training_data.py
    if errorlevel 1 goto :error
    cd ..
    echo.
    
    echo Step 2: Training
    cd training
    echo Warning: This may take several hours!
    set /p confirm=Continue with training? (y/n): 
    if /i "%confirm%"=="y" (
        python train_transformer.py
        if errorlevel 1 goto :error
    ) else (
        echo Skipping training. Make sure you have a trained model before continuing.
    )
    cd ..
    echo.
    
    echo Step 3: ONNX Conversion
    cd training
    python convert_to_onnx.py
    if errorlevel 1 goto :error
    copy onnx_models\transformer_tidal_v1.onnx ..\inference\ >nul 2>&1
    copy onnx_models\model_metadata.json ..\inference\ >nul 2>&1
    cd ..
    echo.
    
    echo Step 4: Local Testing
    echo Starting test server for validation...
    cd testing
    start /min python server.py
    timeout /t 3 >nul
    cd ..
    echo.
    
    echo Step 5: Firebase Deployment
    call deploy-transformer-v1.bat
    
    echo.
    echo ========== Pipeline Complete ==========
    echo Your Transformer v1 model is now ready for production!
    
) else if "%choice%"=="9" (
    echo Exiting setup.
    goto :end
) else (
    echo Invalid choice. Please run again.
    goto :end
)

goto :success

:error
echo.
echo ========== ERROR ==========
echo Pipeline failed at current step.
echo Check the error messages above for details.
goto :end

:success
echo.
echo ========== SUCCESS ==========
echo Operation completed successfully!

:end
echo.
pause
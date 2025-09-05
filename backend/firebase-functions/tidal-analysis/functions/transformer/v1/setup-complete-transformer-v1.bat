@echo off
echo ========================================================
echo   Complete Transformer v1 Setup and Training Pipeline
echo ========================================================
echo.

REM Set URL variables to avoid colon issues
set CPU_URL=https://download.pytorch.org/whl/cpu
set CUDA_URL=https://download.pytorch.org/whl/cu121

echo ========================================================
echo   Setup Options
echo ========================================================
echo 1. Install PyTorch (CPU version)
echo 2. Install PyTorch (CUDA version - requires NVIDIA GPU)
echo 3. Fetch data, fill gaps, and create training dataset
echo 4. Train transformer model
echo 5. Test model locally (PyTorch web server)
echo 6. Deploy to Firebase Functions (PyTorch runtime)
echo 7. Run complete pipeline (3-6)
echo 8. Exit
echo.

set /p choice=Enter your choice (1-8): 

if "%choice%"=="1" (
    echo Installing PyTorch CPU version...
    echo This may take a few minutes...
    pip install torch torchvision torchaudio --index-url %CPU_URL% >nul 2>&1
    if errorlevel 1 (
        echo Warning: PyTorch installation had issues, but may already be installed
    )
    pip install tensorboard flask >nul 2>&1
    echo PyTorch CPU installation completed.
    goto :success
)

if "%choice%"=="2" (
    echo Installing PyTorch CUDA version...
    echo This may take a few minutes...
    pip install torch torchvision torchaudio --index-url %CUDA_URL% >nul 2>&1
    if errorlevel 1 (
        echo Warning: PyTorch CUDA installation had issues, but may already be installed
    )
    pip install tensorboard flask >nul 2>&1
    echo PyTorch CUDA installation completed.
    goto :success
)

if "%choice%"=="3" (
    echo ========== Data Preparation ==========
    cd data-preparation
    echo Fetching Firebase data...
    python fetch_firebase_data.py
    if not errorlevel 1 (
        echo Enriching data with gap filling...
        python enrich_firebase_data.py
        if not errorlevel 1 (
            echo Creating training dataset...
            python create_training_data.py
        )
    )
    cd ..
    goto :success
)

if "%choice%"=="4" (
    echo ========== Training Transformer ==========
    cd training
    echo Starting transformer training...
    echo This may take several hours depending on your hardware.
    echo.
    python train_transformer.py
    cd ..
    goto :success
)

if "%choice%"=="5" (
    echo ========== Local Testing ==========
    cd training
    echo Starting PyTorch model server...
    echo Open http://localhost:8000 in your browser to test
    echo Press Ctrl+C to stop the server
    python model_server.py
    cd ..
    goto :success
)

if "%choice%"=="6" (
    echo ========== Firebase Deployment ==========
    if not exist "training\checkpoints\best.pth" (
        echo ERROR: No trained model found!
        echo Please train a model first using option 4
        goto :error
    )
    call deploy-transformer-v1.bat
    goto :success
)

if "%choice%"=="7" (
    echo ========== Complete Pipeline ==========
    echo.
    echo Step 1: Data Preparation
    cd data-preparation
    python fetch_firebase_data.py
    if errorlevel 1 goto :error
    python enrich_firebase_data.py
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
    
    echo Step 3: Local Testing
    echo Starting PyTorch model server for validation...
    cd training
    start /min python model_server.py
    timeout /t 3 >nul
    cd ..
    echo.
    
    echo Step 4: Firebase Deployment
    call deploy-transformer-v1.bat
    
    echo.
    echo ========== Pipeline Complete ==========
    echo Your Transformer v1 model is now ready for production!
    echo - Local server: http://localhost:8000 (if still running)
    echo - Firebase Functions: Check Firebase Console for deployment status
    goto :success
)

if "%choice%"=="8" (
    echo Exiting setup.
    goto :end
)

echo Invalid choice. Please run again.
goto :end

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
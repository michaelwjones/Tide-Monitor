@echo off
echo LSTM v1 Complete Setup - Interactive Menu
echo ========================================
echo.

:MENU
echo Choose an option:
echo [0] Check GPU Support and Performance
echo [1] Run Step 1: Install PyTorch and dependencies
echo [2] Run Step 2: Fetch training data from Firebase  
echo [3] Run Step 3: Create training sequences
echo [4] Run Step 4: Train LSTM model
echo [5] Run Step 5: Convert model to ONNX
echo [6] Run Step 6: Copy files for Firebase deployment
echo [A] Run ALL steps automatically (1-6)
echo [Q] Quit
echo.
set /p choice=Enter your choice: 

if /i "%choice%"=="0" goto STEP0
if /i "%choice%"=="1" goto STEP1
if /i "%choice%"=="2" goto STEP2
if /i "%choice%"=="3" goto STEP3
if /i "%choice%"=="4" goto STEP4
if /i "%choice%"=="5" goto STEP5
if /i "%choice%"=="6" goto STEP6
if /i "%choice%"=="a" goto RUNALL
if /i "%choice%"=="q" goto END

echo Invalid choice. Please try again.
echo.
goto MENU

:STEP0
echo.
echo Step 0: Checking GPU Support and Performance...
echo ================================================
python check_gpu.py
if errorlevel 1 (
    echo GPU check completed with warnings - you can still proceed with CPU training
    pause
    goto MENU
)
echo GPU check completed successfully!
pause
goto MENU

:STEP1
echo.
echo Step 1: Installing PyTorch and dependencies...
echo =============================================
call install-pytorch-automatic.bat
if errorlevel 1 (
    echo Step 1 failed!
    pause
    goto MENU
)
echo Step 1 completed successfully!
pause
goto MENU

:STEP2
echo.
echo Step 2: Fetching training data from Firebase...
echo ================================================
cd data-preparation
python fetch_firebase_data.py
if errorlevel 1 (
    echo Step 2 failed!
    cd ..
    pause
    goto MENU
)
cd ..
echo Step 2 completed successfully!
pause
goto MENU

:STEP3
echo.
echo Step 3: Creating training sequences...
echo ======================================
cd data-preparation
python create_training_data.py
if errorlevel 1 (
    echo Step 3 failed!
    cd ..
    pause
    goto MENU
)
cd ..
echo Step 3 completed successfully!
pause
goto MENU

:STEP4
echo.
echo Step 4: Training LSTM model...
echo ===============================
cd training
python train_lstm.py
if errorlevel 1 (
    echo Step 4 failed!
    cd ..
    pause
    goto MENU
)
cd ..
echo Step 4 completed successfully!
pause
goto MENU

:STEP5
echo.
echo Step 5: Converting model to ONNX...
echo ===================================
cd training
python convert_to_onnx.py
if errorlevel 1 (
    echo Step 5 failed!
    cd ..
    pause
    goto MENU
)
cd ..
echo Step 5 completed successfully!
pause
goto MENU

:STEP6
echo.
echo Step 6: Copying files for Firebase deployment...
echo ================================================
if not exist "training\inference" (
    echo Error: inference folder not found. Run Step 5 first.
    pause
    goto MENU
)

REM Copy model files to inference directory
copy "training\inference\tidal_lstm.onnx" "inference\" >nul
if errorlevel 1 (
    echo Failed to copy ONNX model
    pause
    goto MENU
)

copy "training\inference\model_metadata.json" "inference\" >nul
if errorlevel 1 (
    echo Failed to copy model metadata
    pause
    goto MENU
)

copy "training\inference\normalization_params.json" "inference\" >nul
if errorlevel 1 (
    echo Failed to copy normalization parameters
    pause
    goto MENU
)

echo Step 6 completed successfully!
echo Files copied to inference/ for Firebase deployment
pause
goto MENU

:RUNALL
echo.
echo Running ALL steps automatically...
echo =================================
call :STEP1
call :STEP2  
call :STEP3
call :STEP4
call :STEP5
call :STEP6
echo.
echo ALL STEPS COMPLETED!
echo Your LSTM v1 model is ready for Firebase deployment.
pause
goto MENU

:END
echo.
echo Setup complete! 
echo ===============
echo.
echo If all steps were successful, you can now:
echo 1. Deploy Firebase Functions: firebase deploy --only functions --source inference
echo 2. Check debug dashboard for 24-hour forecasts
echo.
pause
exit
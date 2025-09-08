@echo off
set ORIGINAL_DIR=%CD%
cd /d "%~dp0"
echo Deploying Transformer v1 PyTorch Tidal Analysis Function...
echo This function performs transformer-based tidal analysis every 5 minutes.
echo.

echo Checking trained PyTorch model...
if not exist "training\checkpoints\best.pth" (
    echo ERROR: Trained PyTorch model not found!
    echo Please train the transformer model first using train_transformer.py
    echo Expected: training\checkpoints\best.pth
    echo.
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b 1
)
echo [OK] Trained PyTorch model found: training\checkpoints\best.pth

echo.
echo Copying model checkpoint to inference directory...
copy "training\checkpoints\best.pth" "inference\best.pth" > nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Model checkpoint copied to inference directory
) else (
    echo [ERROR] Failed to copy model checkpoint
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b 1
)

echo.
echo Checking current configuration...
if exist "inference\.env" (
    echo Found .env file:
    type "inference\.env"
) else (
    echo No .env file found - analysis will be enabled by default
)

echo.
echo Deploying Transformer v1 function to Firebase...
cd inference

echo Setting up Python virtual environment...
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate.bat
echo Installing Python dependencies...
call pip install -r requirements.txt

echo Deploying to Firebase (Python runtime)...
firebase deploy --only functions:run_transformer_v1_analysis
set DEPLOY_RESULT=%ERRORLEVEL%

cd /d "%ORIGINAL_DIR%"
if %DEPLOY_RESULT% EQU 0 (
    echo.
    echo [OK] Transformer v1 tidal analysis deployed successfully!
    echo NOTE: Function runs every 5 minutes and stores results in /tidal-analysis/transformer-v1
    echo INFO: Check Firebase Console for execution logs
    echo.
    echo Analysis results will appear at:
    echo https://tide-monitor-boron-default-rtdb.firebaseio.com/tidal-analysis/transformer-v1.json
) else (
    echo.
    echo [ERROR] Deployment failed!
    echo Check the error messages above
)

echo.
pause
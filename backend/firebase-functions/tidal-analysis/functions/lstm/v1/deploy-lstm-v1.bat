@echo off
echo Deploying LSTM v1 to Firebase Functions...
echo ===========================================
echo.

REM Check if we're in the right directory
if not exist "inference\index.js" (
    echo Error: inference/index.js not found
    echo Make sure you're in the LSTM v1 directory and have completed setup
    pause
    exit /b 1
)

if not exist "inference\tidal_lstm.onnx" (
    echo Error: ONNX model not found
    echo Run setup-complete-lstm-v1.bat first to train and convert the model
    pause
    exit /b 1
)

echo Checking Firebase CLI...
firebase --version >nul 2>&1
if errorlevel 1 (
    echo Error: Firebase CLI not installed
    echo Install with: npm install -g firebase-tools
    pause
    exit /b 1
)

echo Firebase CLI found. Starting deployment...
echo.

REM Change to inference directory
cd inference

REM Install dependencies if needed
if not exist "node_modules" (
    echo Installing Node.js dependencies...
    npm install
    if errorlevel 1 (
        echo Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Deploying Firebase Functions...
firebase deploy --only functions:runLSTMv1Prediction

if errorlevel 1 (
    echo Deployment failed!
    pause
    exit /b 1
)

cd ..

echo.
echo ========================================
echo LSTM v1 Deployment Successful!
echo ========================================
echo.
echo The LSTM v1 function is now active and will:
echo - Run every 6 hours to generate 24-hour forecasts
echo - Store predictions in /tidal-analysis/lstm-v1-forecasts/
echo - Use the last 72 hours of data for iterative prediction
echo.
echo Check the Firebase Functions logs:
echo   firebase functions:log
echo.
echo View predictions on the debug dashboard:
echo   debug/index.html
echo.
pause
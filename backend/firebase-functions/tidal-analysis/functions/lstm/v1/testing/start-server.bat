@echo off
echo ========================================
echo    LSTM v1 Model Testing Server
echo ========================================
echo.

REM Change to the testing directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found
    echo    Please install Python and ensure it's in your PATH
    echo    Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if the trained model exists
if not exist "..\training\trained_models\best_model.pth" (
    echo ❌ Error: Trained model not found
    echo    Expected: ..\training\trained_models\best_model.pth
    echo    Please train the model first by running train_lstm.py
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "server.py" (
    echo ❌ Error: server.py not found in current directory
    pause
    exit /b 1
)

if not exist "index.html" (
    echo ❌ Error: index.html not found in current directory
    pause
    exit /b 1
)

echo ✅ Python found
echo ✅ Trained model found
echo ✅ Server files found
echo.
echo 🚀 Starting LSTM Testing Server...
echo 📍 Server will be available at: http://localhost:8000
echo ⏹️  Press Ctrl+C in the server window to stop
echo.
echo Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul

REM Start browser (try different browsers)
start "" "http://localhost:8000" 2>nul || (
    echo ⚠️  Could not open browser automatically
    echo    Please open: http://localhost:8000
)

echo.
echo Starting Python server...
echo ========================================

REM Start the Python server
python server.py

REM If we get here, the server stopped
echo.
echo 👋 Server stopped
pause
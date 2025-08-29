@echo off
echo ==========================================
echo   Transformer v1 Testing Server Launcher
echo ==========================================
echo.

:: Check if we're in the right directory
if not exist "server.py" (
    echo Error: server.py not found in current directory
    echo Please run this batch file from the testing directory
    pause
    exit /b 1
)

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or later
    pause
    exit /b 1
)

echo Starting Transformer v1 testing server...
echo.

:: Start the server
python server.py

:: If we get here, the server stopped
echo.
echo Server stopped. Press any key to exit.
pause
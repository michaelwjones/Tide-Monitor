@echo off
echo Starting Transformer Tidal Prediction Server...
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -q -r requirements.txt

REM Start the server
echo.
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop
echo.
python model_server.py

pause
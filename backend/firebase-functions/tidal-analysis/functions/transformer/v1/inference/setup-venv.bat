@echo off
echo Setting up Python virtual environment for Firebase Functions...

REM Activate virtual environment and install dependencies
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo Virtual environment setup complete!
pause
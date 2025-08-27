@echo off
echo Manual PyTorch Installation Options
echo ====================================
echo.
echo Choose your preferred installation method:
echo.
echo [1] Default PyTorch (includes CUDA if drivers available)
echo [2] CUDA 11.8 optimized
echo [3] CUDA 12.1 optimized  
echo [4] CPU-only version
echo [Q] Quit
echo.
set /p choice=Enter your choice: 

if /i "%choice%"=="1" goto DEFAULT
if /i "%choice%"=="2" goto CUDA118
if /i "%choice%"=="3" goto CUDA121
if /i "%choice%"=="4" goto CPU
if /i "%choice%"=="q" goto END

echo Invalid choice. Please try again.
pause
goto MENU

:DEFAULT
echo Installing default PyTorch...
pip install torch numpy pandas requests onnx onnxruntime matplotlib tqdm
goto VERIFY

:CUDA118
echo Installing PyTorch with CUDA 11.8...
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas requests onnx onnxruntime matplotlib tqdm
goto VERIFY

:CUDA121
echo Installing PyTorch with CUDA 12.1...
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas requests onnx onnxruntime matplotlib tqdm
goto VERIFY

:CPU
echo Installing CPU-only PyTorch...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas requests onnx onnxruntime matplotlib tqdm
goto VERIFY

:VERIFY
echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo Installation complete!
echo You can now run: python training/train_lstm.py
pause
goto END

:END
exit
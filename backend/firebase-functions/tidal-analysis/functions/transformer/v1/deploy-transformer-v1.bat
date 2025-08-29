@echo off
echo ================================================
echo   Transformer v1 Firebase Functions Deployment
echo ================================================
echo.

:: Check if we're in the right directory
if not exist "inference\index.js" (
    echo Error: inference/index.js not found
    echo Please run this from the transformer/v1 directory
    pause
    exit /b 1
)

:: Check for trained model files
if not exist "training\checkpoints\best.pth" (
    echo Warning: No trained model found at training/checkpoints/best.pth
    echo You may need to train the model first
    echo.
)

:: Check for ONNX model in inference directory
if not exist "inference\transformer_tidal_v1.onnx" (
    echo Warning: ONNX model not found at inference/transformer_tidal_v1.onnx
    echo You may need to run convert_to_onnx.py first
    echo.
)

:: Check for model metadata
if not exist "inference\model_metadata.json" (
    echo Warning: Model metadata not found at inference/model_metadata.json
    echo You may need to run convert_to_onnx.py first
    echo.
)

echo Current directory contents:
dir /b inference\

echo.
echo Deployment Options:
echo 1. Deploy Transformer prediction function (runTransformerv1Prediction)
echo 2. Deploy test function only (testTransformerv1Prediction)
echo 3. Deploy both functions
echo 4. Cancel
echo.

set /p choice=Enter choice (1-4): 

if "%choice%"=="1" (
    echo.
    echo Deploying Transformer v1 prediction function...
    cd inference
    firebase deploy --only functions:runTransformerv1Prediction
    cd ..
) else if "%choice%"=="2" (
    echo.
    echo Deploying Transformer v1 test function...
    cd inference
    firebase deploy --only functions:testTransformerv1Prediction
    cd ..
) else if "%choice%"=="3" (
    echo.
    echo Deploying both Transformer v1 functions...
    cd inference
    firebase deploy --only functions:runTransformerv1Prediction,testTransformerv1Prediction
    cd ..
) else if "%choice%"=="4" (
    echo Deployment cancelled.
    goto :end
) else (
    echo Invalid choice. Please run again.
    goto :end
)

echo.
echo ================================================
echo   Deployment completed!
echo ================================================
echo.
echo Next steps:
echo 1. Check Firebase Console for function status
echo 2. Monitor logs: firebase functions:log
echo 3. Test manually: firebase functions:shell
echo.

:end
pause
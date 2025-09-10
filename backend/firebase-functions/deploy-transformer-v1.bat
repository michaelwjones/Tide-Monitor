@echo off
set ORIGINAL_DIR=%CD%
echo Deploying Transformer v1 Tidal Analysis Function...
echo This function performs transformer-based tidal analysis every 5 minutes.
echo.

echo Checking trained PyTorch model...
if not exist "tidal-analysis\functions\transformer\v1\cloud\training\single-runs\best_seq2seq_single_run.pth" (
    echo ERROR: Trained PyTorch model not found!
    echo Please train the transformer model first using cloud training single-runs
    echo Expected: tidal-analysis\functions\transformer\v1\cloud\training\single-runs\best_seq2seq_single_run.pth
    echo.
    pause
    exit /b 1
)
echo [OK] Trained PyTorch model found

echo.
echo Copying model checkpoint to tidal-analysis root directory...
copy "tidal-analysis\functions\transformer\v1\cloud\training\single-runs\best_seq2seq_single_run.pth" "tidal-analysis\best.pth" > nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Model checkpoint copied to tidal-analysis root
) else (
    echo [ERROR] Failed to copy model checkpoint
    pause
    exit /b 1
)

echo.
echo Deploying all tidal-analysis functions (including transformer v1)...
firebase deploy --only functions --project tide-monitor-boron
set DEPLOY_RESULT=%ERRORLEVEL%

if %DEPLOY_RESULT% EQU 0 (
    echo.
    echo [OK] Transformer v1 tidal analysis deployed successfully!
    echo NOTE: Function runs every 5 minutes and stores results in /tidal-analysis/transformer-v1-forecast
    echo INFO: Check Firebase Console for execution logs
    echo.
    echo Analysis results will appear at:
    echo https://tide-monitor-boron-default-rtdb.firebaseio.com/tidal-analysis/transformer-v1-forecast.json
) else (
    echo.
    echo [ERROR] Deployment failed!
    echo Check the error messages above
)

echo.
pause
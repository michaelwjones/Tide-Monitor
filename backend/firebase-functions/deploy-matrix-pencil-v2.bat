@echo off
cd /d "%~dp0"
echo Deploying Matrix Pencil v2 Tidal Analysis Function...
echo This function performs ENHANCED tidal harmonic analysis with improved accuracy.
echo.
echo 🚀 ENHANCED FEATURES: 2/3 data length, 20 SVD components, 0.1%% threshold, up to 16 frequencies
echo ⚡ HIGHER COST: ~$10-25/month when enabled (vs ~$5-15 for v1)
echo ⚠️  WARNING: The analysis is DISABLED by default to prevent unexpected charges
echo.

echo Checking current configuration...
if exist "tidal-analysis\functions\matrix-pencil\v2\.env" (
    echo Found .env file:
    type "tidal-analysis\functions\matrix-pencil\v2\.env"
) else (
    echo No .env file found - analysis will be disabled by default
)

echo.
echo Current options:
echo [1] Deploy with analysis ENABLED (starts charging immediately - enhanced cost)
echo [2] Deploy with analysis DISABLED (no charges, deploy only)  
echo [3] Just check current status and exit
echo [4] Cancel deployment
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo ⚠️  ENABLING Matrix Pencil v2 analysis - this will start costing MORE money than v1!
    echo TIDAL_ANALYSIS_ENABLED=true > "tidal-analysis\functions\matrix-pencil\v2\.env"
    echo ✅ Analysis enabled. Deploying enhanced function...
    cd tidal-analysis\functions\matrix-pencil\v2
    call firebase deploy --only functions
    cd ..\..\..
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✅ Matrix Pencil v2 analysis deployed and ENABLED!
        echo 💰 Function will run every 5 minutes and incur ENHANCED costs (~$10-25/month)
        echo 📊 Expected: 3-6 tidal components (vs 1-2 in v1)
        echo 🔧 Check Firebase Console for execution logs
        echo 🛑 DISABLE: Set TIDAL_ANALYSIS_ENABLED=false in tidal-analysis\functions\matrix-pencil\v2\.env
    )
) else if "%choice%"=="2" (
    echo.
    echo Ensuring analysis is disabled...
    echo TIDAL_ANALYSIS_ENABLED=false > "tidal-analysis\functions\matrix-pencil\v2\.env"
    echo ✅ Analysis disabled. Deploying enhanced function...
    cd tidal-analysis\functions\matrix-pencil\v2
    call firebase deploy --only functions
    cd ..\..\..
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✅ Matrix Pencil v2 analysis deployed but DISABLED!
        echo 💰 No charges will occur until you enable it
        echo 🚀 ENABLE: Set TIDAL_ANALYSIS_ENABLED=true in tidal-analysis\functions\matrix-pencil\v2\.env
        echo 📈 Enhanced features: 2/3 data length, 0.1%% threshold, up to 16 components
    )
) else if "%choice%"=="3" (
    echo.
    echo Current Matrix Pencil v2 configuration status:
    if exist "tidal-analysis\functions\matrix-pencil\v2\.env" (
        echo .env file contents:
        type "tidal-analysis\functions\matrix-pencil\v2\.env"
    ) else (
        echo No .env file exists - analysis disabled by default
    )
    echo.
    echo Enhanced v2 Features:
    echo - Pencil Parameter L: 2/3 data length (vs 1/3 in v1)
    echo - SVD Rank Limit: 20 components (vs 5 in v1)  
    echo - Threshold: 0.1%% sensitivity (vs 1%% in v1)
    echo - Max Components: 16 (vs 8 in v1)
    echo - Expected Cost: ~$10-25/month when enabled
) else if "%choice%"=="4" (
    echo.
    echo Deployment cancelled.
) else (
    echo.
    echo Invalid choice. Deployment cancelled.
)

echo.
pause
@echo off
cd /d "%~dp0"
echo Deploying Tidal Analysis Function (Matrix Pencil v1)...
echo This function performs advanced tidal harmonic analysis every 5 minutes.
echo.
echo WARNING: This function costs ~$5-15/month when enabled
echo WARNING: The analysis is DISABLED by default to prevent unexpected charges
echo.

echo Checking current configuration...
if exist "tidal-analysis\.env" (
    echo Found .env file:
    type "tidal-analysis\.env"
) else (
    echo No .env file found - analysis will be disabled by default
)

echo.
echo Current options:
echo [1] Deploy with analysis ENABLED (starts charging immediately)
echo [2] Deploy with analysis DISABLED (no charges, deploy only)  
echo [3] Just check current status and exit
echo [4] Cancel deployment
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo WARNING: ENABLING tidal analysis - this will start costing money!
    echo TIDAL_ANALYSIS_ENABLED=true > "tidal-analysis\.env"
    echo [OK] Analysis enabled. Deploying function...
    call firebase deploy --only functions:tidal-analysis
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo [OK] Tidal analysis deployed and ENABLED!
        echo NOTE: Function will run every 5 minutes and incur costs
        echo INFO: Check Firebase Console for execution logs
        echo DISABLE: Set TIDAL_ANALYSIS_ENABLED=false in tidal-analysis\.env
    )
) else if "%choice%"=="2" (
    echo.
    echo Ensuring analysis is disabled...
    echo TIDAL_ANALYSIS_ENABLED=false > "tidal-analysis\.env"
    echo [OK] Analysis disabled. Deploying function...
    call firebase deploy --only functions:tidal-analysis
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo [OK] Tidal analysis deployed but DISABLED!
        echo NOTE: No charges will occur until you enable it
        echo ENABLE: Set TIDAL_ANALYSIS_ENABLED=true in tidal-analysis\.env
    )
) else if "%choice%"=="3" (
    echo.
    echo Current configuration status:
    if exist "tidal-analysis\.env" (
        echo .env file contents:
        type "tidal-analysis\.env"
    ) else (
        echo No .env file exists - analysis disabled by default
    )
    echo.
) else if "%choice%"=="4" (
    echo.
    echo Deployment cancelled.
) else (
    echo.
    echo Invalid choice. Deployment cancelled.
)

echo.
pause
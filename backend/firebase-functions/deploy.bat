@echo off
echo Firebase Functions Deployment Menu
echo ===================================
echo.
echo Available functions:
echo [1] Deploy Tide Enrichment only (NOAA data, always needed)
echo [2] Deploy Tidal Analysis only (Matrix Pencil, costs money when enabled)
echo [3] Deploy BOTH functions
echo [4] Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    call deploy-enrichment.bat
) else if "%choice%"=="2" (
    echo.
    call deploy-tidal-analysis.bat
) else if "%choice%"=="3" (
    echo.
    echo Deploying BOTH functions...
    echo.
    echo 1/2: Deploying Tide Enrichment...
    call firebase deploy --only functions:tide-enrichment
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Tide enrichment deployed successfully!
        echo.
        echo 2/2: Starting Tidal Analysis deployment...
        call deploy-tidal-analysis.bat
    ) else (
        echo [ERROR] Tide enrichment deployment failed. Stopping.
        pause
        exit /b 1
    )
) else if "%choice%"=="4" (
    echo.
    echo Exiting...
    exit /b 0
) else (
    echo.
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)
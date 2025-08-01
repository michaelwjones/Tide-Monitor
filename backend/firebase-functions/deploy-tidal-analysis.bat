@echo off
echo Deploying Tidal Analysis Function (Matrix Pencil v1)...
echo This function performs advanced tidal harmonic analysis every 5 minutes.
echo.
echo ⚠️  COST WARNING: This function costs ~$5-15/month when enabled
echo ⚠️  The analysis is DISABLED by default to prevent unexpected charges
echo.

echo Checking current configuration...
firebase functions:config:get tidal.analysis.enabled 2>nul

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
    echo ⚠️  ENABLING tidal analysis - this will start costing money!
    firebase functions:config:set tidal.analysis.enabled=true
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Analysis enabled. Deploying function...
        firebase deploy --only functions --source tidal-analysis
        if %ERRORLEVEL% EQU 0 (
            echo.
            echo ✅ Tidal analysis deployed and ENABLED!
            echo 💰 Function will run every 5 minutes and incur costs
            echo 📊 Check Firebase Console for execution logs
            echo 🛑 To disable: firebase functions:config:set tidal.analysis.enabled=false
        )
    )
) else if "%choice%"=="2" (
    echo.
    echo Ensuring analysis is disabled...
    firebase functions:config:set tidal.analysis.enabled=false
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Analysis disabled. Deploying function...
        firebase deploy --only functions --source tidal-analysis
        if %ERRORLEVEL% EQU 0 (
            echo.
            echo ✅ Tidal analysis deployed but DISABLED!
            echo 💰 No charges will occur until you enable it
            echo 🚀 To enable: firebase functions:config:set tidal.analysis.enabled=true
        )
    )
) else if "%choice%"=="3" (
    echo.
    echo Current configuration status:
    firebase functions:config:get
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
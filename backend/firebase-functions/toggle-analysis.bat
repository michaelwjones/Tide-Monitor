@echo off
echo Tidal Analysis Toggle - Quick Enable/Disable
echo ================================================
echo This script only changes the configuration setting.
echo No deployment is performed - existing function remains as-is.
echo.

echo Checking current configuration...
firebase functions:config:get tidal.analysis.enabled 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Configuration not found - analysis is disabled by default
    set current_status=disabled
) else (
    for /f "tokens=*" %%i in ('firebase functions:config:get tidal.analysis.enabled 2^>nul') do set current_value=%%i
    if "!current_value!"=="true" (
        set current_status=enabled
    ) else (
        set current_status=disabled
    )
)

echo.
echo Current status: Analysis is %current_status%
echo.
echo Options:
echo [1] Enable analysis (starts charging ~$5-15/month)
echo [2] Disable analysis (stops charges)
echo [3] Check status and exit
echo [4] Cancel
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo ⚠️  ENABLING tidal analysis...
    firebase functions:config:set tidal.analysis.enabled=true
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Analysis ENABLED!
        echo 💰 Function will run every 5 minutes and incur costs
        echo 📊 Check Firebase Console for execution logs
        echo 🛑 To disable again, run this script and choose option 2
    ) else (
        echo ❌ Failed to enable analysis
    )
) else if "%choice%"=="2" (
    echo.
    echo 🛑 DISABLING tidal analysis...
    firebase functions:config:set tidal.analysis.enabled=false
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Analysis DISABLED!
        echo 💰 No charges will occur
        echo 🚀 To enable again, run this script and choose option 1
    ) else (
        echo ❌ Failed to disable analysis
    )
) else if "%choice%"=="3" (
    echo.
    echo Current configuration:
    firebase functions:config:get
    echo.
    echo Note: Changes take effect immediately for deployed functions
) else if "%choice%"=="4" (
    echo.
    echo Operation cancelled.
) else (
    echo.
    echo Invalid choice. Operation cancelled.
)

echo.
echo Note: This only changes the configuration setting.
echo The function must already be deployed for changes to take effect.
echo Use deploy-tidal-analysis.bat if you need to deploy the function.
echo.
pause
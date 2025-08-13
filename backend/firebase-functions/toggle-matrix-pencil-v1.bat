@echo off
cd /d "%~dp0"
echo Matrix Pencil v1 Analysis Toggle - Quick Enable/Disable
echo ================================================
echo This script changes the .env configuration setting.
echo No deployment is performed - existing function remains as-is.
echo.

echo Checking current configuration...
if exist "tidal-analysis\functions\matrix-pencil\v1\.env" (
    echo Found .env file:
    type "tidal-analysis\functions\matrix-pencil\v1\.env"
    for /f "tokens=2 delims==" %%i in ('findstr "TIDAL_ANALYSIS_ENABLED" "tidal-analysis\functions\matrix-pencil\v1\.env" 2^>nul') do set current_value=%%i
    if "!current_value!"=="true" (
        set current_status=enabled
    ) else (
        set current_status=disabled
    )
) else (
    echo No .env file found - analysis is disabled by default
    set current_status=disabled
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
    echo WARNING: ENABLING tidal analysis...
    echo TIDAL_ANALYSIS_ENABLED=true > "tidal-analysis\functions\matrix-pencil\v1\.env"
    echo [OK] Analysis ENABLED!
    echo NOTE: Function will run every 5 minutes and incur costs
    echo INFO: Check Firebase Console for execution logs
    echo DISABLE: To disable again, run this script and choose option 2
    echo.
    echo IMPORTANT: You must redeploy for changes to take effect:
    echo    Run: firebase deploy --only functions --source tidal-analysis/functions/matrix-pencil/v1
    goto :end
) else if "%choice%"=="2" (
    echo.
    echo DISABLING tidal analysis...
    echo TIDAL_ANALYSIS_ENABLED=false > "tidal-analysis\functions\matrix-pencil\v1\.env"
    echo [OK] Analysis DISABLED!
    echo NOTE: No charges will occur
    echo ENABLE: To enable again, run this script and choose option 1
    echo.
    echo IMPORTANT: You must redeploy for changes to take effect:
    echo    Run: firebase deploy --only functions --source tidal-analysis/functions/matrix-pencil/v1
    goto :end
) else if "%choice%"=="3" (
    echo.
    echo Current configuration:
    if exist "tidal-analysis\functions\matrix-pencil\v1\.env" (
        echo .env file contents:
        type "tidal-analysis\functions\matrix-pencil\v1\.env"
    ) else (
        echo No .env file exists - analysis disabled by default
    )
    echo.
    echo Note: Changes take effect after redeployment
    goto :end
) else if "%choice%"=="4" (
    echo.
    echo Operation cancelled.
    goto :end
) else (
    echo.
    echo Invalid choice. Operation cancelled.
    goto :end
)

:end
echo.
echo Note: This only changes the .env configuration file.
echo The function must be redeployed for changes to take effect.
echo Use deploy-matrix-pencil-v1.bat to deploy with the new configuration.
echo.
pause
@echo off
cd /d "%~dp0"
echo Matrix Pencil v2 Analysis Toggle - Enhanced Quick Enable/Disable
echo ================================================================
echo This script changes the .env configuration setting for v2.
echo No deployment is performed - existing function remains as-is.
echo.
echo 🚀 ENHANCED v2 FEATURES: 2/3 data length, 20 SVD components, 0.1%% threshold
echo ⚡ HIGHER COST: ~$10-25/month when enabled (vs ~$5-15 for v1)
echo.

echo Checking current v2 configuration...
if exist "tidal-analysis\functions\matrix-pencil\v2\.env" (
    echo Found .env file:
    type "tidal-analysis\functions\matrix-pencil\v2\.env"
    for /f "tokens=2 delims==" %%i in ('findstr "TIDAL_ANALYSIS_ENABLED" "tidal-analysis\functions\matrix-pencil\v2\.env" 2^>nul') do set current_value=%%i
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
echo Current v2 status: Analysis is %current_status%
echo.
echo Options:
echo [1] Enable v2 analysis (starts charging ~$10-25/month - ENHANCED COST)
echo [2] Disable v2 analysis (stops charges)
echo [3] Check status and exit
echo [4] Cancel
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo ⚠️  ENABLING Matrix Pencil v2 analysis...
    echo TIDAL_ANALYSIS_ENABLED=true > "tidal-analysis\functions\matrix-pencil\v2\.env"
    echo ✅ Matrix Pencil v2 analysis ENABLED!
    echo 💰 Function will run every 5 minutes and incur ENHANCED costs (~$10-25/month)
    echo 📊 Expected improvements: 3-6 components instead of 1-2
    echo 🔧 Enhanced accuracy with 2/3 data length and 0.1%% threshold
    echo 📈 Monitor Firebase Console for improved results
    echo.
    echo 🛑 To disable: run this script again and choose option 2
    goto :end
) else if "%choice%"=="2" (
    echo.
    echo Disabling Matrix Pencil v2 analysis...
    echo TIDAL_ANALYSIS_ENABLED=false > "tidal-analysis\functions\matrix-pencil\v2\.env"
    echo ✅ Matrix Pencil v2 analysis DISABLED!
    echo 💰 No more charges will occur
    echo 🚀 To re-enable: run this script again and choose option 1
    echo.
    echo Enhanced v2 features will be available when re-enabled:
    echo - Better frequency resolution
    echo - More tidal components detected
    echo - Improved accuracy for closely spaced constituents
    goto :end
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
    echo Enhanced v2 Specifications:
    echo - Pencil Parameter L: 2/3 of data length (enhanced from 1/3 in v1)
    echo - SVD Rank Limit: 20 singular values (enhanced from 5 in v1)
    echo - Detection Threshold: 0.1%% sensitivity (enhanced from 1%% in v1)  
    echo - Maximum Components: 16 frequencies (enhanced from 8 in v1)
    echo - Precision: 1e-8 tolerance (enhanced from 1e-6 in v1)
    echo - Expected Cost: ~$10-25/month when enabled
    echo.
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
pause
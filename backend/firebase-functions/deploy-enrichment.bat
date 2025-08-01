@echo off
echo Deploying Tide Enrichment Function...
echo This function enriches sensor data with NOAA environmental conditions.
echo.

firebase deploy --only functions --source tide-enrichment

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Tide enrichment deployed successfully!
    echo This function will automatically enrich new sensor readings with NOAA data.
) else (
    echo.
    echo ❌ Deployment failed. Check the error above.
)

pause
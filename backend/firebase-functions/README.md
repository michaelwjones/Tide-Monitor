# Firebase Cloud Functions

This directory contains Firebase Cloud Functions for the Tide Monitor project:

1. **Tide Enrichment** (`tide-enrichment/`) - Enriches sensor data with NOAA environmental conditions
2. **Tidal Analysis** (`tidal-analysis/`) - Advanced harmonic analysis using Matrix Pencil v1 methodology

## Overview

### Tide Enrichment Function
The `enrichTideData` function automatically triggers when new readings are written to the Firebase `/readings/` path and enriches each entry with real-time wind and water level data from NOAA station 8656483 (Duke Marine Lab, Beaufort, NC).

### Tidal Analysis Function  
The `runTidalAnalysis` function runs every 5 minutes via Cloud Scheduler to perform advanced tidal harmonic analysis using the Matrix Pencil v1 method. Results are stored in `/tidal-analysis/` for use by the debug dashboard.

## Data Enrichment

### Added Fields (from NOAA API)
- **ws**: Wind speed (m/s) 
- **wd**: Wind direction (degrees)
- **gs**: Gust speed (m/s)
- **wm**: Water level in feet (MLLW datum)

### Original Fields (from Particle device)
- **t**: Timestamp (ISO8601 format)
- **w**: Water level in mm (sensor average)
- **hp**: Wave height percentile method (mm)
- **he**: Wave height envelope method (mm) 
- **wp**: Water level percentile method (mm)
- **we**: Water level envelope method (mm)
- **vs**: Valid sample count
- **coreid**: Particle device ID
- **event**: Event name
- **published_at**: Particle cloud timestamp

## NOAA Data Source

**Station**: 8656483 (Duke Marine Lab, Beaufort, NC)
- Wind data: Updates every 6 minutes (metric units)
- Water level: Updates every 1 minute (English units)
- Time zone: Local (EST/EDT)

## Error Handling

The function performs strict validation to ensure data quality:

### Data Validation Rules
- **Wind data**: Must contain at least 1 data point with fields `s`, `d`, and `g` (uses last element)
- **Water data**: Must contain at least 1 data point with fields `v` and `t` (uses last element)
- **Field presence**: All expected fields must be present and not null/undefined
- **Data selection**: Always uses the last element from the data array because NOAA API occasionally returns the whole day's data instead of just the latest reading

### Error Scenarios
If any validation fails, the function sets error values and logs detailed error messages:
- **ws**: -999 (wind speed error)
- **wd**: -999 (wind direction error)  
- **gs**: -999 (gust speed error)
- **wm**: -999 (water level error)

### Error Logging
- **Missing fields**: Logs actual field values received for debugging
- **Array length**: Logs actual data array length when empty or missing
- **API failures**: Logs fetch errors with full error details

These -999 values indicate validation failures or API errors, not actual measurements.

## Example Complete Entry

```json
{
  "coreid": "e00fce683c5052a113e58edd",
  "event": "tideMonitor/reading",
  "gs": "12.3",
  "he": "50", 
  "hp": "46",
  "published_at": "2025-07-14T21:44:51.441Z",
  "t": "2025-07-14T21:44:49Z",
  "vs": "512",
  "w": "619",
  "wd": "180",
  "we": "594",
  "wm": "2.5",
  "wp": "598",
  "ws": "8.7"
}
```

## Development

### Deploy Both Functions
```bash
cd backend/firebase-functions
firebase deploy --only functions
```

### Deploy Individual Functions
```bash
# Deploy only tide enrichment
firebase deploy --only functions --source tide-enrichment

# Deploy only tidal analysis (remember to enable first!)
firebase functions:config:set tidal.analysis.enabled=true
firebase deploy --only functions --source tidal-analysis
```

### View Logs
```bash
# All functions
firebase functions:log

# Specific function
firebase functions:log --only enrichTideData
firebase functions:log --only runTidalAnalysis
```

### Test Locally
```bash
firebase emulators:start --only functions
```

### Tidal Analysis Management
```bash
# Enable tidal analysis (IMPORTANT: Disabled by default)
firebase functions:config:set tidal.analysis.enabled=true

# Disable to save costs
firebase functions:config:set tidal.analysis.enabled=false

# Check current config
firebase functions:config:get
```

## Monitoring

### Check Latest Enriched Data
[View last 5 readings](https://tide-monitor-boron-default-rtdb.firebaseio.com/readings.json?orderBy="$key"&limitToLast=5)

### Check NOAA APIs Directly
- [Wind data](https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=latest&station=8656483&product=wind&units=metric&time_zone=lst_ldt&format=json&application=Michael.wayne.jones@gmail.com)
- [Water level data](https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=latest&station=8656483&product=water_level&datum=MLLW&time_zone=lst_ldt&units=english&format=json&application=Michael.wayne.jones@gmail.com)

### Dashboard Integration

The enriched data is used by the web dashboards for:
- **Main Dashboard**: Smoothed trend visualization with dual-axis display
  - **Water level and wave height**: Left axis (0-6 feet) with cubic spline smoothing
  - **Wind speed**: Right axis (0-40 knots) with automatic unit conversion
  - **Data processing**: 30-point progressive smoothing and -999 error filtering
- **Debug Dashboard**: Advanced analytics with 24-hour default view and 72-hour zoom capability
  - **Intuitive controls**: Dedicated time range buttons and advanced zoom functionality
  - **Async performance**: Chart loads immediately, analysis runs in background
  - **Extended visualization**: 72 hours of wind speed and gust data (0-40 knot range)
  - **Water level comparison**: Sensor vs NOAA measurements over 72-hour period
  - **Advanced trend analysis**: 
    - **Water Level Harmonics**: Matrix Pencil signal analysis for non-periodic tidal reconstruction
    - **Wind/Wave Splines**: 30-point smoothed cubic interpolation
  - **Automatic tidal analysis**: Matrix Pencil multiple frequency detection with comprehensive results table
  - **Auto-refresh**: Analysis updates every 2 minutes with new data
  - **Environmental context**: Integrated NOAA wind and water level data

### Troubleshooting

If enrichment is not working:
1. Check function logs: `firebase functions:log`
2. Verify all fields are present in latest Firebase entries
3. Test NOAA APIs directly using the URLs above
4. Look for -999 error values indicating API failures
5. Check debug dashboard for comprehensive analysis features:
   - 72-hour data visualization for extended pattern analysis
   - Matrix Pencil tidal frequency detection with up to 8 signal components
   - Water level harmonic analysis with non-periodic signal processing
   - Wind/wave cubic spline smoothing (30-point averaging)
   - Automatic tidal analysis table showing detected frequencies (M2, S2, K1, O1, etc.)
   - Unified toggle control for both trend line methodologies
   - Cached Matrix Pencil results for improved performance
# Firebase Cloud Functions

This directory contains Firebase Cloud Functions that automatically enrich tide monitoring data with environmental conditions from NOAA.

## Overview

The `enrichTideData` function automatically triggers when new readings are written to the Firebase `/readings/` path and enriches each entry with real-time wind and water level data from NOAA station 8656483 (Duke Marine Lab, Beaufort, NC).

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

If NOAA APIs are unavailable, the function sets error values:
- **ws**: -999 (wind speed error)
- **wd**: -999 (wind direction error)
- **gs**: -999 (gust speed error)
- **wm**: -999 (water level error)

These -999 values indicate API failures, not actual measurements.

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

### Deploy Functions
```bash
cd backend/firebase-functions
firebase deploy --only functions
```

### View Logs
```bash
firebase functions:log
```

### Test Locally
```bash
firebase emulators:start --only functions
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
- **Debug Dashboard**: Advanced analytics with 72-hour data window and dual trend methodologies
  - **Extended visualization**: 72 hours of wind speed and gust data (0-40 knot range)
  - **Water level comparison**: Sensor vs NOAA measurements over 3-day period
  - **Advanced trend analysis**: 
    - **Water Level Harmonics**: FFT-based automatic tidal period detection
    - **Wind/Wave Splines**: 30-point smoothed cubic interpolation
  - **Automatic tidal analysis**: Real-time frequency table with constituent identification
  - **Environmental context**: Integrated NOAA wind and water level data

### Troubleshooting

If enrichment is not working:
1. Check function logs: `firebase functions:log`
2. Verify all fields are present in latest Firebase entries
3. Test NOAA APIs directly using the URLs above
4. Look for -999 error values indicating API failures
5. Check debug dashboard for comprehensive analysis features:
   - 72-hour data visualization for extended pattern analysis
   - FFT-based tidal frequency detection with automatic constituent identification
   - Water level harmonic analysis with data-driven period detection
   - Wind/wave cubic spline smoothing (30-point averaging)
   - Automatic tidal analysis table showing detected frequencies and classifications
   - Unified toggle control for both trend line methodologies
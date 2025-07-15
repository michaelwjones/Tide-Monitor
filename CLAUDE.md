# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Tide Monitor** project that measures water levels and wave heights using an ultrasonic sensor connected to a Particle Boron 404X microcontroller. The system automatically enriches readings with real-time environmental data from NOAA. The system consists of three main components:

1. **Embedded firmware** (`backend/boron404x/tide-monitor-analog.ino`) - Arduino-style C++ code for the Particle Boron that:
   - Reads analog voltage from HRXL-MaxSonar MB7360 ultrasonic sensor
   - Takes 512 samples every minute for noise reduction
   - Calculates water level (average) and three different wave height measurements (percentile, envelope, binning)
   - Also calculates water level using each of the three wave analysis methods
   - Stores readings offline in RAM when connectivity is lost
   - Publishes data to Particle cloud, which forwards to Firebase

2. **Web dashboard** (`index.html`) - Single-page HTML application that:
   - Fetches data from Firebase Realtime Database
   - Displays real-time chart of last 24 hours using Chart.js
   - Auto-refreshes every 2 minutes

3. **Debug dashboard** (`debug/index.html`) - Additional interface for:
   - Raw data inspection and debugging with all measurement methods
   - NOAA environmental data visualization (wind, water level)
   - System status monitoring
   - Same clean layout as main dashboard for consistency

4. **Firebase Cloud Functions** (`backend/firebase-functions/`) - Serverless functions that:
   - Automatically trigger when new readings arrive
   - Fetch real-time wind and water level data from NOAA station 8656483 (Duke Marine Lab)
   - Enrich each reading with environmental conditions
   - Handle API failures gracefully with error codes

## Data Flow Architecture

```
Ultrasonic Sensor → Particle Boron → Particle Cloud → Firebase → Cloud Functions → Enriched Data → Web Dashboard
                                                                      ↓
                                                              NOAA APIs (Duke Marine Lab)
```

- **Sensor data**: Distance measurements converted to water level (mm)
- **Firebase endpoint**: `https://tide-monitor-boron-default-rtdb.firebaseio.com/readings/`
- **Data enrichment**: Cloud Functions automatically add NOAA environmental data
- **Data format**: JSON with sensor data + NOAA wind/water level data
- **Integration config**: `backend/particle.io/firebase integration.txt` contains Particle webhook configuration

## Key Technical Details

### Embedded System (`tide-monitor-analog.ino`)
- **Sensor pin**: A5 analog input
- **Mount height**: 8 feet (configured in MOUNT_HEIGHT constant)
- **Sampling**: 512 samples per reading with 50ms delays
- **Offline storage**: Up to 1000 readings in RAM (~16 hours)
- **Data validation**: Filters invalid sensor readings (300-5000mm range)
- **Wave calculations**: Three different algorithms for wave height measurement
- **Water level calculations**: Four different water level measurements (overall average + three method-specific levels)
- **JSON length**: Increased to 120 characters to accommodate additional data fields

### Web Dashboard (`index.html`)
- **Chart library**: Chart.js v4.5.0 with date-fns adapter
- **Data range**: Last 1440 readings (24 hours) from Firebase
- **Chart features**: Time-series plot with water level and wave measurements
- **Auto-refresh**: Updates every 2 minutes
- **Timezone**: Eastern Time (America/New_York)
- **Navigation**: Link to debug dashboard in top-right corner

### Debug Dashboard (`debug/index.html`)
- **Chart library**: Chart.js v4.5.0 with date-fns adapter (same as main)
- **Data visualization**: All 8 data fields on multi-axis chart
- **Layout**: Clean design matching main dashboard (no containers/borders)
- **Navigation**: Link back to main dashboard in top-right corner
- **Chart axes**: 
  - Left Y-axis: Water level measurements (0-6 feet)
  - Right Y-axis: Wave height measurements (0-2 feet)
  - Hidden Y-axis: Valid sample count (0-512)

### Debug Dashboard (`debug/index.html`)
- **Multi-axis charts**: Displays all measurement methods and NOAA data
- **Y-axes**: Water level (0-6 ft), Wave height (0-2 ft), Wind speed (0-30 knots)
- **NOAA integration**: Real-time wind and water level from Duke Marine Lab
- **Error handling**: Filters out -999 API failure values
- **Units**: Automatic conversion from m/s to knots for wind data

### Firebase Cloud Functions (`backend/firebase-functions/`)
- **Trigger**: `onValueCreated` for new readings in `/readings/` path
- **APIs**: NOAA station 8656483 (Duke Marine Lab, Beaufort, NC)
- **Data added**: Wind speed (ws), direction (wd), gust (gs), water level (wm)
- **Error handling**: Sets -999 values when NOAA APIs fail
- **Framework**: Firebase Functions v6 with Node.js 22

## Development Commands

This project uses minimal build tools. Development involves:

1. **Firmware development**: Use Particle Workbench or Web IDE for the Arduino code
   - **Flash to device**: Run `flash.bat` in `backend/boron404x/` directory to deploy firmware to Particle Boron
2. **Web dashboard**: Open `index.html` directly in browser or serve with local HTTP server
3. **Firebase Functions**: Deploy Cloud Functions for data enrichment
   - **Deploy**: `cd backend/firebase-functions && firebase deploy --only functions`
   - **Logs**: `firebase functions:log`
   - **Test locally**: `firebase emulators:start --only functions`
4. **Testing**: Manual testing with live sensor data or Firebase data inspection

## Project Structure

```
Tide-Monitor/
├── index.html                           # Main web dashboard (GitHub Pages root)
├── debug/
│   └── index.html                       # Debug dashboard with NOAA data visualization
└── backend/
    ├── boron404x/
    │   └── tide-monitor-analog.ino      # Particle Boron firmware
    ├── firebase-functions/
    │   ├── README.md                    # Cloud Functions documentation
    │   ├── firebase.json                # Firebase configuration
    │   └── tide-monitor/
    │       ├── index.js                 # NOAA data enrichment function
    │       └── package.json             # Node.js dependencies
    └── particle.io/
        └── firebase integration.txt     # Webhook configuration
```

## Firebase Data Schema

Readings are stored with auto-generated keys containing:

### Original Sensor Data (from Particle Boron)
- `t`: ISO8601 timestamp
- `w`: Water level in mm (average of all valid samples)
- `hp`: Wave height (percentile method) in mm
- `he`: Wave height (envelope method) in mm
- `hb`: Wave height (binning method) in mm
- `wp`: Water level (percentile method) in mm
- `we`: Water level (envelope method) in mm
- `wb`: Water level (binning method) in mm
- `vs`: Valid sample count
- `coreid`: Particle device ID
- `event`: Event name
- `published_at`: Particle cloud timestamp

### NOAA Environmental Data (added by Cloud Functions)
- `ws`: Wind speed in m/s from Duke Marine Lab
- `wd`: Wind direction in degrees from Duke Marine Lab
- `gs`: Gust speed in m/s from Duke Marine Lab
- `wm`: Water level in feet (MLLW datum) from Duke Marine Lab

**Note**: NOAA fields show -999 when APIs are unavailable

## Dashboard Features

### Main Dashboard Features
- **Chart visualization**: Water level (blue) and three wave height methods (red, orange, purple)
- **Reference line**: Green horizontal line at 2.5 feet
- **Custom legend**: Color-coded legend below chart
- **Responsive design**: Works on desktop and mobile devices
- **Auto-refresh**: Updates every 2 minutes
- **Security**: CSP headers and content-type protection

### Debug Dashboard Features  
- **Comprehensive data**: Shows all 8 data fields from Firebase
- **Multi-axis visualization**: 
  - Water level measurements: Average (blue), Percentile (cyan), Envelope (magenta), Binning (lime)
  - Wave height measurements: Percentile (red), Envelope (orange), Binning (purple)  
  - Valid samples: Gray line (hidden axis for scale)
- **Clean layout**: Same styling as main dashboard for consistency
- **Navigation**: Easy switching between main and debug views

## Technical Notes

- **Deployment**: Hosted on GitHub Pages with Netlify-style headers
- **Security**: Content Security Policy prevents XSS attacks
- **Error handling**: Graceful degradation when data is unavailable
- **Performance**: Optimized chart rendering with disabled animations
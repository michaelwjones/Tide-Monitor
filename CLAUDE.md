# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Tide Monitor** project that measures water levels and wave heights using an ultrasonic sensor connected to a Particle Boron 404X microcontroller. The system consists of two main components:

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
   - Raw data inspection and debugging
   - System status monitoring
   - Diagnostic tools and troubleshooting

## Data Flow Architecture

```
Ultrasonic Sensor → Particle Boron → Particle Cloud → Firebase → Web Dashboard
```

- **Sensor data**: Distance measurements converted to water level (mm)
- **Firebase endpoint**: `https://tide-monitor-boron-default-rtdb.firebaseio.com/readings/`
- **Data format**: JSON with timestamp (t), water level (w), wave height calculations (hp, he, hb), and method-specific water levels (wp, we, wb)
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

## Development Commands

This project does not use traditional build tools. Development involves:

1. **Firmware development**: Use Particle Workbench or Web IDE for the Arduino code
   - **Flash to device**: Run `flash.bat` in `backend/boron404x/` directory to deploy firmware to Particle Boron
2. **Web dashboard**: Open `index.html` directly in browser or serve with local HTTP server
3. **Testing**: Manual testing with live sensor data or Firebase data inspection

## Project Structure

```
Tide-Monitor/
├── index.html                           # Main web dashboard (GitHub Pages root)
├── debug/
│   └── index.html                       # Debug dashboard (/debug URL path)
└── backend/
    ├── boron404x/
    │   └── tide-monitor-analog.ino      # Particle Boron firmware
    └── particle.io/
        └── firebase integration.txt     # Webhook configuration
```

## Firebase Data Schema

Readings are stored with auto-generated keys containing:
- `t`: ISO8601 timestamp
- `w`: Water level in mm (average of all valid samples)
- `hp`: Wave height (percentile method) in mm
- `he`: Wave height (envelope method) in mm
- `hb`: Wave height (binning method) in mm
- `wp`: Water level (percentile method) in mm
- `we`: Water level (envelope method) in mm
- `wb`: Water level (binning method) in mm
- `vs`: Valid sample count
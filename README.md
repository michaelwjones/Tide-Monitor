# Tide Monitor

Real-time water level and wave height monitoring system using ultrasonic sensors and IoT technology.

## Live Dashboards

- **Main Dashboard:** [https://michaelwjones.github.io/Tide-Monitor/](https://michaelwjones.github.io/Tide-Monitor/)  
- **Debug Dashboard:** [https://michaelwjones.github.io/Tide-Monitor/debug/](https://michaelwjones.github.io/Tide-Monitor/debug/)

## System Overview

Ultrasonic sensor (HRXL-MaxSonar MB7360) → Particle Boron 404X → Firebase → Web dashboards

**Measurements**: Water level, wave heights, wind speed/direction (from NOAA), data quality metrics

**Features**: Real-time visualization, auto-refresh every 2 minutes, 24-hour ML forecasting, wind direction indicators (both dashboards)

## Quick Start

1. **View Live Data**: Visit dashboard links above
2. **Development**: Clone repo, open `index.html` in browser
3. **Advanced Setup**: See `/backend/README.md` for firmware and cloud functions

## Components

- **Firmware**: Arduino C++ for Particle Boron (`backend/boron404x/`)
- **Dashboards**: HTML/JS with Chart.js (`index.html`, `debug/index.html`)  
- **Cloud Functions**: Firebase serverless functions (`backend/firebase-functions/`)
- **ML Forecasting**: LSTM and Transformer models for 24-hour predictions

## Data Format

Sensor data: `t`, `w`, `hp`, `he`, `wp`, `we`, `vs` + NOAA environmental data: `ws`, `wd`, `gs`, `wm`

Data updates every minute, dashboards refresh every 2 minutes.

## Architecture Details

For detailed technical information, see:
- `/backend/README.md` - Hardware, firmware, cloud functions
- `/debug/README.md` - Advanced dashboard features (wind direction, auto-forecasting)
- Method-specific README files for ML models and analysis functions
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Tide Monitor** project that measures water levels and wave heights using an ultrasonic sensor connected to a Particle Boron 404X microcontroller. The system automatically enriches readings with real-time environmental data from NOAA. The system consists of three main components:

1. **Embedded firmware** (`backend/boron404x/tide-monitor-analog.ino`) - Arduino-style C++ code for the Particle Boron that:
   - Reads analog voltage from HRXL-MaxSonar MB7360 ultrasonic sensor
   - Takes 512 samples every minute for noise reduction
   - Calculates water level (average) and two different wave height measurements (percentile, envelope)
   - Also calculates water level using each of the two wave analysis methods
   - Stores readings offline in RAM when connectivity is lost
   - Publishes data to Particle cloud, which forwards to Firebase

2. **Web dashboard** (`index.html`) - Single-page HTML application that:
   - Fetches data from Firebase Realtime Database
   - Displays real-time chart of last 24 hours using Chart.js
   - Auto-refreshes every 2 minutes

3. **Debug dashboard** (`debug/index.html`) - Additional interface for:
   - Raw data inspection and debugging with selected measurement methods
   - NOAA environmental data visualization (wind, water level)
   - System status monitoring
   - Same clean layout as main dashboard for consistency

4. **Firebase Cloud Functions** (`backend/firebase-functions/`) - Four separate serverless function systems:
   - **Data Enrichment** (`tide-enrichment/`): Automatically triggers when new readings arrive
   - **Tidal Analysis Functions** (`tidal-analysis/functions/`): Multiple analysis methods for advanced signal processing
   - **LSTM Forecasting** (`tidal-analysis/functions/lstm/v1/`): Neural network-powered 24-hour water level predictions
   - **Transformer Forecasting** (`tidal-analysis/functions/transformer/v1/`): Single-pass encoder-only transformer for direct 433→144 prediction (no autoregressive generation)
   - Fetch real-time wind and water level data from NOAA station 8656483 (Duke Marine Lab)
   - Enrich each reading with environmental conditions
   - Perform strict data validation requiring at least 1 data point (uses last element)
   - Validate all expected fields are present (wind: s,d,g; water: v,t)
   - Handle variable API response sizes (NOAA occasionally returns full day data)
   - Handle API failures gracefully with error codes and detailed logging
   - Use `.env` files for configuration (modern approach, replaces deprecated `functions:config`)

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
- **Wave calculations**: Two different algorithms for wave height measurement
- **Water level calculations**: Three different water level measurements (overall average + two method-specific levels)
- **JSON length**: Increased to 120 characters to accommodate additional data fields

### Web Dashboard (`index.html`)
- **Chart library**: Chart.js v4.5.0 with date-fns adapter
- **Data range**: Last 1440 readings (24 hours) from Firebase
- **Chart features**: Dual-axis time-series plot with smoothed trend lines
- **Left Y-axis**: Water level and wave height (0-6 feet)
- **Right Y-axis**: Wind speed (0-40 knots)
- **Data smoothing**: 30-point progressive smoothing with cubic spline interpolation
- **Unit conversion**: Wind speed converted from m/s to knots (1 m/s = 1.94384 knots)
- **Error filtering**: Excludes NOAA API failure values (-999)
- **Auto-refresh**: Updates every 2 minutes
- **Timezone**: Eastern Time (America/New_York)
- **Navigation**: Link to debug dashboard in top-right corner

### Debug Dashboard (`debug/index.html`)
- **Chart library**: Chart.js v4.5.0 with date-fns adapter (same as main)
- **Default view**: 24 hours with zoom capability to full 72-hour dataset
- **Time controls**: Dedicated 24-hour and 72-hour buttons for quick navigation
- **Advanced zoom**: Click-and-drag selection and Ctrl+mouse wheel functionality
- **Data visualization**: Selected data fields on multi-axis chart
- **Layout**: Clean design matching main dashboard (no containers/borders)
- **Navigation**: Link back to main dashboard in top-right corner
- **Async loading**: Chart displays immediately, tidal analysis runs in background
- **Auto-refresh**: Updates every 2 minutes with fresh data and analysis
- **Signal analysis**: Matrix Pencil method for non-periodic tidal data analysis
- **Chart axes**: 
  - Left Y-axis: Water level measurements (0-6 feet)
  - Right Y-axis: Wave height measurements (0-1 feet)
- **Transformer Forecasting**: 24-hour water level predictions from transformer v1 model
  - **Smart button**: "Show Forecast" button appears only when recent forecast available (< 10 minutes old)
  - **Point display**: Forecast shown as individual points at 10-minute intervals extending 24 hours into future
  - **Automatic axis extension**: Chart X-axis extends to accommodate forecast timeline
  - **Button logic**: "Show Forecast" adds 24h to timeline, "Hide Forecast" removes extension
  - **Integration**: Fetches from `/tidal-analysis/transformer-v1-forecast` Firebase path

### Debug Dashboard (`debug/index.html`)
- **Multi-axis charts**: Displays selected measurement methods and NOAA environmental data
- **Y-axes**: Water level (0-6 ft), Wave height (0-1 ft), Wind speed (0-40 knots)
- **NOAA integration**: Real-time wind and water level from Duke Marine Lab station 8656483
- **Transformer forecasting**: 24-hour predictions displayed as points when recent forecast available
- **Time range controls**: 24h/72h buttons automatically extend for forecast (48h/96h total)
- **Error handling**: Filters out -999 API failure values and invalid data
- **Units**: Automatic conversion from m/s to knots for wind data
- **Clean interface**: Simplified layout focused on data visualization without analysis tables

### Firebase Cloud Functions (`backend/firebase-functions/`)
- **Trigger**: `onValueCreated` for new readings in `/readings/` path
- **APIs**: NOAA station 8656483 (Duke Marine Lab, Beaufort, NC)
- **Data added**: Wind speed (ws), direction (wd), gust (gs), water level (wm)
- **Data validation**: Requires exactly 1 data point with all expected fields present
- **Wind validation**: Fields s (speed), d (direction), g (gust) must exist
- **Water validation**: Fields v (value), t (time) must exist
- **Error handling**: Sets -999 values when APIs fail or validation fails
- **Error logging**: Detailed console logs for missing fields and array length issues
- **Framework**: Firebase Functions v6 with Node.js 22

## Development Commands

This project uses minimal build tools. Development involves:

1. **Firmware development**: Use Particle Workbench or Web IDE for the Arduino code
   - **Flash to device**: Run `flash.bat` in `backend/boron404x/` directory to deploy firmware to Particle Boron
2. **Web dashboard**: Open `index.html` directly in browser or serve with local HTTP server
3. **Firebase Functions**: Deploy Cloud Functions for data enrichment, tidal analysis, and LSTM forecasting
   - **Enrichment only**: `deploy-enrichment.bat` (NOAA data enrichment, always safe)
   - **Matrix Pencil v1**: `deploy-matrix-pencil-v1.bat` (tidal analysis with cost control)
   - **LSTM v1 Forecasting**: `tidal-analysis/functions/lstm/v1/deploy-lstm-v1.bat` (requires trained model)
   - **Transformer v1 Forecasting**: `tidal-analysis/functions/transformer/v1/deploy-transformer-v1.bat` (requires trained model, deploys specific function only)
   - **Manual CLI**: `firebase deploy --only functions --source tide-enrichment` or `--source tidal-analysis/functions/matrix-pencil/v1` or `--source tidal-analysis/functions/lstm/v1/inference` or `--source tidal-analysis/functions/transformer/v1/inference`
   - **Prerequisites**: Run `npm install` in each function directory before first deployment
   - **Logs**: `firebase functions:log`
   - **Test locally**: `firebase emulators:start --only functions`
4. **LSTM Model Testing**: Comprehensive validation interface for trained models
   - **Web Interface**: `tidal-analysis/functions/lstm/v1/testing/start-server.bat` (Windows auto-launcher)
   - **Manual Start**: `python server.py` in testing folder, then open `http://localhost:8000`
   - **Features**: Real Firebase data fetching, 24-hour forecasting, interactive visualization
   - **Validation**: Test model performance before deploying to production
5. **Testing**: Manual testing with live sensor data or Firebase data inspection

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
    │   ├── firebase.json                # Multi-codebase Firebase configuration
    │   ├── deploy-enrichment.bat        # NOAA enrichment deployment
    │   ├── deploy-matrix-pencil-v1.bat  # Matrix Pencil v1 deployment with cost control
    │   ├── toggle-matrix-pencil-v1.bat  # Matrix Pencil v1 enable/disable toggle
    │   ├── MATRIX_PENCIL_V1.md          # Matrix Pencil methodology documentation
    │   ├── tide-enrichment/             # NOAA data enrichment function
    │   │   ├── index.js
    │   │   └── package.json
    │   └── tidal-analysis/              # Analysis functions container
    │       ├── analysis-functions.csv   # Function tracking spreadsheet
    │       └── functions/               # Organized analysis functions
    │           ├── matrix-pencil/       # Matrix Pencil analysis method
    │           │   └── v1/              # Matrix Pencil version 1
    │           │       ├── README.md    # Analysis deployment guide
    │           │       ├── index.js     # Matrix Pencil v1 implementation
    │           │       └── package.json
    │           ├── lstm/                # LSTM neural network forecasting
    │           │   └── v1/              # LSTM version 1
    │           │       ├── README.md    # LSTM setup and deployment guide
    │           │       ├── LSTM_V1.md   # Technical methodology documentation
    │           │       ├── data-preparation/  # Python scripts for training data
    │           │       ├── training/    # PyTorch model training pipeline
    │           │       ├── testing/     # Model validation and testing interface
    │           │       │   ├── README.md           # Testing documentation
    │           │       │   ├── index.html          # Web-based testing interface
    │           │       │   ├── server.py           # HTTP server for model testing
    │           │       │   ├── test_model.py       # Command-line testing script
    │           │       │   ├── start-server.bat    # Windows server launcher
    │           │       │   └── firebase_fetch.py   # Firebase data utilities
    │           │       ├── inference/   # Firebase Functions ONNX deployment
    │           │       └── deploy-lstm-v1.bat  # Deployment script
    │           └── transformer/         # Transformer neural network forecasting
    │               └── v1/              # Transformer version 1
    │                   ├── README.md    # Transformer setup and deployment guide
    │                   ├── data-preparation/  # Optimized data pipeline with timestamp-based matching
    │                   │   ├── fetch_firebase_data.py     # Raw data fetching with quality filtering
    │                   │   └── create_training_data.py    # Timestamp-based sequence generation
    │                   ├── training/    # PyTorch transformer training pipeline
    │                   ├── testing/     # Model validation and testing interface
    │                   ├── inference/   # Firebase Functions PyTorch deployment
    │                   └── deploy-transformer-v1.bat  # Deployment script
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
- `wp`: Water level (percentile method) in mm
- `we`: Water level (envelope method) in mm
- `vs`: Valid sample count
- `coreid`: Particle device ID
- `event`: Event name
- `published_at`: Particle cloud timestamp

### NOAA Environmental Data (added by Cloud Functions)
- `ws`: Wind speed in m/s from Duke Marine Lab
- `wd`: Wind direction in degrees from Duke Marine Lab
- `gs`: Gust speed in m/s from Duke Marine Lab
- `wm`: Water level in feet (MLLW datum) from Duke Marine Lab

**Note**: NOAA fields show -999 when APIs are unavailable or validation fails (wrong array length or missing fields)

## Dashboard Features

### Main Dashboard Features
- **Chart visualization**: Water level (blue), wave height trend (red), wind speed trend (purple)
- **Dual-axis display**: Water level/waves on left (0-6 feet), wind speed on right (0-40 knots)
- **Smoothed trend lines**: 30-point progressive smoothing with cubic spline interpolation
- **Reference line**: Green horizontal line at 2.5 feet
- **Custom legend**: Color-coded legend below chart
- **Data processing**: Automatic unit conversion (m/s to knots) and error filtering
- **Responsive design**: Works on desktop and mobile devices
- **Auto-refresh**: Updates every 2 minutes
- **Security**: CSP headers and content-type protection

### Debug Dashboard Features  
- **Comprehensive data**: Shows all 6 data fields from Firebase plus NOAA environmental data
- **Multi-axis visualization**: 
  - Water level measurements: Average (blue), Percentile (cyan), Envelope (magenta)
  - Wave height measurements: Percentile (red), Envelope (orange)  
  - Valid samples: Gray line (hidden axis for scale)
  - NOAA data: Wind speed (green), Gust speed (dark green), Duke water level (dark blue)
- **Trend line analysis**: Natural cubic spline interpolation with toggle functionality
  - **Performance**: O(n) algorithm processes 1440 data points in under 100ms
  - **Smoothing**: 7-point adaptive smoothing for optimal curve fitting
  - **Visual clarity**: Original data fades to 30% opacity when trend lines are shown
  - **Edge safety**: Natural splines prevent oscillation artifacts at data boundaries
- **LSTM 24-Hour Forecasting**: Neural network water level predictions
  - **Button**: "Show 24h LSTM Forecast" toggle beside trend line controls
  - **Visual Display**: Dashed orange forecast lines extending 24 hours into future
  - **Iterative Prediction**: Uses last 72 hours to generate 1,440 minute-by-minute forecasts
  - **Auto-Refresh**: Fresh predictions generated every 6 hours by Firebase Functions
  - **Chart Extension**: X-axis automatically extends 24 hours beyond current data when active
- **Clean layout**: Same styling as main dashboard for consistency
- **Navigation**: Easy switching between main and debug views

## Optimized Data Processing Pipeline

### Transformer v1 Data Preparation
The transformer v1 system uses an optimized data pipeline focused on quality filtering and timestamp-based sequence matching:

#### Data Fetching and Filtering (`fetch_firebase_data.py`)
- **Firebase Integration**: Downloads complete dataset from Firebase Realtime Database
- **Quality Filtering**: Removes physically impossible water levels (< -200mm) from source data
- **Dual File Output**: Preserves both raw data and filtered data for analysis
- **Data Validation**: Filters out 3.9% of readings while preserving 96.1% of quality sensor data
- **Performance**: Efficiently processes 100K+ readings with immediate filtering

#### Timestamp-Based Sequence Generation (`create_training_data.py`)
- **Binary Search Optimization**: Uses O(log n) binary search instead of linear search for 100x speed improvement
- **Timestamp Matching**: Finds closest readings within ±5 minutes of target 10-minute intervals
- **No Synthetic Data**: Works with real sensor data only, no gap filling or synthetic values
- **Quality Control**: Filters sequences with insufficient data coverage automatically
- **Temporal Split**: Creates proper train/validation split with temporal gap to prevent data leakage

#### Current Pipeline Results (Latest Run)
- **Source Data**: 97,525 filtered readings from Firebase
- **Generated Sequences**: 18,346 potential 96-hour sequences processed
- **Valid Sequences**: 14,547 sequences (79.3% retention rate) 
- **Training Split**: 11,638 training / 1,743 validation sequences
- **Temporal Gap**: 192.2 hours discarded to ensure no data leakage
- **File Size**: 58.9 MB of high-quality training data
- **Processing Time**: Under 1 minute with binary search optimization

## Technical Notes

### Transformer v1 Inference Improvements (Recent)

**Real-Time Temporal Alignment** - The transformer inference system now uses sophisticated temporal alignment:
- **Current Time Reference**: Uses actual inference time to create proper 72-hour timeline
- **Backwards Timeline Generation**: Creates 433 target times at 10-minute intervals going back from now
- **Smart Data Matching**: Maps each target time to closest available reading (±5 minute tolerance)
- **Minimal Synthetic Data**: Only uses -999 when genuinely no data available in time slots
- **Chronological Preservation**: Maintains proper oldest-to-newest order for model input

**Direct Model Input** - New `predict_24_hours_direct()` method:
- **Bypasses Input Preparation**: Prevents double processing and data corruption 
- **Preserves Synthetic Values**: Maintains exact -999 values as model expects
- **No Mean Contamination**: Eliminates incorrect mean calculations that include -999 values
- **Pre-Validated Sequences**: Feeds exactly 433 points directly to model without modification
- **Enhanced Error Reporting**: Concise file:line - ErrorType: message format in Firebase

**Robust Data Requirements**:
- **Minimum Threshold**: Only 100 readings required (vs previous 4300+ requirement)  
- **Proper Timestamp Base**: Uses last real data timestamp (not synthetic) for prediction alignment
- **Deployment Precision**: Uses `--only functions:run_transformer_v1_analysis` to avoid orphaned function prompts

These improvements fix uniform prediction outputs and ensure natural tidal variations in forecasts.

- **Deployment**: Hosted on GitHub Pages with Netlify-style headers
- **Security**: Content Security Policy prevents XSS attacks
- **Error handling**: Graceful degradation when data is unavailable
- **Performance**: Optimized chart rendering with disabled animations
- **Signal analysis algorithm**: Matrix Pencil method for parameter estimation of non-periodic signals
- **Trend line approach**: Complex exponential decomposition Σ Aₖ e^(σₖ + jωₖ)t with damping factors
- **Mathematical robustness**: SVD-based model order selection and generalized eigenvalue decomposition
- **No periodicity assumptions**: Handles real-world tidal variations and weather effects naturally

## Development History

### Removed Components

**Binning Analysis Method**: Previously implemented as a third wave height calculation method, the binning approach was removed from the system due to performance issues:

- **Analysis Problem**: The method grouped sensor readings into discrete bins to calculate wave heights, but this approach produced erratic results
- **Resolution Issues**: The binning approach provided lower resolution than percentile and envelope methods  
- **Data Quality**: Introduced measurement artifacts that degraded the quality of wave height calculations
- **Removal Scope**: All binning code was removed from firmware (`hb`/`wb` fields), web dashboards, JSON configurations, and documentation

This removal simplified the system to focus on the two reliable wave analysis methods (percentile and envelope) that provide consistent, high-resolution measurements.

**Autoregressive Transformer Architecture (REMOVED)**: The transformer v1 system previously used an autoregressive seq2seq architecture that caused prediction quality issues:

- **Problem**: Training used single-pass (433→144) but inference used 144-step autoregressive loops  
- **Result**: Model learned one behavior but was forced to perform differently during production
- **Symptoms**: Uniform outputs, no sinusoidal patterns, poor tidal prediction quality
- **Root Cause**: Zero-initialization in decoder during autoregressive generation
- **Resolution**: Replaced with single-pass encoder-only transformer architecture

**Current Transformer v1 Architecture (IMPLEMENTED)**: Single-pass encoder-only transformer for reliable tidal prediction:

- **Architecture**: Encoder-only transformer with multi-head attention (16 heads, 8 layers, 512 dimensions)
- **Input/Output**: Direct 433→144 mapping (72 hours input → 24 hours output at 10-minute intervals)
- **Training**: Single forward pass: `model(src)` → predictions (no teacher forcing needed)
- **Inference**: Identical single forward pass: `model(src)` → predictions (no autoregressive generation)
- **Benefits**: Consistent training/inference behavior, natural sinusoidal outputs, reliable tidal patterns
- **Implementation**: `model.py`, `main.py`, `modal_single_run_seq2seq.py` all use single-pass architecture

**CRITICAL**: Never implement autoregressive generation for transformer v1. Always use single-pass encoder-only architecture.

## Memory & Context

Always keep these rules in your context:

- Do not mask a failure with a fallback
- Occasionally I will ask for options, ideas, or advice. In those cases, do not make any code changes.
- Follow directions exactly, make suggestions separately, don't make unauthorized changes.
- When I ask you to make a change A, do not make changes A and B. Only change A. I am a very technical user, if I want B changed I will ask for it.
- Feel free to make suggestions if you think they will help me arrive at my goal.
- Don't run a firebase command unless specifically asked to.
- README files should contain current state, not change history.
- Do not put emojis or unicode in batch files or powershell scripts.
- Do not ever try to pass off fake data as real data.
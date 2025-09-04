# Tide Monitor 🌊

A real-time water level and wave height monitoring system using ultrasonic sensors and IoT technology.

## 🔗 Live Dashboard

**Main Dashboard:** [https://michaelwjones.github.io/Tide-Monitor/](https://michaelwjones.github.io/Tide-Monitor/)  
**Debug Dashboard:** [https://michaelwjones.github.io/Tide-Monitor/debug/](https://michaelwjones.github.io/Tide-Monitor/debug/)

## 📊 What It Does

This system continuously monitors water levels and wave activity using an ultrasonic sensor mounted 8 feet above the water surface. It provides real-time data visualization through web dashboards that update every 2 minutes.

### Key Measurements
- **Water Level**: Distance from sensor to water surface (converted to feet)
- **Wave Heights**: Smoothed trend analysis using cubic spline interpolation
- **Wind Speed**: Environmental data from NOAA (converted to knots)
- **Data Quality**: Valid sample count for each reading

## 🛠️ System Architecture

```
Ultrasonic Sensor → Particle Boron → Particle Cloud → Firebase → Web Dashboard
```

### Hardware
- **Sensor**: HRXL-MaxSonar MB7360 ultrasonic sensor
- **Controller**: Particle Boron 404X microcontroller  
- **Mount Height**: 8 feet above water surface
- **Sampling**: 512 readings per minute for noise reduction

### Software Stack
- **Frontend**: Vanilla HTML/CSS/JavaScript with Chart.js
- **Backend**: Particle Cloud webhooks
- **Database**: Firebase Realtime Database
- **Hosting**: GitHub Pages

## 📈 Dashboard Features

### Main Dashboard
- **24-hour water level trend** with smooth line charts
- **Wave height visualization** using cubic spline smoothed trend lines
- **Wind speed monitoring** with 0-40 knots range on dedicated right axis
- **Dual-axis display**: Water level (left, 0-6 feet) and wind speed (right, 0-40 knots)
- **Reference line** at 2.5 feet for context
- **Auto-refresh** every 2 minutes
- **Mobile responsive** design

### Debug Dashboard  
- **Default 24-hour view** with zoom capability to full 72 hours for comprehensive analysis
- **Intuitive time controls**: Dedicated 24-hour and 72-hour buttons for quick navigation
- **Advanced zoom functionality**: Click-and-drag selection and Ctrl+mouse wheel for precise control
- **Multi-axis charts** for detailed analysis with extended timeframe
- **Enhanced visualization** of all wave calculation methods and NOAA environmental data
- **System diagnostics** with valid sample tracking and wind/gust monitoring
- **24-Hour Transformer Forecasting**: Neural network-powered water level predictions
  - **Smart visibility**: Forecast button appears only when recent predictions are available (< 10 minutes old)
  - **Point-based display**: Predictions shown as individual points extending 24 hours into future
  - **Automatic axis extension**: Chart timeline extends to accommodate forecast display
  - **Adaptive time controls**: 24h/72h buttons account for forecast extension (48h/96h total)
  - **Direct Prediction**: Uses sequence-to-sequence transformer with last 72 hours to generate 144 future predictions
  - **Visual Integration**: Orange forecast points extend 24 hours into future with no connecting lines
  - **Auto-Refresh**: Fresh forecasts generated every 5 minutes via Firebase Functions
  - **Machine Learning**: PyTorch-trained transformer with direct prediction for cloud inference
- **NOAA environmental data** integration (wind, water level from Duke Marine Lab)
- **Wind range**: 0-30 knots for comprehensive weather tracking
- **Clean interface**: Simplified layout focused on data visualization without analysis tables

## 🚀 Quick Start

### View the Data
Simply visit the live dashboard links above - no installation required!

### Development Setup
1. Clone this repository
2. Open `index.html` in your browser for the main dashboard
3. Open `debug/index.html` for the debug interface
4. Or serve with a local HTTP server for development

### Deploy Firmware (Advanced)
1. Install Particle Workbench
2. Navigate to `backend/boron404x/`
3. Run `flash.bat` to deploy to your Particle Boron device

### Deploy Cloud Functions (Advanced)
1. Navigate to `backend/firebase-functions/`
2. Use `deploy-enrichment.bat` for NOAA data enrichment (always safe)
3. Use `deploy-matrix-pencil-v1.bat` for tidal analysis (costs money when enabled)
4. Use `tidal-analysis/functions/lstm/v1/deploy-lstm-v1.bat` for LSTM forecasting (requires model training)

## 📊 Data Format

The system collects readings every minute with the following data points:

```json
{
  "t": "2024-01-15T10:30:00.000Z",
  "w": 2438,
  "hp": 152,
  "he": 178,
  "wp": 2445,
  "we": 2431,
  "vs": 487,
  "ws": "8.7",
  "wd": "180",
  "gs": "12.3",
  "wm": "2.5"
}
```

### Sensor Data Fields
- `t`: Timestamp (ISO8601)
- `w`: Water level in mm (average method)
- `hp/he`: Wave heights using percentile/envelope methods
- `wp/we`: Water levels using percentile/envelope methods  
- `vs`: Valid sample count (out of 512)

### NOAA Environmental Data (added via Cloud Functions)
- `ws`: Wind speed in m/s from Duke Marine Lab
- `wd`: Wind direction in degrees from Duke Marine Lab  
- `gs`: Gust speed in m/s from Duke Marine Lab
- `wm`: Water level in feet (MLLW datum) from Duke Marine Lab

**Note**: NOAA fields show -999 when APIs are unavailable or data validation fails. The system uses the last element from NOAA data arrays because the API occasionally returns the whole day's data instead of just the latest reading.

## 🔧 Technical Details

### Wave Analysis Methods
1. **Percentile Method**: Uses statistical percentiles to calculate wave heights with cubic spline smoothing
2. **Envelope Method**: Available in debug dashboard for detailed analysis
   
### Main Dashboard Visualization
- **Smoothed Trend Lines**: All wave and wind data uses 30-point progressive smoothing
- **Cubic Spline Interpolation**: Natural splines with Thomas algorithm for optimal performance
- **Unit Conversion**: Wind speed converted from m/s to knots (1 m/s = 1.94384 knots)
- **Error Filtering**: Automatically excludes NOAA API failure values (-999)

### Advanced Trend Line Analysis

#### Water Level Harmonics (Matrix Pencil Method)
- **Mathematical Form**: `f(t) = Σ Aₖ e^(σₖ + jωₖ)t` (complex exponential decomposition)
- **Signal Components**: Up to 8 components with automatic model order selection via SVD
- **Matrix Pencil Analysis**: Advanced signal processing for non-periodic tidal data
- **Frequency Detection**: Detects multiple tidal constituents simultaneously (M2 ~12.4h, S2 ~12.0h, K1 ~23.9h, O1 ~25.8h)
- **Non-Periodic Capability**: Handles real-world tidal variations without periodicity assumptions
- **SVD-Based Selection**: Automatic component selection using 1% singular value threshold
- **Amplitude & Damping**: Estimates both oscillation amplitude and damping coefficients
- **Full Chart Coverage**: Trend lines span entire display range with no edge artifacts

#### Wind/Wave Cubic Splines
- **Enhanced Smoothing**: Progressive 1-30 point moving averages (16 smoothing levels)
- **Algorithm**: Natural cubic splines with Thomas algorithm (O(n) complexity)
- **Edge Protection**: Prevents oscillation artifacts at data boundaries
- **Dynamic Window**: Adaptive smoothing based on distance from edges
- **Superior Noise Reduction**: 30-point averaging for highly variable environmental data

### Failed Analysis Methods
**Binning Method** (Removed): This method grouped readings into bins for wave analysis but was removed due to:
- **Higher Erratic Behavior**: Produced more inconsistent results compared to percentile and envelope methods
- **Lower Resolution**: Provided less precise measurements due to the binning approach
- **Poor Signal Quality**: The discrete nature of binning introduced artifacts that degraded wave height calculations

### Data Processing
- **Noise Reduction**: 512 samples averaged per reading
- **Outlier Filtering**: Invalid readings (outside 300-5000mm) are discarded
- **Offline Storage**: Up to 1000 readings stored locally when connectivity is lost
- **Cloud Sync**: Automatic upload when connection is restored

### Advanced Analytics
- **Dual Methodology Trend Analysis**: Unified interface with specialized algorithms
  - **Water Levels**: Matrix Pencil signal analysis for non-periodic tidal reconstruction
  - **Wind/Wave Data**: Enhanced 30-point smoothed cubic splines
- **LSTM Neural Network Forecasting**: 24-hour predictive modeling
  - **Iterative Architecture**: 72-hour input sequences generate 1,440 future predictions
  - **Machine Learning**: PyTorch-trained LSTM with 2-layer, 128-unit architecture
  - **Cloud Inference**: ONNX deployment on Firebase Functions with 6-hour update cycle
  - **Visual Integration**: Dashed orange forecast lines extending 24 hours beyond current data
- **Model Validation**: Analysis Error plot shows residuals (measured - predicted + 1 offset)
  - **Quality Assessment**: Visual feedback on Matrix Pencil reconstruction accuracy
  - **Error Distribution**: Reveals systematic biases or random errors in tidal modeling
  - **Real-time Validation**: Updates automatically with each analysis refresh
- **Performance Optimized**: Cached Matrix Pencil results prevent duplicate computation
- **Edge-Safe**: Matrix Pencil reconstruction covers full chart range without artifacts
- **Toggle Control**: Single button controls both trend line methodologies
- **Visual Distinction**: Matrix Pencil uses thick dashed lines, splines use standard dashing

### Security & Performance
- **Content Security Policy**: Protection against XSS attacks
- **Optimized Rendering**: Chart animations disabled for performance
- **Error Handling**: Graceful degradation when data is unavailable
- **Responsive Design**: Works on desktop and mobile devices
- **Fast Trend Lines**: Cached Matrix Pencil analysis with full 4318-sample processing, cubic splines under 150ms
- **Automatic Tidal Analysis**: Matrix Pencil multiple frequency detection with comprehensive results table
- **Extended Data Window**: 72-hour dataset with Matrix Pencil analysis provides robust multi-component tidal detection

## 📱 Mobile Support

Both dashboards are optimized for mobile viewing with:
- Responsive layouts that adapt to screen size
- Touch-friendly interactions
- Simplified legends for smaller screens
- Fast loading times

## 🔄 Data Updates

- **Sensor readings**: Every minute
- **Dashboard refresh**: Every 2 minutes
- **Data retention**: 72 hours visible (4318 readings) with Matrix Pencil analysis on full dataset
- **Model validation**: Analysis Error plot shows reconstruction accuracy in real-time
- **Timezone**: Eastern Time (America/New_York)

## 🤝 Contributing

This project welcomes contributions! Areas for improvement:
- Additional wave analysis algorithms
- Enhanced mobile features
- Data export capabilities
- Historical data analysis
- Alert systems for unusual readings

## 📄 License

This project is open source. Please check the repository for license details.

## 🆘 Support

For technical questions or issues:
- Check the debug dashboard for system diagnostics
- Review the browser console for error messages
- Ensure stable internet connection for real-time updates

---

**Live System Status**: Check the debug dashboard to view current sensor readings and system health.
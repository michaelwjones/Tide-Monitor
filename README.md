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
- **Default 24-hour view** with zoom capability to 72 hours for comprehensive analysis
- **Intuitive time controls**: Dedicated 24-hour and 72-hour buttons for quick navigation
- **Advanced zoom functionality**: Click-and-drag selection and Ctrl+mouse wheel for precise control
- **Multi-axis charts** for detailed analysis with extended timeframe
- **Enhanced visualization** of wave calculation methods
- **System diagnostics** with valid sample tracking
- **Advanced trend line analysis** with dual methodologies:
  - **Water Level Harmonics**: FFT-detected tidal pattern fitting with automatic period detection
  - **Wind/Wave Splines**: 30-point smoothed cubic spline interpolation
- **Automatic tidal frequency analysis** with comprehensive results table
  - **Immediate chart display**: Chart loads instantly when data arrives
  - **Background analysis**: FFT analysis runs asynchronously for smooth user experience
  - **Auto-updating**: Analysis refreshes every 2 minutes with new data
- **NOAA environmental data** integration (wind, water level)
- **Wind range**: 0-40 knots for comprehensive weather tracking
- **Real-time tidal constituent identification** (M2, S2, O1, K1, etc.)

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

#### Water Level Harmonics
- **Mathematical Form**: `f(t) = a0 + a1*sin(a2*t + a3)`
- **Coefficients**: 4 parameters (DC offset, amplitude, frequency, phase)
- **FFT-Based Detection**: Automatic tidal period discovery from 72 hours of data
- **Frequency Range**: 6-48 hour periods with 30-minute resolution
- **Pattern Recognition**: Detects dominant tidal constituents automatically
- **Least Squares Fitting**: Robust amplitude and phase calculation
- **Tidal Classification**: Automatic identification of diurnal, semi-diurnal, and harmonic components

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
  - **Water Levels**: FFT-based harmonic analysis with automatic tidal detection
  - **Wind/Wave Data**: Enhanced 30-point smoothed cubic splines
- **Performance Optimized**: O(n) complexity using Thomas algorithm for fast rendering
- **Edge-Safe**: Natural splines prevent oscillation artifacts at data boundaries
- **Toggle Control**: Single button controls both trend line methodologies
- **Visual Distinction**: Harmonics use thick dashed lines, splines use standard dashing

### Security & Performance
- **Content Security Policy**: Protection against XSS attacks
- **Optimized Rendering**: Chart animations disabled for performance
- **Error Handling**: Graceful degradation when data is unavailable
- **Responsive Design**: Works on desktop and mobile devices
- **Fast Trend Lines**: Harmonic computation under 100ms, cubic splines under 150ms for 4320 data points
- **Automatic Tidal Analysis**: FFT-based period detection with comprehensive frequency table
- **Extended Data Window**: 72-hour analysis provides 3+ tidal cycles for robust frequency detection

## 📱 Mobile Support

Both dashboards are optimized for mobile viewing with:
- Responsive layouts that adapt to screen size
- Touch-friendly interactions
- Simplified legends for smaller screens
- Fast loading times

## 🔄 Data Updates

- **Sensor readings**: Every minute
- **Dashboard refresh**: Every 2 minutes
- **Data retention**: 72 hours visible (4320 readings) for comprehensive analysis
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
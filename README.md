# Tide Monitor 🌊

A real-time water level and wave height monitoring system using ultrasonic sensors and IoT technology.

## 🔗 Live Dashboard

**Main Dashboard:** [https://michaelwjones.github.io/Tide-Monitor/](https://michaelwjones.github.io/Tide-Monitor/)  
**Debug Dashboard:** [https://michaelwjones.github.io/Tide-Monitor/debug/](https://michaelwjones.github.io/Tide-Monitor/debug/)

## 📊 What It Does

This system continuously monitors water levels and wave activity using an ultrasonic sensor mounted 8 feet above the water surface. It provides real-time data visualization through web dashboards that update every 2 minutes.

### Key Measurements
- **Water Level**: Distance from sensor to water surface (converted to feet)
- **Wave Heights**: Two different calculation methods for wave analysis
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
- **Wave height visualization** using two different analysis methods
- **Reference line** at 2.5 feet for context
- **Auto-refresh** every 2 minutes
- **Mobile responsive** design

### Debug Dashboard  
- **All data fields** from the sensor system
- **Multi-axis charts** for detailed analysis
- **Enhanced visualization** of wave calculation methods
- **System diagnostics** with valid sample tracking

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
  "vs": 487
}
```

- `t`: Timestamp (ISO8601)
- `w`: Water level in mm (average method)
- `hp/he`: Wave heights using percentile/envelope methods
- `wp/we`: Water levels using percentile/envelope methods  
- `vs`: Valid sample count (out of 512)

## 🔧 Technical Details

### Wave Analysis Methods
1. **Percentile Method**: Uses statistical percentiles to calculate wave heights
2. **Envelope Method**: Analyzes the envelope of the wave signal

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

### Security & Performance
- **Content Security Policy**: Protection against XSS attacks
- **Optimized Rendering**: Chart animations disabled for performance
- **Error Handling**: Graceful degradation when data is unavailable
- **Responsive Design**: Works on desktop and mobile devices

## 📱 Mobile Support

Both dashboards are optimized for mobile viewing with:
- Responsive layouts that adapt to screen size
- Touch-friendly interactions
- Simplified legends for smaller screens
- Fast loading times

## 🔄 Data Updates

- **Sensor readings**: Every minute
- **Dashboard refresh**: Every 2 minutes
- **Data retention**: 24 hours visible (1440 readings)
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
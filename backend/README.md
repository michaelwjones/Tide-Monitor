# Backend Architecture

The backend consists of embedded firmware and cloud-based data processing systems.

## Components

### Embedded Firmware (`boron404x/`)
- **Hardware**: Particle Boron 404X with HRXL-MaxSonar MB7360 ultrasonic sensor
- **Firmware**: `tide-monitor-analog.ino` - Arduino C++ code
- **Deployment**: Run `flash.bat` to deploy firmware to device

**Key Features**:
- 512 samples per minute for noise reduction
- Two wave analysis methods (percentile, envelope)  
- Offline storage for 1000 readings (~16 hours)
- Automatic cloud sync when connectivity restored

### Cloud Functions (`firebase-functions/`)
- **Data Enrichment**: Automatic NOAA environmental data integration
- **ML Forecasting**: LSTM and Transformer neural networks for 24-hour predictions
- **Tidal Analysis**: Matrix Pencil signal processing for trend analysis
- **Deployment**: See `/firebase-functions/README.md` for detailed setup

### Integration (`particle.io/`)
- **Webhook Config**: `firebase integration.txt` - Particle Cloud to Firebase integration
- **Data Flow**: Particle Cloud → Firebase Realtime Database → Cloud Functions

## Data Schema

**Sensor Data**: `t` (timestamp), `w` (water level), `hp`/`he` (wave heights), `wp`/`we` (water levels), `vs` (valid samples)

**NOAA Data**: `ws` (wind speed), `wd` (wind direction), `gs` (gust speed), `wm` (water level from Duke Marine Lab)

**Error Handling**: NOAA fields = -999 when APIs fail or validation errors occur

## Development Workflow

1. **Firmware Development**: Use Particle Workbench, deploy with `flash.bat`
2. **Cloud Functions**: See `/firebase-functions/README.md` for deployment procedures
3. **Testing**: Monitor via Firebase console and dashboard error logs

For component-specific details, see README files in respective subdirectories.
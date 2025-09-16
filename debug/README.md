# Debug Dashboard

Advanced monitoring interface with detailed visualizations and wind analysis features.

## Features

### Core Visualizations
- **Water Level Tracking**: Real-time sensor measurements in feet
- **Wave Height Analysis**: Percentile-based wave height measurements  
- **Wind Speed Monitoring**: NOAA wind speed data in knots
- **Duke Marine Lab Data**: Official water level reference data
- **Reference Line**: 3-foot reference marker for context

### Wind Direction Indicator
- **Visual Arrow**: Black arrow extending from wind speed plot endpoint
- **Precision**: Rounds to nearest 22.5° (16 compass directions: N, NNE, NE, etc.)
- **Real-time Updates**: Updates every 2 minutes with chart refresh
- **Smart Positioning**: Arrow base starts exactly at latest wind speed data point

### Automatic Forecasting
- **Transformer Models**: Displays 24-hour water level forecasts when available
- **Auto-Display**: Shows forecast automatically if generated within last 10 minutes
- **Visual Integration**: Purple dots distinguish forecast from historical data
- **Timeline Extension**: Chart extends 24 hours into future when forecast present

### Time Range Controls
- **24 Hours**: Focus on recent trends and immediate forecast
- **72 Hours**: Extended view for pattern analysis
- **Smart Scaling**: Automatically adjusts timeline based on forecast availability

## Data Sources

- **Sensor Data**: Ultrasonic measurements from Particle Boron 404X
- **NOAA Environmental**: Wind speed, direction, gusts from Duke Marine Lab station
- **ML Forecasts**: Transformer v1/v2 models for 24-hour predictions
- **Reference Data**: Duke Marine Lab official water levels

## Usage

1. **Access**: [Debug Dashboard](https://michaelwjones.github.io/Tide-Monitor/debug/)
2. **Navigation**: Use time range buttons to adjust view
3. **Wind Analysis**: Check arrow direction for current wind conditions
4. **Forecast Review**: Purple dots show upcoming predictions when available

## Technical Details

- **Refresh Rate**: Auto-refresh every 2 minutes
- **Data Range**: Last 72 hours available, default 24-hour view
- **Forecast Logic**: Only shows predictions generated within 10 minutes
- **Wind Rounding**: 22.5° increments for clear directional indication

## Related Documentation

- [Main Dashboard](../README.md) - Project overview and basic dashboard
- [Backend Architecture](../backend/README.md) - Hardware and cloud functions
- [Tidal Analysis](../backend/firebase-functions/tidal-analysis/README.md) - ML forecasting methods
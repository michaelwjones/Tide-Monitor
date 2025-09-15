# Firebase Cloud Functions

Cloud-based data processing for the Tide Monitor system.

## Function Types

### Data Enrichment (`tide-enrichment/`)
- **Purpose**: Enriches sensor readings with NOAA environmental data  
- **Trigger**: Automatic when new readings arrive
- **Source**: NOAA station 8656483 (Duke Marine Lab)
- **Added Data**: Wind speed/direction/gust, water level

### Tidal Analysis (`tidal-analysis/functions/`)
- **Matrix Pencil v1**: Harmonic analysis every 5 minutes
- **LSTM v1**: 24-hour forecasting every 6 hours  
- **Transformer v1**: 24-hour forecasting every 5 minutes
- **Results**: Stored in Firebase for dashboard consumption

For detailed information on specific analysis methods, see `/tidal-analysis/README.md` and method-specific README files.

## Data Schema

**Sensor Fields**: `t`, `w`, `hp`, `he`, `wp`, `we`, `vs` (from Particle device)
**NOAA Fields**: `ws`, `wd`, `gs`, `wm` (added by enrichment function)

**Error Values**: -999 indicates NOAA API failures or validation errors

## Deployment

**Quick Deploy (Batch Files)**:
- `deploy-enrichment.bat` - Safe, always deployable
- `deploy-matrix-pencil-v1.bat` - Cost-controlled analysis
- Method-specific deploy scripts in respective directories

**Manual Deploy**:
```bash
firebase deploy --only functions --source tide-enrichment
firebase deploy --only functions --source tidal-analysis/functions/[method]/[version]
```

**Prerequisites**: Run `npm install` in each function directory before first deployment

## Development Commands

- **Logs**: `firebase functions:log`
- **Local Testing**: `firebase emulators:start --only functions`
- **Toggle Analysis**: `toggle-matrix-pencil-v1.bat`

For detailed deployment procedures and troubleshooting, see method-specific README files.
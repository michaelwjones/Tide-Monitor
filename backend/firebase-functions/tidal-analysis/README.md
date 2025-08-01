# Tidal Analysis Firebase Functions

This directory contains Firebase Cloud Functions for advanced tidal harmonic analysis using the Matrix Pencil v1 methodology.

## Quick Setup

### 1. Install Dependencies
```bash
cd tidal-analysis
npm install
```

### 2. Enable Analysis (Important!)
The analysis is **disabled by default**. To enable:

Set the environment variable:
```bash
# From the firebase-functions directory
firebase functions:config:set tidal.analysis.enabled=true
```

**Note**: You control when analysis runs by setting this variable. This prevents unnecessary computation costs.

### 3. Deploy Function

**Option A: Using Batch File (Recommended)**
```batch
# Interactive deployment with cost control
deploy-tidal-analysis.bat
```

**Option B: Command Line**
```bash
# From the firebase-functions directory
firebase deploy --only functions --source tidal-analysis
```

### 4. Set up Cloud Scheduler
The function automatically runs every 5 minutes when enabled. Cloud Scheduler will be created automatically on first deployment.

## Usage

### Enable Analysis

**Option A: Using Batch File**
```batch
# Run interactive deployment and choose "Deploy with analysis ENABLED"
deploy-tidal-analysis.bat
```

**Option B: Command Line**
```bash
firebase functions:config:set tidal.analysis.enabled=true
firebase deploy --only functions --source tidal-analysis
```

### Disable Analysis

**Option A: Using Batch File**
```batch  
# Run interactive deployment and choose "Deploy with analysis DISABLED"
deploy-tidal-analysis.bat
```

**Option B: Command Line**
```bash
firebase functions:config:set tidal.analysis.enabled=false
firebase deploy --only functions --source tidal-analysis
```

### Monitor Logs
```bash
firebase functions:log --only runTidalAnalysis
```

### Manual Trigger (Testing)
```bash
# Using Firebase CLI
firebase functions:shell
> triggerTidalAnalysis()
```

## Function Details

### `runTidalAnalysis`
- **Trigger**: Cloud Scheduler every 5 minutes
- **Purpose**: Performs Matrix Pencil tidal analysis on last 72 hours of data
- **Data Source**: `/readings/` (last 4320 entries)
- **Output**: Stores results in `/tidal-analysis/`
- **Error Handling**: Stores errors in `/tidal-analysis-error/`

### `triggerTidalAnalysis`
- **Trigger**: HTTP callable function
- **Purpose**: Manual analysis trigger for testing
- **Returns**: Success/error status

## Data Flow

```
Sensor Data → Firebase /readings/ → Cloud Function → Matrix Pencil Analysis → /tidal-analysis/
                                                                              ↓
Debug Dashboard ← Fetch Results ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

## Analysis Results

Results are stored in Firebase at `/tidal-analysis/` with structure:
```json
{
  "methodology": "matrix-pencil-v1",
  "timestamp": "2025-08-01T10:30:00.000Z",
  "computationTimeMs": 45230,
  "dataPoints": 4318,
  "timeSpanHours": "71.9",
  "components": [
    {
      "frequency": 0.000140845,
      "periodMs": 44628000,
      "periodHours": 12.397,
      "damping": -1.2e-6,
      "amplitude": 0.2847,
      "phase": 1.832,
      "power": 0.0811
    }
  ],
  "dcComponent": 2.438,
  "modelOrder": 4,
  "pencilParam": 2159
}
```

## Cost Management

### When Analysis is ENABLED:
- Runs every 5 minutes = 288 executions/day
- ~45 seconds average execution time
- Estimated cost: ~$5-15/month (varies by region)

### When Analysis is DISABLED:
- Function still deployed but doesn't execute
- No computation costs
- Minimal storage costs for existing results

### Best Practices:
1. **Enable only when needed** for testing/monitoring
2. **Monitor logs** for computation time trends  
3. **Disable during development** to avoid unnecessary costs
4. **Use manual trigger** for one-off analysis

## Troubleshooting

### Function Not Running
1. Check if analysis is enabled: `firebase functions:config:get`
2. Verify Cloud Scheduler job exists in Google Cloud Console
3. Check function logs: `firebase functions:log`

### Analysis Failures
1. Check `/tidal-analysis-error/` in Firebase database
2. Verify sufficient data in `/readings/` (need 100+ entries)
3. Review function logs for detailed error messages

### Performance Issues
1. Monitor `computationTimeMs` in results
2. Check memory usage in Cloud Console
3. Consider reducing model complexity if timeouts occur

### No Results in Debug Dashboard
1. Verify analysis is enabled and running
2. Check if `/tidal-analysis/` exists in Firebase
3. Ensure debug dashboard can fetch from Firebase (CORS/permissions)

## Development

### Local Testing
```bash
firebase emulators:start --only functions
# Test in browser at http://localhost:5001
```

### Function Structure
- `index.js` - Main function implementation
- `package.json` - Dependencies and metadata
- `README.md` - This file

### Key Dependencies
- `firebase-admin` - Firebase database access
- `firebase-functions` - Cloud Functions runtime

## Security

- Functions use Firebase Admin SDK with automatic service account
- No external API calls (uses existing Firebase data)
- Results stored in same database as sensor data
- Cloud Scheduler uses internal Google authentication

## Monitoring

Monitor function health via:
1. **Firebase Console**: Function logs and metrics
2. **Google Cloud Console**: Cloud Scheduler job status
3. **Database**: Check timestamp of latest results
4. **Debug Dashboard**: Verify analysis table updates
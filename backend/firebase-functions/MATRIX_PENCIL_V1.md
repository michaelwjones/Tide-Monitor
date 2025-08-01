# Matrix Pencil v1 - Tidal Analysis Methodology

**Version**: matrix-pencil-v1  
**Implementation Date**: 2025-08-01  
**Location**: Firebase Cloud Functions - `tidal-analysis/index.js`

## Overview

Matrix Pencil v1 is an advanced signal processing method implemented for non-periodic tidal harmonic analysis. Unlike traditional FFT-based approaches that assume periodicity, Matrix Pencil can handle real-world tidal variations including weather effects, datum shifts, and irregular patterns.

## Mathematical Foundation

The Matrix Pencil method decomposes signals into complex exponentials:

```
f(t) = Σ Aₖ e^(σₖ + jωₖ)t
```

Where:
- `Aₖ` = Complex amplitude 
- `σₖ` = Damping coefficient (real part)
- `ωₖ` = Angular frequency (imaginary part)
- `t` = Time

This formulation naturally handles:
- **Non-periodic signals** (no periodicity assumptions)
- **Damped oscillations** (growing/decaying amplitudes)
- **Multiple frequency components** (up to 8 simultaneous)
- **Phase relationships** (timing information)

## Algorithm Details

### 1. Data Preprocessing
- **Input**: 72 hours of water level data (~4320 samples at 1-minute intervals)
- **Filtering**: Remove invalid readings outside 300-5000mm range
- **DC Removal**: Subtract mean to focus on oscillatory components
- **Units**: Convert from mm to feet (divide by 304.8)

### 2. Hankel Matrix Construction
Creates structured matrices from time series data:
```
Y₀ = [y(0)   y(1)   ... y(L-1)  ]     Y₁ = [y(1)   y(2)   ... y(L)    ]
     [y(1)   y(2)   ... y(L)    ]          [y(2)   y(3)   ... y(L+1)  ]
     [⋮      ⋮      ⋱   ⋮       ]          [⋮      ⋮      ⋱   ⋮       ]
     [y(M-1) y(M)  ... y(N-1)   ]          [y(M)   y(M+1) ... y(N)    ]
```

Where:
- `L` = Pencil parameter (up to N/2, typically N/3)
- `M` = N - L + 1 (number of rows)
- `N` = Total data points

### 3. SVD-Based Model Order Selection
- Perform Singular Value Decomposition on Y₀
- **Threshold**: Keep components > 1% of maximum singular value
- **Model Order**: Automatically determined (typically 2-8 components)
- **Robustness**: SVD handles noisy data and overdetermined systems

### 4. Generalized Eigenvalue Problem
Solve: `V₁z = λV₂z`

Where V₁, V₂ are submatrices of the right singular vectors, and eigenvalues λ encode:
- **Frequency**: `ω = arg(λ) / Δt`
- **Damping**: `σ = ln(|λ|) / Δt`
- **Stability**: |λ| < 1 for decaying, |λ| > 1 for growing

### 5. Parameter Estimation
For each eigenvalue λₖ:
- **Frequency**: `fₖ = |atan2(imag(λₖ), real(λₖ))| / (2π * Δt)`
- **Period**: `Tₖ = 1 / fₖ`
- **Damping**: `σₖ = ln(|λₖ|) / Δt`
- **Amplitude & Phase**: Least-squares fitting to original data

### 6. Conjugate Pair Filtering
- Process only eigenvalues with positive imaginary parts
- Avoids duplicate frequencies from complex conjugate pairs
- Ensures unique tidal component identification

## Expected Tidal Components

Matrix Pencil v1 typically detects these constituents:

| Component | Period | Description |
|-----------|--------|-------------|
| **M2** | ~12.42h | Principal lunar semi-diurnal |
| **S2** | ~12.00h | Principal solar semi-diurnal |
| **O1** | ~25.82h | Lunar diurnal |
| **K1** | ~23.93h | Lunar/solar diurnal |
| **N2** | ~12.66h | Lunar elliptic semi-diurnal |
| **P1** | ~24.07h | Solar diurnal |

## Implementation Parameters

### Cloud Function Configuration
- **Runtime**: Node.js 22
- **Memory**: Default (256MB, auto-scales as needed)
- **Timeout**: 540 seconds (9 minutes max)
- **Schedule**: Every 5 minutes via Cloud Scheduler
- **Trigger**: `onSchedule('*/5 * * * *')`

### Matrix Pencil Parameters
- **Data Window**: 4320 samples (72 hours)
- **Pencil Parameter L**: Up to N/2 (typically ~2160)
- **Model Order**: 2-8 (SVD threshold: 1%)
- **SVD Iterations**: 50 max, tolerance 1e-6
- **QR Iterations**: 20 max, tolerance 1e-4

### Performance Characteristics
- **Computation Time**: ~10-60 seconds (varies with data complexity)
- **Memory Usage**: ~50-100MB peak
- **Data Processing**: 4320 samples × 8 components = ~34K operations
- **Numerical Stability**: Double precision, condition number checking

## Data Storage Format

Results stored in Firebase Realtime Database at `/tidal-analysis/`:

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

## Error Handling

### Computation Failures
- **Matrix Singularity**: Automatic condition number checking
- **SVD Failure**: Fallback to reduced model order
- **Eigenvalue Issues**: Robust QR algorithm with multiple iterations
- **Numerical Overflow**: Input validation and range checking

### API Integration
- **Firebase Connection**: Retry logic and timeout handling
- **Data Validation**: Required field checking and type validation
- **Storage Failures**: Atomic updates with rollback capability
- **Network Issues**: Automatic Cloud Function retry

### Monitoring
- **Console Logging**: Detailed computation progress and timing
- **Error Storage**: Failed analyses stored in `/tidal-analysis-error/`
- **Performance Metrics**: Computation time and data point tracking
- **Result Validation**: Component count and frequency range checks

## Usage Instructions

### Enable/Disable Analysis
Set environment variable in Firebase Functions:
```bash
firebase functions:config:set tidal.analysis.enabled=true
# or
firebase functions:config:set tidal.analysis.enabled=false
```

### Deploy Function
```bash
cd backend/firebase-functions
firebase deploy --only functions:tidal-analysis
```

### Monitor Execution
```bash
firebase functions:log --only tidal-analysis
```

### Manual Trigger (Testing)
```javascript
// Call via Firebase SDK
const functions = firebase.functions();
const triggerAnalysis = functions.httpsCallable('triggerTidalAnalysis');
const result = await triggerAnalysis();
```

## Future Enhancements

### Version 2 Considerations
- **Adaptive Windowing**: Variable analysis periods based on data quality
- **Multi-Station Analysis**: Combine multiple NOAA stations
- **Machine Learning**: Hybrid ML-Matrix Pencil approach
- **Real-Time Processing**: Streaming analysis for immediate updates
- **Enhanced Constituents**: Detection of shallow water harmonics

### Performance Optimization
- **Parallel Processing**: Multi-threaded SVD computation
- **Caching**: Incremental updates for overlapping windows
- **Memory Management**: Streaming data processing
- **GPU Acceleration**: Cloud Function GPU instances

## References

- Hua, Y. & Sarkar, T.K. (1990). "Matrix Pencil Method for Estimating Parameters of Exponentially Damped/Undamped Sinusoids in Noise"
- Roy, R. & Kailath, T. (1989). "ESPRIT-Estimation of Signal Parameters via Rotational Invariance Techniques"
- Sarkar, T.K. & Pereira, O. (1995). "Using the Matrix Pencil Method to Estimate the Parameters of a Sum of Complex Exponentials"
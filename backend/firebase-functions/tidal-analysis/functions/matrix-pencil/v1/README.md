# Matrix Pencil v1 Tidal Analysis

This directory contains the Matrix Pencil v1 implementation for advanced tidal harmonic analysis.

## Quick Setup

### 1. Install Dependencies
```bash
cd tidal-analysis/functions/matrix-pencil/v1
npm install
```

### 2. Configure Analysis
Edit `.env` file:
```env
TIDAL_ANALYSIS_ENABLED=true   # Enable analysis (costs money)
TIDAL_ANALYSIS_ENABLED=false  # Disable analysis (no costs)
```

### 3. Deploy Function
```bash
# From firebase-functions directory
deploy-matrix-pencil-v1.bat
```

## Analysis Method

Matrix Pencil v1 uses complex exponential decomposition to identify tidal constituents:
- **Signal model**: Σ Aₖ e^(σₖ + jωₖ)t  
- **Parameter estimation**: SVD-based model order selection
- **Non-periodic**: Handles real-world tidal variations without periodicity assumptions

## Results Storage

Analysis results are stored in Firebase at `/tidal-analysis/matrix-pencil-v1/` with:
- Detected tidal frequencies and periods
- Component amplitudes and phases  
- Damping factors and power levels
- Model order and computation metrics

## Cost Management

- **Enabled**: ~$5-15/month (runs every 5 minutes)
- **Disabled**: No computation costs
- **Control**: Use `.env` file to enable/disable
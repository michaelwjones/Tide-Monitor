# Matrix Pencil v2 Tidal Analysis

Enhanced Matrix Pencil implementation with significantly improved accuracy and frequency resolution.

## Key Improvements over v1

### Algorithm Enhancements
- **Larger Pencil Parameter L**: Uses 2/3 of data length (vs 1/3 in v1) for better frequency resolution
- **Higher SVD Rank Limit**: Processes up to 20 singular values (vs 5 in v1) for better model order selection
- **More Sensitive Threshold**: Uses 0.1% threshold (vs 1% in v1) for detecting weaker tidal components
- **Enhanced Precision**: Higher precision SVD computation with 1e-8 tolerance (vs 1e-6 in v1)
- **More Components**: Supports up to 16 signal components (vs 8 in v1)

### Computational Improvements
- **Better Eigenvalue Solver**: Enhanced generalized eigenvalue computation with improved numerical stability
- **Enhanced Deflation**: More robust matrix deflation for finding multiple eigenvalues
- **Improved Filtering**: Better criteria for selecting physically meaningful tidal frequencies (1-50 hour periods)

## Expected Performance

### Accuracy Improvements
- **Better Frequency Resolution**: Can distinguish closely spaced tidal constituents (e.g., M2 vs S2)
- **More Components**: Typically finds 3-6 components instead of 1-2
- **Reduced Single-Frequency Issues**: The 90% → 10% problem should be significantly reduced

### Computational Cost
- **Higher CPU Usage**: ~2-3x more computation time due to larger L and enhanced SVD
- **Memory Usage**: Increased memory requirements for larger matrices
- **Still Cost-Effective**: Expected ~$10-25/month when enabled (vs ~$5-15 for v1)

## Quick Setup

### 1. Install Dependencies
```bash
cd tidal-analysis/functions/matrix-pencil/v2
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
deploy-matrix-pencil-v2.bat
```

## Analysis Method

Matrix Pencil v2 uses enhanced complex exponential decomposition:
- **Signal model**: Σ Aₖ e^(σₖ + jωₖ)t with improved parameter estimation
- **Enhanced SVD**: Higher rank decomposition with better precision
- **Larger L**: 2/3 data length for superior frequency resolution
- **Non-periodic**: Handles real-world tidal variations without periodicity assumptions

## Results Storage

Analysis results are stored in Firebase at `/tidal-analysis/matrix-pencil-v2/` with:
- Enhanced frequency detection (typically 3-6 components)
- Improved period accuracy for major tidal constituents
- Better amplitude and phase estimation
- Enhanced damping factor computation

## Cost Management

- **Enabled**: ~$10-25/month (enhanced computation, runs every 5 minutes)
- **Disabled**: No computation costs
- **Control**: Use `.env` file to enable/disable
- **Monitoring**: Check Firebase Console for computation time trends

## Comparison with v1

| Feature | v1 | v2 | Improvement |
|---------|----|----|-------------|
| Pencil Parameter L | sn/3 | 2*sn/3 | 2x larger |
| SVD Rank Limit | 5 | 20 | 4x more singular values |
| Threshold | 1% | 0.1% | 10x more sensitive |
| Max Components | 8 | 16 | 2x more components |
| Precision | 1e-6 | 1e-8 | 100x more precise |
| Typical Components Found | 1-2 | 3-6 | 2-3x more |

## Expected Tidal Constituents

With enhanced resolution, v2 should reliably detect:
- **M2** (12.42h): Principal lunar semi-diurnal
- **S2** (12.00h): Principal solar semi-diurnal  
- **K1** (23.93h): Lunar diurnal
- **O1** (25.82h): Principal lunar diurnal
- **Additional constituents**: N2, K2, P1, Q1 depending on location

## Troubleshooting

### High Computation Time
- Monitor `computationTimeMs` in Firebase results
- Typical v2 computation: 60-180 seconds (vs 20-60s for v1)
- If >300s consistently, consider reducing data points

### Too Many/Few Components
- Adjust threshold in code if needed (currently 0.1%)
- Check `modelOrder` in results
- Enhanced algorithm should be much more stable than v1

### No Improvement over v1
- Ensure sufficient data (need 100+ points)
- Check that L parameter is actually larger in logs
- Verify SVD rank limit increase in console output
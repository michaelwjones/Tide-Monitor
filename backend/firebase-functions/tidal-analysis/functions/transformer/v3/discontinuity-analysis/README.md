# Transformer v2 Discontinuity Analysis

Single-purpose tool for debugging discontinuities in transformer v2 tidal predictions.

## Problem Statement

Transformer v2 produces smooth predictions 97% of the time, but ~3% have discontinuities:
- Wrong starting heights (level jumps)
- Cusps with incorrect tidal direction
- Phase errors in tidal cycles

## Quick Start

```bash
# Analyze training data quality
python analyze.py --data training

# Test for inference discontinuities  
python analyze.py --data inference --num-tests 100

# Specific training analysis
python analyze.py --data training --training-analysis continuity sequences

# Specific inference analysis
python analyze.py --data inference --inference-analysis level_jumps direction_changes
```

## Analysis Options

### Training Data Analysis
Validates training data quality to rule out data issues.

**Available analysis types:**
- `continuity` - Gap analysis between input/output sequences
- `direction` - Direction consistency at sequence boundaries  
- `statistics` - Basic dataset statistics
- `sequences` - Identify worst sequences for manual inspection

**Example:**
```bash
python analyze.py --data training --training-analysis continuity direction
```

### Inference Analysis
Tests live inference for discontinuities using Firebase-identical processing.

**Available analysis types:**
- `level_jumps` - Significant height differences at prediction start
- `direction_changes` - Wrong tidal direction (rising vs falling)
- `cusps` - Correct start level but wrong tidal phase
- `phase_errors` - Temporal misalignment in tidal cycles

**Example:**
```bash
python analyze.py --data inference --inference-analysis level_jumps cusps --num-tests 200
```

## Key Findings

**Training Data Quality (12,126 sequences):**
- 97.43% have excellent continuity (≤10mm gaps)
- 99.97% have good continuity (≤20mm gaps)
- Only 1 sequence (0.008%) has gap >50mm
- **Conclusion**: Training data is healthy, discontinuities are model/inference issues

**Sequences to Manually Inspect:**
- Sequence 436: 83.0mm gap (type "437" in testing interface)
- Sequence 4359: 29.0mm gap (type "4360")  
- Sequence 6441: 23.0mm gap (type "6442")
- Sequence 276: 22.0mm gap (type "277")
- Sequence 2538: 20.0mm gap (type "2539")

## Directory Structure

```
discontinuity-analysis/
├── analyze.py                    # Single entry point tool
├── README.md                     # This documentation
├── core/                         # Core detection modules
│   ├── discontinuity_detector.py      # 4 detection algorithms
│   └── __init__.py
└── results/                      # Analysis outputs
    ├── sequences_to_inspect.json       # Suspect sequences
    ├── discontinuity_results.json      # Inference test results
    └── model_analysis_results/         # Training data plots
```

## Detection Algorithms

**Level Jumps**: Detect significant height differences between input end and prediction start
**Direction Reversals**: Detect when prediction trends opposite to input trend
**Cusps**: Detect correct start level but wrong tidal phase/direction
**Phase Errors**: Detect temporal misalignment in tidal harmonic patterns

## Firebase Integration

Uses shared data packaging tool (`../shared/data_packaging.py`) for exact Firebase processing logic:
- ±5 minute tolerance for timestamp matching
- -999 value handling for missing data  
- Binary search optimization for performance
- Identical normalization and sequence creation
- Automatic fresh data fetching when needed

## Output

**Training Analysis**: Console output with statistics and suspect sequence list
**Inference Analysis**: JSON file with detailed discontinuity results and console summary

## Analysis Results

### Training Data Analysis (Complete)
**Status**: ✅ Completed successfully  
**Result**: Excellent data quality - 97.43% sequences have ≤10mm gaps  
**Conclusion**: Training data is healthy, discontinuities are model/inference issues

### Inference Analysis (Working with Shared Data Tool)
**Status**: ✅ Now operational with shared data packaging tool  
**Update**: Uses `../shared/data_packaging.py` for Firebase-identical processing  
**Features**: Automatic fresh data fetching, consistent normalization handling

## Key Findings

1. **Training data quality**: Excellent (97.4% good continuity)
2. **Shared tool integration**: Now uses consistent data packaging across all components
3. **Normalization issue discovered**: Input sequences properly normalized (~-0.4) but predictions raw mm values (~783mm)
4. **Model status**: Loads successfully (26.6M parameters, RMSE 0.26)

## Next Steps

1. ✅ **Training analysis complete** - Data quality confirmed excellent
2. ✅ **Shared tool integration** - Now uses consistent data packaging across all transformer v2 components
3. ✅ **Inference analysis operational** - Automatic fresh data fetching enabled
4. ✅ **Manual inspection ready** - Check sequences 437, 4360, 6442, 277, 2539 in testing interface
5. 🔄 **Normalization investigation** - Address ~783mm discontinuities caused by normalization mismatch
6. 🔄 **Model architecture review** - Focus on inference output processing and denormalization
# Transformer v3 Sanity Analysis

Single-purpose tool for model sanity checks and debugging discontinuities in transformer v3 tidal predictions.

## Problem Statement

Transformer v3 produces smooth predictions 97% of the time, but ~3% have discontinuities:
- Wrong starting heights (level jumps)
- Cusps with incorrect tidal direction
- Phase errors in tidal cycles

## Quick Start

```bash
# Analyze training data quality
python analyze.py --data training

# Test for inference sanity
python analyze.py --data inference --num-tests 100

# Specific training analysis
python analyze.py --data training --training-analysis continuity sequences

# Specific inference analysis
python analyze.py --data inference --inference-analysis level_jumps direction_changes
```

## Analysis Options

### Training Data Analysis
Validates training data quality to rule out data issues in the 3-dataset structure.

**Available analysis types:**
- `continuity` - Gap analysis between input/output sequences
- `direction` - Direction consistency at sequence boundaries  
- `statistics` - Basic dataset statistics
- `sequences` - Identify worst sequences for manual inspection

**Example:**
```bash
python analyze.py --data training --training-analysis continuity direction
```

### Sanity Analysis
Tests live inference for discontinuities using pre-generated sanity test dataset.

**Available analysis types:**
- `level_jumps` - Significant height differences at prediction start
- `direction_changes` - Wrong tidal direction (rising vs falling)
- `cusps` - Correct start level but wrong tidal phase
- `phase_errors` - Temporal misalignment in tidal cycles

**Example:**
```bash
python analyze.py --data inference --inference-analysis level_jumps cusps --num-tests 200
```

## v3 Architecture Changes

**Three-dataset structure:**
- Training: 13,388 sequences (main historical data)
- Validation: 3,346 sequences (recent data for validation)
- Sanity: 1,440 sequences (last 4 days, input-only for testing)

**Timestamp-based naming:** Sequences named by YYYYMMDD_HHMM format instead of incremental IDs

**Pre-generated sanity dataset:** Uses X_sanity.npy with consistent test conditions

## Directory Structure

```
sanity-analysis/
├── analyze.py                    # Single entry point tool
├── README.md                     # This documentation
├── core/                         # Core detection modules
│   ├── discontinuity_detector.py      # 4 detection algorithms
│   └── __init__.py
└── results/                      # Analysis outputs
    ├── sequences_to_inspect.json       # Suspect sequences
    ├── sanity_results.json             # Inference test results
    └── model_analysis_results/         # Training data plots
```

## Detection Algorithms

**Level Jumps**: Detect significant height differences between input end and prediction start
**Direction Reversals**: Detect when prediction trends opposite to input trend
**Cusps**: Detect correct start level but wrong tidal phase/direction
**Phase Errors**: Detect temporal misalignment in tidal harmonic patterns

## Data Processing

Uses consistent data processing across v3:
- Timestamp-based sequence identification
- -999 value handling for missing data  
- Pre-generated sanity sequences for consistent testing
- Identical normalization and sequence creation

## Output

**Training Analysis**: Console output with statistics and suspect sequence list
**Sanity Analysis**: JSON file with detailed discontinuity results and console summary

## Analysis Results

### Training Data Analysis
**Status**: Available for v3 datasets
**Features**: Analyzes 3-dataset structure (training/validation/sanity)

### Sanity Analysis (Using Pre-generated Dataset)
**Status**: ✅ Operational with pre-generated sanity sequences
**Update**: Uses X_sanity.npy with 1440 timestamp-based sequences
**Features**: Consistent test conditions, timestamp-based naming

## Key Features

1. **Three-dataset structure**: Training, validation, and sanity test datasets
2. **Timestamp-based naming**: Sequences identified by YYYYMMDD_HHMM format
3. **Pre-generated sanity dataset**: 1440 input-only sequences for consistent testing
4. **Improved data integrity**: -999 handling for missing data points

## Next Steps

1. ✅ **v3 Architecture implemented** - Three-dataset structure with sanity testing
2. ✅ **Sanity dataset generated** - 1440 pre-generated sequences with timestamp naming
3. ✅ **Analysis tools updated** - Renamed from discontinuity-analysis to sanity-analysis
4. 🔄 **Model training** - Train v3 model with new dataset structure
5. 🔄 **Sanity testing** - Comprehensive evaluation with dedicated test dataset
6. 🔄 **Discontinuity investigation** - Focus on actual data discontinuities vs model issues
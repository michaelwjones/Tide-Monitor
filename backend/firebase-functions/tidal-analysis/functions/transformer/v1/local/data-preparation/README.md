# Transformer v1 Data Preparation

Optimized data pipeline for processing Firebase sensor data into training sequences for the Transformer model.

## Overview

This pipeline has been optimized for performance and reliability, using binary search algorithms and timestamp-based matching to efficiently process large datasets.

## Scripts

### `fetch_firebase_data.py`
Downloads and filters sensor data from Firebase Realtime Database.

**Features:**
- **Firebase Integration**: Downloads complete dataset from `https://tide-monitor-boron-default-rtdb.firebaseio.com/readings.json`
- **Quality Filtering**: Removes physically impossible water levels (< -200mm)
- **Dual File Output**: Creates both raw and filtered data files
- **Progress Reporting**: Shows download and filtering statistics

**Output Files:**
- `data/firebase_raw_data.json` - Complete unfiltered data (for reference)
- `data/firebase_filtered_data.json` - Clean data with invalid readings removed

**Usage:**
```bash
python fetch_firebase_data.py
```

**Latest Results:**
- Downloaded: 101,489 total readings
- Filtered: 3,964 invalid readings (< -200mm)
- Kept: 97,525 clean readings (96.1% retention)

### `create_training_data.py`
Generates training sequences using optimized timestamp-based matching.

**Features:**
- **Binary Search Optimization**: O(log n) lookup instead of O(n) linear search
- **Timestamp Matching**: Finds closest readings within ±5 minutes of target times
- **No Synthetic Data**: Works exclusively with real sensor data
- **Temporal Split**: Creates proper train/validation split with temporal gap
- **Progress Reporting**: Shows processing status every 500 sequences

**Processing Pipeline:**
1. **Load Filtered Data**: Reads `firebase_filtered_data.json`
2. **Parse Chronologically**: Sorts 97K+ readings by timestamp
3. **Generate Sequences**: Creates 18K+ overlapping 96-hour sequences
4. **Binary Search Matching**: Maps target times to closest readings
5. **Quality Filtering**: Removes sequences with insufficient coverage
6. **Normalization**: Z-score normalization across all data
7. **Temporal Split**: 80/20 split with gap to prevent data leakage

**Output Files:**
- `data/X_train.npy` - Training input sequences (38.4 MB)
- `data/X_val.npy` - Validation input sequences (5.8 MB) 
- `data/y_train.npy` - Training target sequences (12.8 MB)
- `data/y_val.npy` - Validation target sequences (1.9 MB)
- `data/normalization_params.json` - Mean/std for inference
- `data/metadata.json` - Dataset statistics

**Usage:**
```bash
python create_training_data.py
```

**Latest Results:**
- Source: 97,525 filtered readings
- Generated: 18,346 potential sequences
- Valid: 14,547 sequences (79.3% retention)
- Training: 11,638 sequences / Validation: 1,743 sequences
- Temporal gap: 192.2 hours to prevent data leakage
- Processing time: < 1 minute

## Data Format

### Sequence Structure
- **Input Length**: 433 points (72 hours at 10-minute intervals)
- **Output Length**: 144 points (24 hours at 10-minute intervals)
- **Total Timespan**: 96 hours per sequence

### Timestamp Matching
- **Target Intervals**: Exact 10-minute spacing
- **Tolerance**: ±5 minutes for finding closest readings
- **Quality Control**: Sequences rejected if insufficient coverage

### Normalization
- **Method**: Z-score normalization (mean=0, std=1)
- **Statistics**: Calculated from all real sensor data
- **Parameters**: Saved for inference denormalization

## Performance Optimizations

### Binary Search Algorithm
**Before:** Linear search through all readings for each target time
- Complexity: O(n) where n = ~1000 readings per sequence
- Total operations: ~5.7 billion comparisons

**After:** Binary search with pre-extracted timestamps
- Complexity: O(log n) where n = ~1000 readings per sequence  
- Total operations: ~57 million comparisons
- **Speed improvement: ~100x faster**

### Memory Optimization
- Pre-extract timestamps once per sequence
- Reuse timestamp arrays for multiple searches
- Efficient array operations with NumPy

### Progress Reporting
- Real-time updates during processing
- Percentage completion tracking
- Performance statistics and retention rates

## Quality Assurance

### Data Validation
- **Physical Limits**: Water levels must be ≥ -200mm
- **Timestamp Coverage**: Sequences need readings within ±5 minutes of targets
- **Temporal Separation**: Train/validation split with proper gap

### Error Handling
- **Missing Data**: Sequences rejected if insufficient coverage
- **Invalid Values**: Filtered at source during fetch
- **Type Safety**: Robust parsing with error catching

## Troubleshooting

### Common Issues
1. **File Not Found**: Run `fetch_firebase_data.py` first
2. **Insufficient Data**: Check minimum reading requirements
3. **Memory Issues**: Large datasets may need more RAM
4. **Slow Processing**: Ensure binary search optimization is active

### Performance Tips
- Use SSD storage for faster I/O
- Ensure adequate RAM (8GB+ recommended)
- Monitor progress output for bottlenecks

## Future Enhancements
- **Parallel Processing**: Multi-threaded sequence generation
- **Incremental Updates**: Process only new data
- **Data Compression**: Reduce file sizes for deployment
- **Quality Metrics**: Advanced data validation

## Dependencies
- `numpy` - Array operations and normalization
- `pandas` - Data manipulation (if needed)
- `datetime` - Timestamp parsing and matching
- `json` - Data file handling
- `random` - Sequence offset generation

All dependencies are standard Python libraries or commonly available packages.
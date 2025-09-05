import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

def load_firebase_data():
    """Load raw Firebase data"""
    try:
        with open('data/firebase_raw_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: firebase_raw_data.json not found. Run fetch_firebase_data.py first.")
        return None

def parse_and_sort_readings(raw_data):
    """Parse readings and sort by timestamp, removing wildly out-of-range timestamps"""
    readings = []
    current_year = datetime.now().year
    skipped_timestamps = 0
    
    print("Parsing and filtering readings...")
    
    for key, reading in raw_data.items():
        if 'w' in reading and 't' in reading:
            try:
                timestamp = datetime.fromisoformat(reading['t'].replace('Z', '+00:00'))
                
                # Remove wildly out-of-range timestamps (e.g., years in the future/past)
                if not (2020 <= timestamp.year <= current_year + 1):
                    skipped_timestamps += 1
                    continue
                
                water_level = float(reading['w'])
                
                readings.append({
                    'timestamp': timestamp,
                    'water_level': water_level,
                    'original_data': reading  # Keep original data for enriched output
                })
            except (ValueError, TypeError):
                continue  # Skip invalid readings
    
    # Sort by timestamp
    readings.sort(key=lambda x: x['timestamp'])
    
    print(f"Parsed {len(readings)} valid readings")
    if skipped_timestamps > 0:
        print(f"Skipped {skipped_timestamps} readings with out-of-range timestamps")
    
    return readings

def group_readings_by_day(readings):
    """Group readings by day and identify complete days"""
    days = defaultdict(list)
    
    for reading in readings:
        date_key = reading['timestamp'].date()
        days[date_key].append(reading)
    
    # Sort days
    sorted_days = sorted(days.keys())
    
    if not sorted_days:
        return {}, None, None
    
    # Find first and last complete days (days with some data)
    first_day = sorted_days[0]
    last_day = sorted_days[-1]
    
    # Filter to only include complete days (skip first and last day if they're partial)
    complete_days = {}
    for date_key in sorted_days:
        # Skip first day if it doesn't start at midnight or has less than 12 hours
        if date_key == first_day:
            first_reading = min(days[date_key], key=lambda x: x['timestamp'])
            if (first_reading['timestamp'].hour != 0 or 
                first_reading['timestamp'].minute > 10 or  # Allow some tolerance
                len(days[date_key]) < 720):  # Less than 12 hours of data
                continue
        
        # Skip last day if it's today (might be incomplete)
        if date_key == last_day and date_key == datetime.now().date():
            continue
        
        complete_days[date_key] = days[date_key]
    
    complete_dates = sorted(complete_days.keys())
    first_complete = complete_dates[0] if complete_dates else None
    last_complete = complete_dates[-1] if complete_dates else None
    
    return complete_days, first_complete, last_complete

def find_gaps_and_fill(day_readings, date_key):
    """Find gaps in a day's readings and fill them with -999 water level"""
    # Sort readings by timestamp
    day_readings.sort(key=lambda x: x['timestamp'])
    
    # Generate expected timestamps for the full day (1440 minutes)
    day_start = datetime.combine(date_key, datetime.min.time().replace(tzinfo=day_readings[0]['timestamp'].tzinfo))
    expected_timestamps = [day_start + timedelta(minutes=i) for i in range(1440)]
    
    # Create set of existing timestamps (rounded to nearest minute)
    existing_minutes = set()
    for reading in day_readings:
        # Round to nearest minute
        minute_mark = reading['timestamp'].replace(second=0, microsecond=0)
        existing_minutes.add(minute_mark)
    
    # Find missing timestamps
    missing_timestamps = []
    for expected_ts in expected_timestamps:
        if expected_ts not in existing_minutes:
            missing_timestamps.append(expected_ts)
    
    # Check for gaps larger than 1 hour (60 minutes)
    large_gaps = []
    if missing_timestamps:
        missing_timestamps.sort()
        current_gap_start = missing_timestamps[0]
        current_gap_size = 1
        
        for i in range(1, len(missing_timestamps)):
            if missing_timestamps[i] - missing_timestamps[i-1] == timedelta(minutes=1):
                current_gap_size += 1
            else:
                # End of current gap
                if current_gap_size > 60:  # Gap larger than 1 hour
                    large_gaps.append((current_gap_start, current_gap_size))
                
                # Start new gap
                current_gap_start = missing_timestamps[i]
                current_gap_size = 1
        
        # Check the last gap
        if current_gap_size > 60:
            large_gaps.append((current_gap_start, current_gap_size))
    
    # Calculate target number of readings for the day
    target_readings = 1440
    for gap_start, gap_size in large_gaps:
        target_readings -= gap_size  # Don't fill large gaps
        print(f"  Found large gap on {date_key}: {gap_size} minutes starting at {gap_start.strftime('%H:%M')}")
    
    # Fill smaller gaps only
    filled_readings = day_readings.copy()
    filled_count = 0
    
    for expected_ts in expected_timestamps:
        if expected_ts not in existing_minutes:
            # Check if this timestamp is in a large gap
            in_large_gap = False
            for gap_start, gap_size in large_gaps:
                gap_end = gap_start + timedelta(minutes=gap_size - 1)
                if gap_start <= expected_ts <= gap_end:
                    in_large_gap = True
                    break
            
            if not in_large_gap:
                # Create a synthetic reading with -999 water level
                synthetic_reading = {
                    'timestamp': expected_ts,
                    'water_level': -999,
                    'original_data': {
                        't': expected_ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'w': '-999',
                        'synthetic': True  # Mark as synthetic
                    }
                }
                filled_readings.append(synthetic_reading)
                filled_count += 1
    
    # Sort the final readings
    filled_readings.sort(key=lambda x: x['timestamp'])
    
    return filled_readings, filled_count, len(large_gaps), target_readings

def enrich_firebase_data():
    """Main function to enrich Firebase data with gap filling"""
    print("Transformer v1 Data Enrichment - Gap Filling")
    print("=" * 50)
    
    # Load raw data
    raw_data = load_firebase_data()
    if not raw_data:
        return None
    
    # Parse and sort readings
    readings = parse_and_sort_readings(raw_data)
    if not readings:
        print("No valid readings found")
        return None
    
    print(f"Time range: {readings[0]['timestamp']} to {readings[-1]['timestamp']}")
    
    # Group by day
    days, first_complete, last_complete = group_readings_by_day(readings)
    if not days:
        print("No complete days found")
        return None
    
    print(f"Processing {len(days)} complete days from {first_complete} to {last_complete}")
    
    # Process each day
    enriched_data = {}
    statistics = {
        'total_days_processed': 0,
        'total_original_readings': 0,
        'total_synthetic_readings': 0,
        'days_with_large_gaps': 0,
        'total_large_gaps': 0,
        'daily_stats': []
    }
    
    for date_key in sorted(days.keys()):
        day_readings = days[date_key]
        original_count = len(day_readings)
        
        # Fill gaps for this day
        filled_readings, filled_count, large_gap_count, target_count = find_gaps_and_fill(day_readings, date_key)
        
        # Add enriched readings to output data
        for reading in filled_readings:
            # Generate a key similar to Firebase format
            key = f"enriched_{date_key}_{reading['timestamp'].strftime('%H%M')}"
            enriched_data[key] = reading['original_data']
        
        # Update statistics
        statistics['total_days_processed'] += 1
        statistics['total_original_readings'] += original_count
        statistics['total_synthetic_readings'] += filled_count
        
        if large_gap_count > 0:
            statistics['days_with_large_gaps'] += 1
            statistics['total_large_gaps'] += large_gap_count
        
        daily_stat = {
            'date': date_key.isoformat(),
            'original_readings': original_count,
            'synthetic_readings': filled_count,
            'final_readings': len(filled_readings),
            'target_readings': target_count,
            'large_gaps': large_gap_count,
            'completion_rate': len(filled_readings) / target_count * 100 if target_count > 0 else 0
        }
        statistics['daily_stats'].append(daily_stat)
        
        print(f"  {date_key}: {original_count} -> {len(filled_readings)} readings (+{filled_count} synthetic)")
        if large_gap_count > 0:
            print(f"    {large_gap_count} large gaps found, target adjusted to {target_count}")
    
    # Save enriched data
    os.makedirs('data', exist_ok=True)
    
    with open('data/firebase_enriched_data.json', 'w') as f:
        json.dump(enriched_data, f, indent=2)
    
    print(f"\nEnriched data saved to data/firebase_enriched_data.json")
    print(f"Total readings: {len(enriched_data)}")
    
    # Print detailed statistics
    print(f"\nEnrichment Statistics:")
    print(f"=" * 30)
    print(f"Days processed: {statistics['total_days_processed']}")
    print(f"Original readings: {statistics['total_original_readings']}")
    print(f"Synthetic readings: {statistics['total_synthetic_readings']}")
    print(f"Total enriched readings: {statistics['total_original_readings'] + statistics['total_synthetic_readings']}")
    print(f"Days with large gaps (>1 hour): {statistics['days_with_large_gaps']}")
    print(f"Total large gaps: {statistics['total_large_gaps']}")
    
    if statistics['total_days_processed'] > 0:
        avg_original = statistics['total_original_readings'] / statistics['total_days_processed']
        avg_synthetic = statistics['total_synthetic_readings'] / statistics['total_days_processed']
        print(f"Average original readings per day: {avg_original:.1f}")
        print(f"Average synthetic readings per day: {avg_synthetic:.1f}")
        
        # Distribution of synthetic readings
        synthetic_counts = [stat['synthetic_readings'] for stat in statistics['daily_stats']]
        if synthetic_counts:
            print(f"Synthetic reading distribution:")
            print(f"  Min: {min(synthetic_counts)} per day")
            print(f"  Max: {max(synthetic_counts)} per day")
            print(f"  Average: {sum(synthetic_counts)/len(synthetic_counts):.1f} per day")
    
    # Save statistics
    with open('data/enrichment_statistics.json', 'w') as f:
        # Convert date objects to strings for JSON serialization
        stats_copy = statistics.copy()
        json.dump(stats_copy, f, indent=2, default=str)
    
    print(f"\nStatistics saved to data/enrichment_statistics.json")
    print(f"\nNext step: Run create_training_data.py with the enriched dataset")
    
    return enriched_data

if __name__ == "__main__":
    enrich_firebase_data()
#!/usr/bin/env python3
"""
Fetch real data from Firebase for Transformer testing
Gets exactly 4320 readings (72 hours) of water level data for seq2seq input
"""

import requests
import json
from datetime import datetime, timedelta
import numpy as np

def fetch_firebase_data(hours=72):
    """
    Fetch the most recent data from Firebase
    Returns list of water level readings (w field) in chronological order
    """
    firebase_url = "https://tide-monitor-boron-default-rtdb.firebaseio.com/readings.json"
    
    try:
        print(f"📡 Fetching last {hours} hours from Firebase...")
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_iso = cutoff_time.isoformat()
        
        response = requests.get(firebase_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            raise ValueError("No data received from Firebase")
        
        print(f"📦 Received {len(data)} total readings")
        
        # Filter and sort data
        valid_readings = []
        
        for key, reading in data.items():
            if not isinstance(reading, dict):
                continue
                
            # Check if reading has required fields
            if 't' not in reading or 'w' not in reading:
                continue
                
            try:
                # Parse timestamp
                timestamp_str = reading['t']
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                # Filter by time
                if timestamp < cutoff_time:
                    continue
                
                # Get water level
                water_level = float(reading['w'])
                
                # Basic validation (reasonable water level range)
                if 300 < water_level < 5000:  # Sensor range in mm
                    valid_readings.append({
                        'timestamp': timestamp,
                        'water_level': water_level,
                        'iso_time': timestamp_str
                    })
                    
            except (ValueError, TypeError, KeyError) as e:
                # Skip invalid readings
                continue
        
        if not valid_readings:
            raise ValueError("No valid readings found in the specified time range")
        
        # Sort by timestamp (oldest first)
        valid_readings.sort(key=lambda x: x['timestamp'])
        
        print(f"✅ Found {len(valid_readings)} valid readings in last {hours} hours")
        print(f"📅 Time range: {valid_readings[0]['iso_time']} to {valid_readings[-1]['iso_time']}")
        print(f"🌊 Water level range: {min(r['water_level'] for r in valid_readings):.1f} - {max(r['water_level'] for r in valid_readings):.1f} mm")
        
        # Extract just the water levels
        water_levels = [r['water_level'] for r in valid_readings]
        timestamps = [r['timestamp'] for r in valid_readings]
        
        return water_levels, timestamps, valid_readings
        
    except requests.RequestException as e:
        raise Exception(f"Firebase request failed: {e}")
    except Exception as e:
        raise Exception(f"Data processing failed: {e}")

def get_transformer_input_sequence():
    """
    Get exactly 4320 readings for Transformer seq2seq input (72 hours)
    Transformer requires fixed-length input for proper attention computation
    """
    try:
        water_levels, timestamps, full_data = fetch_firebase_data(hours=72)
        
        target_length = 4320  # 72 hours * 60 minutes = 4320 readings
        
        if len(water_levels) < target_length:
            # Try extending the time range
            print(f"⚠️  Only {len(water_levels)} readings found, extending search...")
            for extended_hours in [96, 120, 168]:  # 4, 5, 7 days
                print(f"🔍 Trying {extended_hours} hours...")
                water_levels, timestamps, full_data = fetch_firebase_data(hours=extended_hours)
                
                if len(water_levels) >= target_length:
                    break
            
            if len(water_levels) < target_length:
                print(f"⚠️  Still insufficient data: {len(water_levels)} readings")
                print(f"🔧 Padding sequence to {target_length} with mean value")
                
                # Pad with mean value at the beginning
                mean_level = np.mean(water_levels)
                pad_length = target_length - len(water_levels)
                
                # Create padding with slight variation
                padding = np.random.normal(mean_level, np.std(water_levels) * 0.1, pad_length)
                padding = np.clip(padding, 300, 5000)  # Keep in valid range
                
                # Create timestamps for padding
                first_time = timestamps[0]
                pad_timestamps = [
                    first_time - timedelta(minutes=pad_length - i) 
                    for i in range(pad_length)
                ]
                
                water_levels = list(padding) + water_levels
                timestamps = pad_timestamps + timestamps
                
                print(f"✅ Padded to {len(water_levels)} readings")
        
        # Take the most recent 4320 readings
        if len(water_levels) > target_length:
            water_levels = water_levels[-target_length:]
            timestamps = timestamps[-target_length:]
            print(f"📊 Using most recent {target_length} readings")
        
        print(f"🎯 Final sequence length: {len(water_levels)} (target: {target_length})")
        
        return water_levels, timestamps
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None, None

def create_sample_data():
    """
    Create realistic tidal sample data for testing
    Generates 4320 readings with tidal patterns and noise
    """
    print("🧪 Generating sample tidal data...")
    
    # Parameters for realistic tidal simulation
    hours = 72
    minutes_per_hour = 60
    total_minutes = hours * minutes_per_hour  # 4320
    
    # Time array
    t = np.linspace(0, hours, total_minutes)
    
    # Tidal components (realistic for North Carolina coast)
    # Semi-diurnal tide (12.42 hour period)
    M2 = 800 * np.sin(2 * np.pi * t / 12.42)  # Principal semi-diurnal
    
    # Diurnal components (24 hour period)  
    K1 = 300 * np.sin(2 * np.pi * t / 24.0)   # Diurnal
    
    # Weather effects (longer periods)
    weather = 200 * np.sin(2 * np.pi * t / 48.0) + 100 * np.sin(2 * np.pi * t / 36.0)
    
    # Base water level (in mm)
    base_level = 2000  # 2 meters
    
    # Combine components
    water_levels = base_level + M2 + K1 + weather
    
    # Add realistic noise
    noise = np.random.normal(0, 50, len(water_levels))  # 5cm std dev
    water_levels += noise
    
    # Ensure values are in valid sensor range
    water_levels = np.clip(water_levels, 300, 5000)
    
    # Create timestamps (current time going backwards)
    now = datetime.now()
    timestamps = [now - timedelta(minutes=total_minutes - i - 1) for i in range(total_minutes)]
    
    print(f"✅ Generated {len(water_levels)} sample readings")
    print(f"🌊 Range: {water_levels.min():.1f} - {water_levels.max():.1f} mm")
    print(f"📅 Time span: {timestamps[0].strftime('%Y-%m-%d %H:%M')} to {timestamps[-1].strftime('%Y-%m-%d %H:%M')}")
    
    return list(water_levels), timestamps

def main():
    """Test the Firebase fetch functionality"""
    print("🧪 Testing Transformer Firebase Data Fetch")
    print("=" * 45)
    
    # Test real data fetch
    print("\n1️⃣  Testing real Firebase data:")
    water_levels, timestamps = get_transformer_input_sequence()
    
    if water_levels:
        print(f"✅ Successfully fetched {len(water_levels)} readings")
        print(f"📈 First 5: {[f'{w:.1f}' for w in water_levels[:5]]}")
        print(f"📉 Last 5: {[f'{w:.1f}' for w in water_levels[-5:]]}")
        
        # Save for inspection
        with open('transformer_real_data.json', 'w') as f:
            json.dump({
                'water_levels': water_levels,
                'timestamps': [t.isoformat() for t in timestamps],
                'count': len(water_levels),
                'stats': {
                    'min': float(min(water_levels)),
                    'max': float(max(water_levels)),
                    'mean': float(np.mean(water_levels)),
                    'std': float(np.std(water_levels))
                }
            }, f, indent=2)
        
        print(f"💾 Real data saved to transformer_real_data.json")
    
    # Test sample data generation
    print("\n2️⃣  Testing sample data generation:")
    sample_levels, sample_times = create_sample_data()
    
    with open('transformer_sample_data.json', 'w') as f:
        json.dump({
            'water_levels': sample_levels,
            'timestamps': [t.isoformat() for t in sample_times],
            'count': len(sample_levels),
            'stats': {
                'min': float(min(sample_levels)),
                'max': float(max(sample_levels)),
                'mean': float(np.mean(sample_levels)),
                'std': float(np.std(sample_levels))
            }
        }, f, indent=2)
    
    print(f"💾 Sample data saved to transformer_sample_data.json")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
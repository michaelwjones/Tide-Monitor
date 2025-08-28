#!/usr/bin/env python3
"""
Fetch real data from Firebase for LSTM testing
Gets the most recent 72 hours of water level data
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
        
        # Calculate cutoff time (72 hours ago)
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
                
                # Filter by time (last 72 hours)
                if timestamp < cutoff_time:
                    continue
                
                # Get water level
                water_level = float(reading['w'])
                
                # Basic validation (reasonable water level range)
                if 1000 < water_level < 3000:  # 1-3 meters in mm
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

def get_latest_72_readings():
    """
    Get exactly 72 readings for LSTM input
    If more than 72, takes the most recent 72
    If less than 72, raises an error
    """
    try:
        water_levels, timestamps, full_data = fetch_firebase_data(hours=72)
        
        if len(water_levels) < 72:
            # Try extending the time range
            print(f"⚠️  Only {len(water_levels)} readings found, extending search to 120 hours...")
            water_levels, timestamps, full_data = fetch_firebase_data(hours=120)
            
            if len(water_levels) < 72:
                raise ValueError(f"Insufficient data: only {len(water_levels)} readings found (need 72)")
        
        # Take the most recent 72 readings
        if len(water_levels) > 72:
            water_levels = water_levels[-72:]
            timestamps = timestamps[-72:]
            full_data = full_data[-72:]
            print(f"📊 Using most recent 72 readings")
        
        return water_levels, timestamps, full_data
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None, None, None

def main():
    """Test the Firebase fetch functionality"""
    print("🧪 Testing Firebase Data Fetch")
    print("=" * 35)
    
    water_levels, timestamps, full_data = get_latest_72_readings()
    
    if water_levels:
        print(f"\n✅ Successfully fetched {len(water_levels)} readings")
        print(f"📈 Sample data (first 5): {[f'{w:.1f}' for w in water_levels[:5]]}")
        print(f"📉 Sample data (last 5): {[f'{w:.1f}' for w in water_levels[-5:]]}")
        
        # Save for manual inspection
        with open('firebase_test_data.json', 'w') as f:
            json.dump({
                'water_levels': water_levels,
                'timestamps': [t.isoformat() for t in timestamps],
                'count': len(water_levels),
                'range': [min(water_levels), max(water_levels)]
            }, f, indent=2)
        
        print(f"💾 Data saved to firebase_test_data.json")
        
    else:
        print("❌ Failed to fetch data")

if __name__ == "__main__":
    main()
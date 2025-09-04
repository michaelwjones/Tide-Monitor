#!/usr/bin/env python3
"""
Firebase Data Fetching for Transformer v1 Testing
Gets exactly 433 readings (72 hours @ 10min intervals) of water level data for seq2seq input
"""

import requests
import numpy as np
from datetime import datetime, timedelta

def fetch_firebase_data():
    """
    Fetch the last 4320 readings from Firebase (3 days of 1-minute data)
    Then downsample to 433 readings (10-minute intervals for 72 hours)
    """
    firebase_url = "https://tide-monitor-boron-default-rtdb.firebaseio.com/readings.json?orderBy=\"$key\"&limitToLast=4320"
    
    try:
        print("📡 Fetching last 4320 readings from Firebase...")
        
        response = requests.get(firebase_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            raise ValueError("No data received from Firebase")
        
        print(f"📦 Received {len(data)} readings")
        
        # Convert to list and sort by timestamp
        readings = []
        for key, reading in data.items():
            if not isinstance(reading, dict):
                continue
                
            # Check if reading has required fields
            if 't' not in reading or 'w' not in reading:
                continue
                
            try:
                # Parse timestamp
                timestamp_str = reading['t']
                if timestamp_str.endswith('Z'):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str + '+00:00')
                
                # Get water level
                water_level = float(reading['w'])
                
                # Basic validation (reasonable water level range)
                if 300 < water_level < 5000:  # Sensor range in mm
                    readings.append({
                        'timestamp': timestamp,
                        'water_level': water_level,
                        'iso_time': timestamp_str
                    })
                    
            except (ValueError, TypeError, KeyError) as e:
                continue
        
        if not readings:
            raise ValueError("No valid readings found after filtering")
        
        # Sort by timestamp (oldest first)
        readings.sort(key=lambda x: x['timestamp'])
        
        print(f"✅ Found {len(readings)} valid readings")
        print(f"📅 Time range: {readings[0]['iso_time']} to {readings[-1]['iso_time']}")
        
        # Downsample from 1-minute to 10-minute intervals
        # Take every 10th reading to get exactly 432 readings
        print(f"📊 Downsampling to 10-minute intervals...")
        
        # Take every 10th reading to get 433 readings from 4320
        if len(readings) >= 4320:
            # Perfect case: take every 10th reading, including first and last
            downsampled = [readings[i] for i in range(0, len(readings), 10)]
            # Ensure we include the very last reading
            if readings[-1] not in downsampled:
                downsampled.append(readings[-1])
        else:
            # Less than ideal: use what we have and space evenly
            if len(readings) >= 433:
                indices = np.linspace(0, len(readings)-1, 433, dtype=int)
                downsampled = [readings[i] for i in indices]
            else:
                # Very few readings - just use what we have
                downsampled = readings
        
        # Ensure exactly 433 readings
        if len(downsampled) > 433:
            downsampled = downsampled[-433:]  # Take last 433
        
        print(f"✅ Downsampled to {len(downsampled)} readings for transformer input")
        
        if downsampled:
            print(f"📅 Final time range: {downsampled[0]['iso_time']} to {downsampled[-1]['iso_time']}")
            print(f"🌊 Water level range: {min(r['water_level'] for r in downsampled):.1f} - {max(r['water_level'] for r in downsampled):.1f} mm")
        
        # Extract water levels and timestamps
        water_levels = [r['water_level'] for r in downsampled]
        timestamps = [r['timestamp'] for r in downsampled]
        
        return water_levels, timestamps
        
    except requests.RequestException as e:
        raise Exception(f"Firebase request failed: {e}")
    except Exception as e:
        raise Exception(f"Data processing failed: {e}")

def get_transformer_input_sequence():
    """
    Get exactly 433 readings for Transformer seq2seq input (72 hours @ 10min intervals)
    Transformer requires fixed-length input for proper attention computation
    """
    try:
        water_levels, timestamps = fetch_firebase_data()
        
        target_length = 433  # 72 hours * 6 readings/hour + 1 = 433 readings (first + last)
        
        if len(water_levels) < target_length:
            print(f"⚠️  Only {len(water_levels)} readings found, need {target_length}")
            if len(water_levels) < 50:  # Too few to work with
                print("❌ Insufficient data - using sample data instead")
                return create_sample_data()
            else:
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
                    first_time - timedelta(minutes=10 * (pad_length - i)) 
                    for i in range(pad_length)
                ]
                
                water_levels = list(padding) + water_levels
                timestamps = pad_timestamps + timestamps

        # Take the most recent 433 readings
        if len(water_levels) > target_length:
            water_levels = water_levels[-target_length:]
            timestamps = timestamps[-target_length:]
        
        print(f"🎯 Final sequence: {len(water_levels)} readings")
        
        return water_levels, timestamps
        
    except Exception as e:
        print(f"❌ Error fetching Firebase data: {e}")
        print("🧪 Falling back to sample data")
        return create_sample_data()

def create_sample_data():
    """
    Generates 433 readings with tidal patterns and noise
    """
    print("🧪 Generating 433 sample readings...")
    
    hours = 72  # 72 hours of data
    total_minutes = hours * 6 + 1  # 433 readings at 10-minute intervals (first + last)
    
    # Time array (72 hours)
    t = np.linspace(0, hours, total_minutes)
    
    # Base water level (around 2000mm)
    base_level = 2000
    
    # Tidal components
    M2_tide = 400 * np.sin(2 * np.pi * t / 12.42)  # Principal semi-diurnal (12.42 hours)
    S2_tide = 150 * np.sin(2 * np.pi * t / 12.0)   # Solar semi-diurnal (12 hours)
    K1_tide = 100 * np.sin(2 * np.pi * t / 25.8)   # Diurnal (23.93 hours)
    O1_tide = 80 * np.sin(2 * np.pi * t / 25.8)    # Principal diurnal (25.82 hours)
    
    # Add some weather effects and noise
    weather_trend = 100 * np.sin(2 * np.pi * t / 48)  # 2-day weather cycle
    noise = np.random.normal(0, 20, total_minutes)      # Random noise
    
    # Combine all components
    water_levels = base_level + M2_tide + S2_tide + K1_tide + O1_tide + weather_trend + noise
    
    # Ensure realistic range
    water_levels = np.clip(water_levels, 300, 5000)
    
    # Generate timestamps (every 10 minutes for 72 hours)
    start_time = datetime.now() - timedelta(hours=72)
    timestamps = [start_time + timedelta(minutes=10 * i) for i in range(total_minutes)]
    
    print(f"✅ Generated {len(water_levels)} sample readings")
    print(f"🌊 Range: {min(water_levels):.1f} - {max(water_levels):.1f} mm")
    
    return water_levels.tolist(), timestamps
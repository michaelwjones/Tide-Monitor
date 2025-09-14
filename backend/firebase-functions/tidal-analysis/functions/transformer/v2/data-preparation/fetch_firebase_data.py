import requests
import json
import os
from datetime import datetime

def fetch_firebase_data():
    """
    Fetch all water level data from Firebase Realtime Database.
    Downloads readings from June 30, 2025 to present for Transformer training.
    """
    print("Fetching tide monitor data from Firebase...")
    
    # Firebase Realtime Database endpoint
    firebase_url = "https://tide-monitor-boron-default-rtdb.firebaseio.com/readings.json"
    
    try:
        response = requests.get(firebase_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            print("No data found in Firebase database")
            return None
            
        print(f"Successfully fetched {len(data)} readings from Firebase")
        
        # Filter out readings with invalid water levels (< -200mm) and old dates (July 6, 2025 and earlier)
        filtered_data = {}
        water_level_skipped = 0
        date_skipped = 0
        
        # Cutoff date: July 7, 2025 03:00:00 UTC (everything before this is filtered out)
        cutoff_date = datetime(2025, 7, 7, 3, 0, 0)
        
        for key, reading in data.items():
            skip_reading = False
            
            # Check date filter first
            if 't' in reading:
                try:
                    # Parse timestamp as UTC 
                    timestamp_str = reading['t'].replace('Z', '+00:00')
                    timestamp = datetime.fromisoformat(timestamp_str)
                    # Convert to naive UTC for comparison
                    if timestamp.tzinfo:
                        timestamp = timestamp.replace(tzinfo=None)
                    
                    if timestamp < cutoff_date:
                        date_skipped += 1
                        skip_reading = True
                except (ValueError, TypeError):
                    # Skip readings with invalid timestamps
                    date_skipped += 1
                    skip_reading = True
            else:
                # Skip readings without timestamp
                date_skipped += 1
                skip_reading = True
            
            if not skip_reading and 'w' in reading:
                try:
                    water_level = float(reading['w'])
                    if water_level < -200:
                        water_level_skipped += 1
                        skip_reading = True
                except (ValueError, TypeError):
                    # Skip readings with invalid water level values
                    water_level_skipped += 1
                    skip_reading = True
            elif not skip_reading:
                # Keep readings without water level data if they pass date filter
                pass
            
            if not skip_reading:
                filtered_data[key] = reading
        
        total_skipped = water_level_skipped + date_skipped
        print(f"Filtered out {date_skipped} readings from July 7, 2025 03:00 UTC and earlier")
        print(f"Filtered out {water_level_skipped} readings with invalid water levels (< -200mm)")
        print(f"Total filtered: {total_skipped} readings")
        print(f"Kept {len(filtered_data)} readings after filtering")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save raw data (unfiltered)
        with open('data/firebase_raw_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save filtered data
        with open('data/firebase_filtered_data.json', 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print("Raw data saved to data/firebase_raw_data.json")
        print("Filtered data saved to data/firebase_filtered_data.json")
        return filtered_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Firebase: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None

if __name__ == "__main__":
    print("Transformer v2 Data Preparation - Firebase Data Fetcher")
    print("=" * 55)
    
    data = fetch_firebase_data()
    
    if data:
        # Show basic statistics
        readings_with_water_level = sum(1 for reading in data.values() if 'w' in reading)
        print(f"\nData Summary:")
        print(f"Total readings: {len(data)}")
        print(f"Readings with water level: {readings_with_water_level}")
        
        if readings_with_water_level > 0:
            print(f"\nNext step: Run create_training_data.py to process this filtered data for Transformer training")
        else:
            print(f"\nWarning: No water level data found in readings")
    else:
        print("Failed to fetch data. Please check your internet connection and try again.")
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
        
        # Filter out readings with invalid water levels (< -200mm)
        filtered_data = {}
        skipped_count = 0
        
        for key, reading in data.items():
            if 'w' in reading:
                try:
                    water_level = float(reading['w'])
                    if water_level >= -200:
                        filtered_data[key] = reading
                    else:
                        skipped_count += 1
                except (ValueError, TypeError):
                    # Skip readings with invalid water level values
                    skipped_count += 1
            else:
                # Keep readings without water level data
                filtered_data[key] = reading
        
        print(f"Filtered out {skipped_count} readings with invalid water levels (< -200mm)")
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
    print("Transformer v1 Data Preparation - Firebase Data Fetcher")
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
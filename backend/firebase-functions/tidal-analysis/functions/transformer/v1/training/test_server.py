#!/usr/bin/env python3
"""
Test script for the transformer model server.
"""
import requests
import json
import numpy as np
import time


def test_server(base_url='http://localhost:8000'):
    """Test the transformer model server endpoints"""
    
    print("Testing Transformer Model Server")
    print("=" * 40)
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        print("   ✓ Health check passed")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return False
    
    # Test 2: Model info
    print("\n2. Testing model info...")
    try:
        response = requests.get(f"{base_url}/info")
        assert response.status_code == 200
        info = response.json()
        print("   ✓ Model info retrieved")
        print(f"   Model: {info['model']['architecture']}")
        print(f"   Parameters: {info['model']['total_parameters']:,}")
        print(f"   Training loss: {info['training']['final_loss']:.6f}")
    except Exception as e:
        print(f"   ✗ Model info failed: {e}")
        return False
    
    # Test 3: Single prediction
    print("\n3. Testing single prediction...")
    try:
        # Generate example water level data (72 hours = 433 points)
        example_data = np.random.randn(433) * 50 + 1000  # Random around 1000mm
        
        payload = {
            "water_levels": example_data.tolist(),
            "start_time": "2025-01-01T00:00:00Z"
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", json=payload)
        inference_time = time.time() - start_time
        
        assert response.status_code == 200
        result = response.json()
        
        print("   ✓ Single prediction successful")
        print(f"   Input points: {len(payload['water_levels'])}")
        print(f"   Output points: {len(result['predictions'])}")
        print(f"   Inference time: {inference_time*1000:.1f} ms")
        print(f"   Sample predictions: {result['predictions'][:3]} ... {result['predictions'][-3:]}")
        
        # Verify timestamps were generated
        if 'timestamps' in result:
            print(f"   Timestamps: {result['timestamps'][0]} to {result['timestamps'][-1]}")
        
    except Exception as e:
        print(f"   ✗ Single prediction failed: {e}")
        return False
    
    # Test 4: Batch prediction
    print("\n4. Testing batch prediction...")
    try:
        batch_payload = {
            "requests": [
                {
                    "water_levels": (np.random.randn(433) * 50 + 1000).tolist(),
                    "id": "test_1"
                },
                {
                    "water_levels": (np.random.randn(433) * 50 + 1100).tolist(), 
                    "id": "test_2"
                }
            ]
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/predict/batch", json=batch_payload)
        batch_time = time.time() - start_time
        
        assert response.status_code == 200
        result = response.json()
        
        print("   ✓ Batch prediction successful")
        print(f"   Batch size: {result['metadata']['batch_size']}")
        print(f"   Successful: {result['metadata']['successful_predictions']}")
        print(f"   Batch time: {batch_time*1000:.1f} ms")
        print(f"   Per-prediction: {batch_time*1000/2:.1f} ms")
        
    except Exception as e:
        print(f"   ✗ Batch prediction failed: {e}")
        return False
    
    # Test 5: Error handling
    print("\n5. Testing error handling...")
    try:
        # Test with wrong input size
        bad_payload = {"water_levels": [1, 2, 3]}  # Too few values
        response = requests.post(f"{base_url}/predict", json=bad_payload)
        assert response.status_code == 400
        print("   ✓ Input validation works")
        
        # Test with no JSON
        response = requests.post(f"{base_url}/predict")
        assert response.status_code == 400
        print("   ✓ Missing JSON handling works")
        
    except Exception as e:
        print(f"   ✗ Error handling test failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("All tests passed! Server is working correctly.")
    return True


def generate_example_request():
    """Generate an example request for manual testing"""
    
    print("\nExample API Request:")
    print("=" * 20)
    
    # Generate realistic-looking water level data
    base_level = 1000  # mm
    hours = np.linspace(0, 72, 433)  # 72 hours
    
    # Simulate tidal pattern + noise
    tidal = 200 * np.sin(2 * np.pi * hours / 12.4)  # 12.4 hour tidal cycle
    noise = np.random.randn(433) * 20
    water_levels = base_level + tidal + noise
    
    example = {
        "water_levels": water_levels.tolist(),
        "start_time": "2025-01-01T00:00:00Z"
    }
    
    print("POST /predict")
    print("Content-Type: application/json")
    print()
    print("Payload (truncated):")
    print(json.dumps({
        "water_levels": water_levels[:5].tolist() + ["..."] + water_levels[-5:].tolist(),
        "start_time": "2025-01-01T00:00:00Z",
        "note": f"Full array has {len(water_levels)} values"
    }, indent=2))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        generate_example_request()
    else:
        print("Make sure the server is running: python model_server.py")
        print("Then run this test script.")
        print()
        
        try:
            success = test_server()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nTest suite failed: {e}")
            sys.exit(1)
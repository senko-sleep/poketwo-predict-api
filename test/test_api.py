"""Test the Pokemon API with Supabase integration."""
import requests
import json
import sys
import os

# Add parent directory to path to import from the api
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get("http://localhost:8080/health")
        print(f"Health Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_prediction():
    """Test prediction with a sample image."""
    try:
        # Use a small test image (1x1 pixel PNG)
        test_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        
        response = requests.post(
            "http://localhost:8080/predict",
            data=test_image,
            headers={'Content-Type': 'image/png'}
        )
        print(f"Prediction Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Pokemon API with Supabase integration...")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint:")
    health_ok = test_health()
    
    # Test prediction endpoint
    print("\n2. Testing Prediction Endpoint:")
    prediction_ok = test_prediction()
    
    print("\n" + "=" * 50)
    if health_ok and prediction_ok:
        print("All tests passed!")
    else:
        print("Some tests failed")
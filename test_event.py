import requests
import os
import sys

# Test the health endpoint first
response = requests.get("http://127.0.0.1:8080/health")
print("Health check:", response.json())

# Test with an event pokemon image (gender_creative_flag_vivillon)
print("\n=== Testing event pokemon (gender_creative_flag_vivillon) ===")
image_url = "https://cdn.poketwo.net/images/50280.png"
try:
    image_response = requests.get(image_url, timeout=5)
    image_bytes = image_response.content

    # Test prediction endpoint
    predict_response = requests.post(
        "http://127.0.0.1:8080/predict",
        data=image_bytes,
        headers={"Content-Type": "image/png"}
    )
    result = predict_response.json()
    print("Prediction result:", result)
except Exception as e:
    print(f"Error testing event pokemon: {e}")
    sys.exit(1)

# Test with a regular pokemon to ensure normal predictions still work
print("\n=== Testing regular pokemon (pikachu) ===")
regular_url = "https://cdn.poketwo.net/images/25.png"
try:
    regular_response = requests.get(regular_url, timeout=5)
    regular_bytes = regular_response.content

    regular_predict = requests.post(
        "http://127.0.0.1:8080/predict",
        data=regular_bytes,
        headers={"Content-Type": "image/png"}
    )
    result = regular_predict.json()
    print("Regular pokemon prediction:", result)
except Exception as e:
    print(f"Error testing regular pokemon: {e}")
    sys.exit(1)

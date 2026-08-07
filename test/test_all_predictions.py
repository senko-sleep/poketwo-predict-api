import requests
import json
import os

# Load test data
test_json_path = os.path.join(os.path.dirname(__file__), 'test.json')
with open(test_json_path, 'r') as f:
    test_data = json.load(f)

url = "https://pokemon-predict-api.vercel.app/predict/url"

print("Testing all predictions from test.json")
print("=" * 60)

results = []
for test_case in test_data['test_images']:
    image_url = test_case['url']
    expected_name = test_case['name']
    
    data = {"url": image_url}
    
    try:
        response = requests.post(url, json=data, timeout=30)
        result = response.json()
        
        predicted_name = result.get('pokemon', 'unknown')
        confidence = result.get('confidence', 'N/A')
        
        is_correct = predicted_name == expected_name
        status = "✓" if is_correct else "✗"
        
        print(f"{status} Expected: {expected_name}")
        print(f"  Predicted: {predicted_name} ({confidence})")
        print(f"  URL: {image_url[:60]}...")
        print()
        
        results.append({
            'expected': expected_name,
            'predicted': predicted_name,
            'confidence': confidence,
            'correct': is_correct,
            'url': image_url
        })
    except Exception as e:
        print(f"✗ Error testing {expected_name}: {e}")
        print()
        results.append({
            'expected': expected_name,
            'predicted': 'error',
            'confidence': 'N/A',
            'correct': False,
            'url': image_url
        })

print("=" * 60)
print(f"Summary: {sum(1 for r in results if r['correct'])}/{len(results)} correct")

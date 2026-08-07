import requests
import json

url = "http://localhost:8080/predict/url"
# Test with the actual sammy image
image_url = "https://images-ext-1.discordapp.net/external/9c9M10rPZaxqGDC-D17qnBDv8n49Z4Hfp8RbN0MS3BQ/%3Fsize%3D1024/https/cdn.discordapp.com/avatars/1074905720073506876/6e92a896fc8009a836763720c223da70.png?format=webp&quality=lossless"

data = {"url": image_url}

response = requests.post(url, json=data, timeout=30)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

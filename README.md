# Pokemon ONNX Prediction API

A standalone Flask API service for Pokemon spawn prediction using ONNX model inference.

## Features

- **ONNX Model Inference**: Fast prediction using ONNX Runtime
- **REST API**: Simple POST endpoints for image prediction
- **Docker Ready**: Containerized for easy deployment
- **Free Hosting Compatible**: Works with Render, Railway, and similar platforms

## API Endpoints

### POST /predict
Predict Pokemon from image bytes.

**Request:**
- Content-Type: application/octet-stream
- Body: Raw image bytes

**Response:**
```json
{
  "pokemon": "koffing",
  "confidence": "94.13%",
  "confidence_raw": 0.9413,
  "top_index": 123
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "num_labels": 1332
}
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The API will be available at http://localhost:8080

## Docker

```bash
# Build the image
docker build -t pokemon-predict-api .

# Run the container
docker run -p 8080:8080 pokemon-predict-api
```

## Deployment

### Render (Free)

1. Create a new Web Service on Render
2. Connect this GitHub repository
3. Render will automatically detect the Dockerfile
4. Deploy!

### Railway (Free)

1. Create a new project on Railway
2. Connect this GitHub repository
3. Railway will automatically detect the Dockerfile
4. Deploy!

### Other Platforms

Any platform that supports Docker containers will work:
- Fly.io
- Heroku (with Container Registry)
- DigitalOcean App Platform
- AWS App Runner

## Usage Example

```python
import requests

# Load image
with open("pokemon.jpg", "rb") as f:
    image_bytes = f.read()

# Send prediction request
response = requests.post(
    "https://your-api-url.com/predict",
    data=image_bytes,
    headers={"Content-Type": "application/octet-stream"}
)

result = response.json()
print(f"Pokemon: {result['pokemon']}")
print(f"Confidence: {result['confidence']}")
```

## Model Files

- `pokemon_cnn_v2.onnx`: ONNX model file (45 MB)
- `labels_v2.json`: Pokemon label mappings

These files are included in the repository. When updating the model, replace these files and redeploy.

## Environment Variables

- `PORT`: Server port (default: 8080)
- `ONNX_MODEL_PATH`: Path to ONNX model (default: pokemon_cnn_v2.onnx)
- `LABELS_PATH`: Path to labels file (default: labels_v2.json)

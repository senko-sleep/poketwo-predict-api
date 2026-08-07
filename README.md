# Pokemon Predict API

Simple, clean Pokemon image recognition API. Handles both event and normal Pokemon prediction with a straightforward interface.

## Deployment

**Live API:** https://pokemon-predict-api.vercel.app

## Structure

```
pokemon-predict-api/
├── models/                 # Model files
│   ├── pokemon_cnn_v2.onnx
│   └── labels_v2.json
├── recognition.py          # Image recognition module
├── api.py                 # REST API
├── vercel.json            # Vercel deployment config
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Usage

### Running Locally

```bash
python api.py
```

The API will start on `http://127.0.0.1:8080`

### API Endpoints

- `GET /health` - Health check with statistics
  - Returns: status, model_loaded, total_predictions, average_prediction_time_ms, prediction_count
- `POST /predict` - Predict Pokemon from image bytes
  - Returns: pokemon, confidence, confidence_raw, prediction_time_ms
- `POST /predict/url` - Predict Pokemon from image URL  
  - Returns: pokemon, confidence, confidence_raw, prediction_time_ms

### Using the Recognition Module

```python
from recognition import PokemonRecognizer

recognizer = PokemonRecognizer()
pokemon_name, confidence = recognizer.predict(image_bytes)
print(f"Predicted: {pokemon_name} @ {confidence * 100:.2f}%")
```

## Features

- **Simple Recognition**: Clean image recognition for Pokemon
- **Unified System**: Handles both event and normal Pokemon in one system
- **No Complexity**: No layers of files or unnecessary complexity
- **Clean Imports**: Simple, straightforward import structure
- **Statistics Tracking**: Built-in prediction time and count tracking
- **Health Monitoring**: Real-time API health and performance metrics
- **Production Deployed**: Hosted on Vercel for reliable access

## Statistics

The API tracks:
- Total number of predictions made
- Average prediction time in milliseconds
- Prediction count (last 100 predictions kept in memory)

## Deployment

Deployed to Vercel at: https://pokemon-predict-api.vercel.app

To redeploy:
```bash
npx vercel --prod --yes
```

## Dependencies

- Flask
- Flask-CORS
- ONNX Runtime
- NumPy
- Pillow
- Requests
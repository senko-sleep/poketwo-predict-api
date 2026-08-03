# Pokemon Prediction API

A high-performance Pokemon spawn prediction API optimized for Poketwo with reinforcement learning capabilities.

## Project Structure

```
pokemon-predict-api/
├── docs/                          # Documentation
│   ├── README.md                  # Main documentation
│   ├── PERFORMANCE.md             # Performance tuning guide
│   └── REINFORCEMENT_LEARNING.md  # Reinforcement learning system docs
├── models/                        # Model files and data
│   ├── pokemon_cnn_v2.onnx        # Main ONNX model
│   ├── labels_v2.json             # Pokemon labels
│   ├── event_embedding_index.npz  # Event embedding index
│   ├── event_embedding_meta.json   # Event embedding metadata
│   ├── event_labels.json          # Event labels
│   └── event_label_config.json    # Event label configuration
├── scripts/                       # Utility scripts
│   ├── start_fast.bat             # Windows fast startup script
│   ├── start_fast.sh              # Linux/Mac fast startup script
│   ├── test_event.py              # Event testing script
│   └── test_feedback.py           # Feedback system testing
├── src/                           # Source code
│   ├── app.py                     # Main Flask application
│   ├── poketwo_feedback.py        # Poketwo feedback system
│   └── POKETWO_INTEGRATION.py     # Discord bot integration example
├── run.py                         # Main entry point
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
└── .gitignore                     # Git ignore rules
```

## Quick Start

### Standard Mode
```bash
python run.py
```

### High Performance Mode
```bash
# Windows
.\scripts\start_fast.bat

# Linux/Mac
./scripts/start_fast.sh
```

### Docker
```bash
docker build -t pokemon-predict-api .
docker run -p 8080:8080 pokemon-predict-api
```

## Features

- **High Performance**: Optimized for high-volume spawn processing
- **Reinforcement Learning**: Learns from Poketwo catch messages to improve accuracy
- **Adaptive Caching**: Instant responses for duplicate spawns
- **GPU Acceleration**: Automatic GPU detection and fallback
- **Event Pokemon Detection**: Special handling for event variants
- **Configurable**: Extensive environment variable configuration

## API Endpoints

- `POST /predict` - Predict Pokemon from image bytes
- `POST /predict/url` - Predict Pokemon from image URL
- `GET /health` - Health check and performance stats
- `POST /feedback` - Submit Poketwo catch message feedback
- `GET /feedback/stats` - Get reinforcement learning statistics

## Configuration

Environment variables can be set to customize performance:

- `ENABLE_TTA` - Enable test-time augmentation (default: false)
- `ENABLE_GPU` - Enable GPU acceleration (default: true)
- `ENABLE_CACHE` - Enable prediction caching (default: true)
- `ENABLE_EVENT_EMBEDDING` - Enable event Pokemon detection (default: true)
- `ENABLE_FEEDBACK` - Enable reinforcement learning (default: true)
- `MAX_WORKERS` - Maximum concurrent workers (default: 4)
- `CACHE_SIZE` - Maximum cache entries (default: 1000)

## Documentation

- [Performance Guide](docs/PERFORMANCE.md) - Performance tuning and optimization
- [Reinforcement Learning](docs/REINFORCEMENT_LEARNING.md) - Feedback system documentation
- [Integration Example](src/POKETWO_INTEGRATION.py) - Discord bot integration

## Requirements

- Python 3.11+
- ONNX Runtime
- Flask
- NumPy
- PIL/Pillow
- Requests

See `requirements.txt` for complete list.
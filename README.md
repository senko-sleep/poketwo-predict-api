# Pokemon Prediction API

A high-performance Pokemon spawn prediction API optimized for Poketwo with reinforcement learning capabilities.

## Quick Start

```bash
# Standard mode
python run.py

# High performance mode
.\scripts\start_fast.bat  # Windows
./scripts/start_fast.sh   # Linux/Mac
```

## Project Structure

```
pokemon-predict-api/
├── docs/          # Documentation
├── models/        # Model files and data
├── scripts/       # Utility scripts
├── src/           # Source code
├── run.py         # Main entry point
└── requirements.txt
```

## Features

- **High Performance**: Optimized for high-volume spawn processing
- **Reinforcement Learning**: Learns from Poketwo catch messages
- **Adaptive Caching**: Instant responses for duplicate spawns
- **GPU Acceleration**: Automatic GPU detection and fallback
- **Event Pokemon Detection**: Special handling for event variants

## API Endpoints

- `POST /predict` - Predict Pokemon from image bytes
- `POST /predict/url` - Predict Pokemon from image URL
- `GET /health` - Health check and performance stats
- `POST /feedback` - Submit Poketwo catch message feedback

## Documentation

See [docs/](docs/) for detailed documentation:
- [Main README](docs/MAIN_README.md) - Complete documentation
- [Performance Guide](docs/PERFORMANCE.md) - Performance tuning
- [Reinforcement Learning](docs/REINFORCEMENT_LEARNING.md) - Feedback system
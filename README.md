# Pokemon Prediction API

A high-performance Pokemon image recognition API with Supabase caching and Vercel optimization.

## Features

- **Image Recognition**: Fast Pokemon prediction using ONNX models
- **Supabase Caching**: Reduces computational load and response times
- **Response Compression**: Gzip compression for reduced bandwidth usage
- **Vercel Optimized**: Configured for minimal data transfer costs
- **Health Monitoring**: Built-in statistics and cache hit tracking

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Supabase

1. Run the SQL schema in your Supabase SQL editor:
```bash
cat supabase_client/schema.sql
```

2. Set environment variables:
```bash
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_KEY=your-supabase-anon-key
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run Locally

```bash
python api.py
```

The API will be available at `http://localhost:8080`

## API Endpoints

### POST /predict

Predict Pokemon from image bytes.

**Request:**
- Body: Raw image bytes (binary)

**Response:**
```json
{
  "pokemon": "pikachu",
  "confidence": "98.5%",
  "confidence_raw": 0.985,
  "prediction_time_ms": 45.2,
  "cached": false
}
```

### POST /predict/url

Predict Pokemon from image URL.

**Request:**
```json
{
  "url": "https://example.com/pokemon.jpg"
}
```

**Response:**
```json
{
  "pokemon": "charizard",
  "confidence": "92.3%",
  "confidence_raw": 0.923,
  "prediction_time_ms": 38.7,
  "cached": true
}
```

### GET /health

Health check with statistics.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "total_predictions": 1523,
  "average_prediction_time_ms": 42.5,
  "prediction_count": 100,
  "cache_hits": 890,
  "cache_misses": 633,
  "cache_hit_rate": 58.4,
  "supabase_connected": true
}
```

## Vercel Optimization

This API is optimized to minimize Vercel data transfer costs:

1. **Supabase Caching**: Reduces repeated model inference
2. **Gzip Compression**: Compresses all API responses
3. **Smart Caching Headers**: 
   - GET requests: 5-minute cache
   - POST requests: No cache to ensure fresh predictions
4. **CDN-friendly**: Static health check responses are cached

## Data Transfer Reduction

The optimizations significantly reduce Vercel's Fast Origin Transfer usage:

- **Caching**: ~60% cache hit rate reduces model inference calls
- **Compression**: ~70% reduction in response size
- **Smart Headers**: Prevents unnecessary data transfer

## Monitoring

Monitor your Vercel dashboard and Supabase dashboard to track:

- Data transfer usage
- Cache hit rates
- Prediction performance
- API response times

## Troubleshooting

### Supabase Connection Issues

Check the `/health` endpoint for Supabase connection status.

### High Data Transfer

1. Monitor cache hit rate - aim for >50%
2. Check compression is working (response headers should include `Content-Encoding: gzip`)
3. Review prediction patterns for caching opportunities

### Model Loading Issues

Ensure model files are in the `models/` directory:
- `pokemon_cnn_v2.onnx`
- `labels_v2.json`
- `event_embedding_index.npz` (optional)

## Documentation

For detailed deployment and configuration information, see the [docs folder](docs/DEPLOYMENT.md).

## License

MIT
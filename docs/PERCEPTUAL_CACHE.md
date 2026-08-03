# Perceptual Caching System

## Overview

The perceptual caching system provides sprite-invariant caching for Pokemon spawn predictions, allowing repeated spawns to be recognized instantly without running inference again. This is especially effective for Poketwo where the same Pokemon sprites are reused across different backgrounds.

## How It Works

### 1. Sprite Extraction
- Uses OpenCV adaptive thresholding to separate foreground (Pokemon) from background
- Finds the largest contour and extracts the bounding box
- Handles edge cases with padding and minimum size checks

### 2. Canonicalization
- Crops the sprite to its bounding box
- Resizes to a fixed canonical size (64x64 pixels)
- Converts to RGBA to handle transparency
- Eliminates scale differences from the original resizing step

### 3. Perceptual Hashing
- Computes perceptual hash using `imagehash` library (phash)
- Computes hashes for both normal and flipped orientations
- Uses hamming distance threshold (default: 5) for matching
- Tolerates small compression/antialiasing differences

### 4. Cache Storage
- Persistent key-value store using pickle file
- Keys: perceptual hashes (normal and flipped)
- Values: prediction results with metadata
- Thread-safe operations with configurable size limits

### 5. Prediction Pipeline
1. Extract sprite bounding box from image
2. Canonicalize to 64x64 pixels
3. Compute perceptual hashes (normal + flipped)
4. Check cache for matches within hamming threshold
5. If cache hit: return cached prediction instantly
6. If cache miss: run ONNX model, store result in cache

## Configuration

Environment variables:
- `PERCEPTUAL_CACHE_ENABLED` (default: `true`): Enable/disable perceptual cache
- Cache file: `perceptual_cache.pkl` (in project root)
- Canonical size: 64x64 pixels
- Hamming threshold: 5 bits
- Max cache size: 10,000 entries

## Performance Impact

### First Prediction (Cache Miss)
- Bounding box extraction: ~100-200ms
- Canonicalization: ~50-100ms  
- Hash computation: ~50-100ms
- ONNX inference: ~2000ms
- **Total: ~2200-2400ms**

### Subsequent Predictions (Cache Hit)
- Bounding box extraction: ~100-200ms
- Canonicalization: ~50-100ms
- Hash computation: ~50-100ms
- Cache lookup: ~1-5ms
- **Total: ~200-400ms** (5-10x faster than ONNX)

### Benefits
- **Sprite Invariance**: Same Pokemon on different backgrounds cached as single entry
- **Flip Invariance**: Both orientations map to same cache entry
- **Scale Invariance**: Canonical size eliminates scale differences
- **Position Invariance**: Bounding box extraction isolates sprite from background

## Integration

The perceptual cache is automatically integrated into the `/predict` endpoint. No code changes needed for basic usage.

### Manual Usage

```python
from src.perceptual_cache import PerceptualCache
from PIL import Image

# Initialize cache
cache = PerceptualCache(
    cache_file="perceptual_cache.pkl",
    canonical_size=64,
    hamming_threshold=5,
    max_cache_size=10000
)

# Predict with caching
image = Image.open("pokemon.jpg")
prediction = cache.predict(
    image,
    onnx_predict_func=my_onnx_function,
    force_model=False
)

print(f"Pokemon: {prediction['pokemon']}")
print(f"Cache hit: {prediction['cache_hit']}")

# Get statistics
stats = cache.get_statistics()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

## Monitoring

### Health Endpoint
```bash
curl http://localhost:8080/health
```

Response includes perceptual cache statistics:
```json
{
  "perceptual_cache": {
    "enabled": true,
    "statistics": {
      "cache_entries": 150,
      "cache_hits": 120,
      "cache_misses": 30,
      "hit_rate": 0.8,
      "canonical_size": 64,
      "hamming_threshold": 5
    }
  }
}
```

### Cache Management
```python
# Clear cache
cache.clear_cache()

# Save cache manually
cache.save_cache()

# Get statistics
stats = cache.get_statistics()
```

## Advantages for Poketwo

1. **Repeated Spawns**: Same sprites appear frequently during events
2. **Background Independence**: Caching works regardless of background
3. **Flip Handling**: Both orientations automatically recognized
4. **No Model Retraining**: Works with existing ONNX model
5. **Instant Recognition**: Subsequent spawns are 5-10x faster

## Technical Details

### Bounding Box Extraction
- Uses adaptive thresholding for foreground segmentation
- Finds largest contour (assumes Pokemon is main object)
- Adds padding to include edges
- Fallback to entire image if extraction fails

### Perceptual Hashing
- Uses `imagehash.phash` for robust perceptual hashing
- Hamming distance allows for small variations
- Both orientations stored for flip invariance
- Fallback to simple hash if imagehash unavailable

### Cache Persistence
- Uses pickle for serialization
- Saves periodically (every 10 requests)
- Thread-safe operations
- FIFO eviction when size limit reached

## Limitations

- Requires successful sprite extraction
- May have false positives with very similar sprites
- Bounding box extraction can fail on complex backgrounds
- Additional processing time (~200-400ms) for cache operations

## Future Improvements

- More sophisticated sprite extraction (deep learning based)
- Adaptive hamming threshold based on Pokemon type
- Multi-scale caching for different sprite sizes
- Background-aware caching for event Pokemon
- Distributed cache (Redis) for multi-instance deployments
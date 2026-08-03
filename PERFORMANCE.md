# Pokemon Prediction API - Performance Configuration

This API is optimized for high-volume spawn processing with multiple performance tuning options.

## Environment Variables

### Core Performance Settings

- `ENABLE_TTA` (default: `false`): Enable Test-Time Augmentation (forward + flip). Disabling this gives ~2x speed improvement at minimal accuracy cost.
- `ENABLE_GPU` (default: `true`): Enable GPU acceleration if available (CUDA/TensorRT).
- `MAX_WORKERS` (default: `4`): Maximum number of concurrent worker threads.
- `ENABLE_CACHE` (default: `true`): Enable prediction caching for duplicate images.
- `CACHE_SIZE` (default: `1000`): Maximum number of cached predictions.
- `ENABLE_EVENT_EMBEDDING` (default: `true`): Enable event Pokemon detection via embedding index.

### Performance Recommendations

**For maximum speed during high spawn events:**
```bash
ENABLE_TTA=false ENABLE_GPU=true ENABLE_CACHE=true ENABLE_EVENT_EMBEDDING=false MAX_WORKERS=8
```

**For balanced performance/accuracy:**
```bash
ENABLE_TTA=false ENABLE_GPU=true ENABLE_CACHE=true ENABLE_EVENT_EMBEDDING=true MAX_WORKERS=4
```

**For maximum accuracy (slower):**
```bash
ENABLE_TTA=true ENABLE_GPU=true ENABLE_CACHE=true ENABLE_EVENT_EMBEDDING=true MAX_WORKERS=4
```

## Performance Impact

### TTA (Test-Time Augmentation)
- **Enabled**: ~4000ms per prediction (2 forward passes)
- **Disabled**: ~2000ms per prediction (1 forward pass)
- **Impact**: 2x speed improvement, minimal accuracy loss

### Event Embedding
- **Enabled**: +500-1000ms per prediction for event Pokemon detection
- **Disabled**: Faster prediction, no event Pokemon overrides
- **Impact**: Significant speed improvement during events, may miss special event variants

### Caching
- **Enabled**: ~10ms for cache hits (instant response for duplicate spawns)
- **Disabled**: Full prediction time for every request
- **Impact**: Massive speed improvement for duplicate spawns (common during events)

### GPU Acceleration
- **Enabled**: ~500-1000ms per prediction (if GPU available)
- **Disabled**: ~2000ms per prediction (CPU only)
- **Impact**: 2-4x speed improvement if CUDA/TensorRT available

## Expected Performance

### Current Configuration (Optimized)
- Single prediction: ~2000ms
- Cache hit: ~10ms
- Concurrent requests: ~2000ms average (threaded)

### High-Volume Configuration
- Single prediction: ~1000ms (no event embedding)
- Cache hit: ~10ms
- Concurrent requests: ~1000ms average (threaded)

## Monitoring

Check the `/health` endpoint for current performance configuration and cache statistics:

```bash
curl http://localhost:8080/health
```

Response includes:
- Current performance settings
- Cache entry count
- Model loading status
- Event embedding status
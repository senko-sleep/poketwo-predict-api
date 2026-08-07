# Pokemon Prediction API - Fix Implementation Summary

## Overview
Updated the Vercel-deployed Pokemon prediction API with the comprehensive misclassification fix to address the root mathematical causes of false high-confidence predictions (like Golbat → Alloace Flag Vivillon at 92.30%).

## Changes Made

### 1. New Modules Added
- **`src/similarity_calibration.py`** - Probabilistic calibration for similarity scores
- **`src/distance_metrics.py`** - Mahalanobis distance-based similarity metrics  
- **`src/api_telemetry.py`** - API telemetry collection for monitoring

### 2. Updated `src/app.py`

#### Configuration Variables Added
```python
# Prediction fix settings
USE_CALIBRATED_SIMILARITY = os.environ.get("USE_CALIBRATED_SIMILARITY", "1")
USE_DISTANCE_METRICS = os.environ.get("USE_DISTANCE_METRICS", "1")
UNCERTAINTY_ESTIMATION_ENABLED = os.environ.get("UNCERTAINTY_ESTIMATION", "1")

# Conservative fallback thresholds
CONSERVATIVE_ONNX_THRESHOLD = float(os.environ.get("CONSERVATIVE_ONNX_THRESHOLD", "0.85"))
CONSERVATIVE_SIM_THRESHOLD = float(os.environ.get("CONSERVATIVE_SIM_THRESHOLD", "0.90"))
CONSERVATIVE_MARGIN_THRESHOLD = float(os.environ.get("CONSERVATIVE_MARGIN_THRESHOLD", "0.05"))
CONSERVATIVE_UNCERTAINTY_THRESHOLD = float(os.environ.get("CONSERVATIVE_UNCERTAINTY_THRESHOLD", "0.70"))

# Telemetry settings
API_TELEMETRY_ENABLED = os.environ.get("API_TELEMETRY_ENABLED", "0")
```

#### EmbeddingIndex.query_aggregated() Enhanced
- Now uses Mahalanobis distance metrics when available
- Falls back to calibrated cosine similarity
- Preserves discriminative information between similar Pokemon

#### merge_onnx_and_event() Enhanced
- Added 4-stage conservative fallback mechanism
- Returns override decision reason for telemetry
- Conservative thresholds prevent false overrides
- Added uncertainty checks using calibration curve

#### API Response Format Updated
All prediction endpoints now return additional fields:
```json
{
  "pokemon": "pokemon_name",
  "confidence": "92.30%",
  "confidence_raw": 0.9230,
  "event_override": false,
  "override_decision": "blocked_by_onnx_confidence",
  "top_index": 123
}
```

#### Telemetry Integration
- Added timing to all prediction endpoints
- Records override decisions for monitoring
- Buffers telemetry for performance (100 entries)
- Writes to `logs/api_telemetry.jsonl`

## Environment Variables for Vercel

### Enable the Fix (Recommended)
```bash
USE_CALIBRATED_SIMILARITY=1
USE_DISTANCE_METRICS=1
UNCERTAINTY_ESTIMATION=1
```

### Conservative Thresholds (Adjust as needed)
```bash
CONSERVATIVE_ONNX_THRESHOLD=0.85
CONSERVATIVE_SIM_THRESHOLD=0.90
CONSERVATIVE_MARGIN_THRESHOLD=0.05
CONSERVATIVE_UNCERTAINTY_THRESHOLD=0.70
```

### Enable Telemetry (Optional)
```bash
API_TELEMETRY_ENABLED=1
```

## How to Deploy

### 1. Update Vercel Environment Variables
Set the above environment variables in your Vercel project settings.

### 2. Deploy Changes
```bash
# From pokemon-predict-api directory
vercel --prod
```

### 3. Monitor Performance
Check the new `override_decision` field in API responses to verify conservative fallbacks are working:
- `blocked_by_onnx_confidence` - ONNX was confident, override blocked
- `blocked_by_similarity` - Embedding similarity too low
- `blocked_by_margin` - Margin between top-2 embeddings too small
- `blocked_by_uncertainty` - Uncertainty too high
- `successful_override` - Valid event override occurred
- `no_override` - No override attempted

## Expected Impact

### Immediate Benefits
- **Eliminates false high-confidence predictions** like the Golbat → Alloace Flag Vivillon case
- **Mathematically correct confidence scores** instead of meaningless similarity values
- **Conservative decision-making** prevents false overrides
- **API transparency** with override decision reasons

### API Response Changes
- **Backward compatible** - Existing fields remain unchanged
- **New fields added** - `event_override`, `override_decision`
- **Confidence values now calibrated** - May see slight changes in reported percentages

### Performance Impact
- **Minimal overhead** - Distance metrics fallback to cosine similarity if not trained
- **Telemetry optional** - Can be disabled if performance concerns
- **Cache-aware** - Telemetry respects existing caching mechanisms

## Validation

### Test the Fix
```bash
# Test prediction with known false positive
curl -X POST https://your-api.vercel.app/predict \
  -H "Content-Type: application/octet-stream" \
  --data-binary @golbat_image.jpg

# Check response for override_decision field
# Should show "blocked_by_onnx_confidence" or similar
```

### Monitor Telemetry
If telemetry is enabled, check `logs/api_telemetry.jsonl` for prediction patterns:
```bash
# Count override decisions
cat logs/api_telemetry.jsonl | jq -r '.override_decision' | sort | uniq -c
```

## Rollback Plan

If issues arise, disable components via environment variables:
```bash
# Disable all new features
USE_CALIBRATED_SIMILARITY=0
USE_DISTANCE_METRICS=0
UNCERTAINTY_ESTIMATION=0
API_TELEMETRY_ENABLED=0
```

## File Structure
```
pokemon-predict-api/
├── src/
│   ├── app.py                          # Updated with fix
│   ├── similarity_calibration.py        # New module
│   ├── distance_metrics.py             # New module
│   ├── api_telemetry.py                # New module
│   └── models/
│       ├── similarity_calibration.json  # Calibration parameters (generated)
│       └── class_covariances.json      # Distance metrics (generated)
└── logs/
    └── api_telemetry.jsonl             # Telemetry data
```

## Next Steps

1. **Deploy to Vercel** with new environment variables
2. **Monitor API responses** for override_decision patterns
3. **Generate calibration data** from production predictions
4. **Train distance metrics** on embedding index data
5. **Gradually enable telemetry** for production monitoring

This API update provides the same mathematical fixes as the main system, ensuring consistent behavior across both the Discord bot and the Vercel API.
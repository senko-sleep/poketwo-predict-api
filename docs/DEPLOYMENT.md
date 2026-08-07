# Deployment Guide for Pokemon Prediction API

## Vercel Data Transfer Optimization

This deployment includes several optimizations to minimize Vercel's Fast Origin Transfer usage:

### Current Status
- **Usage**: 9.45 GB / 10 GB (Pro plan)
- **Goal**: Prevent running out of data transfer without degrading bot performance

### Implemented Optimizations

1. **Supabase Caching Layer**
   - Caches prediction results to avoid repeated model inference
   - Expected 50-70% cache hit rate for repeated Pokemon spawns
   - Reduces computational load and response times

2. **Response Compression**
   - Gzip compression enabled via Flask-Compress
   - Reduces response size by ~70%
   - Configured with Brotli fallback

3. **Smart Caching Headers**
   - GET requests (health): 5-minute cache
   - POST requests (predictions): No cache to ensure accuracy
   - Vercel CDN leverages cache headers effectively

4. **Supabase Integration**
   - Offloads data storage from Vercel
   - Reduces database transfer costs
   - Provides analytics and monitoring

## Setup Instructions

### 1. Supabase Database Setup

1. Go to your Supabase project: https://supabase.com/dashboard
2. Navigate to SQL Editor
3. Run the schema file:
```sql
-- Copy contents from supabase_client/schema.sql
```

This creates:
- `pokemon_predictions` table for caching
- `prediction_stats` table for analytics
- Indexes for fast lookups
- Row Level Security policies

### 2. Environment Variables

Set these in Vercel or your `.env` file:

```bash
SUPABASE_URL=https://fkjizoetccgfmthotttm.supabase.co
SUPABASE_KEY=sb_publishable_5Wwz7UrMYkPP1RSuP-orZg_zsPjv23w
```

### 3. Deploy to Vercel

```bash
# From pokemon-predict-api directory
vercel deploy
```

### 4. Monitor Performance

Check the `/health` endpoint regularly:
```bash
curl https://your-api.vercel.app/health
```

Key metrics to watch:
- `cache_hit_rate`: Should increase over time (target: >50%)
- `supabase_connected`: Should be `true`
- `average_prediction_time_ms`: Should be stable

## Expected Data Transfer Reduction

Based on typical Pokemon bot usage patterns:

### Before Optimization
- Every prediction: ~500KB response
- No caching: 100% model inference
- No compression: Full response size

### After Optimization
- Cached predictions: ~150KB (70% smaller)
- 60% cache hit rate: 40% model inference
- Compression: All responses compressed

**Estimated reduction: 60-80% data transfer**

## Troubleshooting

### High Data Transfer Despite Optimizations

1. **Check cache hit rate**:
   - If low (<30%), may need to adjust cache TTL
   - Consider if Pokemon images are too variable

2. **Verify compression**:
   - Check response headers for `Content-Encoding: gzip`
   - Test with: `curl -I https://your-api.vercel.app/health`

3. **Monitor Supabase**:
   - Check if caching is working in Supabase dashboard
   - Verify table has entries from predictions

### Supabase Connection Issues

1. Verify credentials in environment variables
2. Check Supabase project is active
3. Ensure table schema was created correctly
4. Test health endpoint for `supabase_connected: true`

### Cache Not Working

1. Check Supabase table exists and has data
2. Verify image hashing is consistent
3. Check if cache hit rate increases over time
4. Review logs for cache errors

## Maintenance

### Regular Tasks

1. **Monitor cache hit rate** (weekly)
2. **Check data transfer usage** (daily)
3. **Review Supabase storage** (monthly)
4. **Clean old cache entries** (quarterly)

### Cache Cleanup

Run this in Supabase SQL editor to clean old entries:
```sql
-- Delete predictions older than 30 days
DELETE FROM pokemon_predictions 
WHERE created_at < NOW() - INTERVAL '30 days';
```

## Performance Tuning

### Adjust Cache Duration

Edit `api.py` to change cache behavior:
```python
# In add_header function
response.headers['Cache-Control'] = 'public, max-age=600'  # 10 minutes
```

### Optimize Model Loading

If model loading is slow, consider:
- Using smaller model files
- Implementing model caching
- Using edge functions for static models

### Database Indexing

The schema includes optimal indexes, but monitor query performance:
```sql
-- Check slow queries in Supabase dashboard
-- Add additional indexes if needed
```

## Cost Monitoring

### Vercel
- Monitor dashboard for data transfer usage
- Set up alerts for 80% threshold
- Consider upgrading if consistently near limits

### Supabase
- Free tier includes 500MB database
- Monitor storage usage in dashboard
- Clean old predictions regularly

## Emergency Measures

If data transfer approaches limit:

1. **Temporary Measures**:
   - Increase cache duration
   - Implement rate limiting
   - Use smaller response formats

2. **Long-term Solutions**:
   - Upgrade Vercel plan
   - Implement CDN caching
   - Use image optimization

## Success Metrics

Target metrics after deployment:
- Cache hit rate: >50%
- Data transfer reduction: >60%
- API response time: <500ms
- Supabase connection: 100% uptime
- Error rate: <1%
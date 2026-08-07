-- Supabase database schema for Pokemon prediction caching
-- Run this in your Supabase SQL editor to set up the tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Pokemon predictions cache table
CREATE TABLE IF NOT EXISTS pokemon_predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    image_hash TEXT NOT NULL UNIQUE,
    pokemon_name TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    prediction_time_ms FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on image_hash for fast lookups
CREATE INDEX IF NOT EXISTS idx_pokemon_predictions_image_hash 
ON pokemon_predictions(image_hash);

-- Create index on pokemon_name for analytics
CREATE INDEX IF NOT EXISTS idx_pokemon_predictions_pokemon_name 
ON pokemon_predictions(pokemon_name);

-- Create index on created_at for cleanup
CREATE INDEX IF NOT EXISTS idx_pokemon_predictions_created_at 
ON pokemon_predictions(created_at);

-- Prediction statistics table
CREATE TABLE IF NOT EXISTS prediction_stats (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    total_predictions INTEGER DEFAULT 0,
    average_prediction_time_ms FLOAT DEFAULT 0,
    cache_hit_rate FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on created_at for stats
CREATE INDEX IF NOT EXISTS idx_prediction_stats_created_at 
ON prediction_stats(created_at);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_pokemon_predictions_updated_at 
    BEFORE UPDATE ON pokemon_predictions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE pokemon_predictions ENABLE ROW LEVEL SECURITY;

-- Allow public read access (for API usage)
CREATE POLICY "Allow public read access on pokemon_predictions"
    ON pokemon_predictions FOR SELECT
    USING (true);

-- Allow public insert access (for API usage)
CREATE POLICY "Allow public insert access on pokemon_predictions"
    ON pokemon_predictions FOR INSERT
    WITH CHECK (true);

ALTER TABLE prediction_stats ENABLE ROW LEVEL SECURITY;

-- Allow public read access on prediction_stats
CREATE POLICY "Allow public read access on prediction_stats"
    ON prediction_stats FOR SELECT
    USING (true);

-- Allow public insert access on prediction_stats
CREATE POLICY "Allow public insert access on prediction_stats"
    ON prediction_stats FOR INSERT
    WITH CHECK (true);

-- Optional: Create a view for cache hit analytics
CREATE OR REPLACE VIEW cache_analytics AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT image_hash) as unique_predictions,
    AVG(prediction_time_ms) as avg_prediction_time_ms,
    AVG(confidence) as avg_confidence
FROM pokemon_predictions
GROUP BY DATE(created_at)
ORDER BY date DESC;
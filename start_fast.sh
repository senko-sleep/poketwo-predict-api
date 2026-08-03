#!/bin/bash
# Start script for maximum performance during high spawn events
# Disables event embedding for maximum speed, keeps caching enabled
# Enables feedback system for Poketwo reinforcement learning

export ENABLE_TTA=false
export ENABLE_GPU=true
export ENABLE_CACHE=true
export ENABLE_EVENT_EMBEDDING=false
export MAX_WORKERS=8
export CACHE_SIZE=2000
export FEEDBACK_ENABLED=true

echo "Starting Pokemon Prediction API in HIGH PERFORMANCE mode"
echo "Configuration:"
echo "  - TTA: $ENABLE_TTA"
echo "  - GPU: $ENABLE_GPU"
echo "  - Cache: $ENABLE_CACHE"
echo "  - Event Embedding: $ENABLE_EVENT_EMBEDDING"
echo "  - Max Workers: $MAX_WORKERS"
echo "  - Cache Size: $CACHE_SIZE"
echo "  - Feedback System: $FEEDBACK_ENABLED"

python app.py
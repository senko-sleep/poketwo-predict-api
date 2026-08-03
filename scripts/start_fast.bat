@echo off
REM Start script for maximum performance during high spawn events
REM Disables event embedding for maximum speed, keeps caching enabled
REM Enables feedback system for Poketwo reinforcement learning

set ENABLE_TTA=false
set ENABLE_GPU=true
set ENABLE_CACHE=true
set ENABLE_EVENT_EMBEDDING=false
set MAX_WORKERS=8
set CACHE_SIZE=2000
set FEEDBACK_ENABLED=true

echo Starting Pokemon Prediction API in HIGH PERFORMANCE mode
echo Configuration:
echo   - TTA: %ENABLE_TTA%
echo   - GPU: %ENABLE_GPU%
echo   - Cache: %ENABLE_CACHE%
echo   - Event Embedding: %ENABLE_EVENT_EMBEDDING%
echo   - Max Workers: %MAX_WORKERS%
echo   - Cache Size: %CACHE_SIZE%
echo   - Feedback System: %FEEDBACK_ENABLED%

cd ..
python run.py
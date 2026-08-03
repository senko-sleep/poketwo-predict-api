"""
Main entry point for the Pokemon Prediction API
"""
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app import app, session, ENABLE_TTA, ENABLE_GPU, MAX_WORKERS, ENABLE_CACHE, CACHE_SIZE, ENABLE_EVENT_EMBEDDING, FEEDBACK_ENABLED, PERCEPTUAL_CACHE_ENABLED

if __name__ == "__main__":
    # Verify model loaded
    if session is None:
        print("ERROR: Model failed to load. Server cannot start.")
        sys.exit(1)
    
    print("Model loaded successfully, starting server...")
    print(f"Performance configuration:")
    print(f"  - TTA (Test-Time Augmentation): {ENABLE_TTA}")
    print(f"  - GPU Acceleration: {ENABLE_GPU}")
    print(f"  - Max Workers: {MAX_WORKERS}")
    print(f"  - Caching: {ENABLE_CACHE}")
    print(f"  - Cache Size: {CACHE_SIZE}")
    print(f"  - Event Embedding: {ENABLE_EVENT_EMBEDDING}")
    print(f"  - Feedback System: {FEEDBACK_ENABLED}")
    print(f"  - Perceptual Cache: {PERCEPTUAL_CACHE_ENABLED}")
    
    # Print available routes
    print("\nAvailable routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.methods} {rule.rule}")
    
    # Run server with threaded mode for concurrent requests
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
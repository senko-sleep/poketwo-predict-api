"""
Modal deployment for Pokemon Prediction API
Uses FastAPI with ASGI for better compatibility
"""
import os
import io
import time
import sys
import threading
from pathlib import Path
from modal import Image, App, asgi_app
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# Add the src directory to the path for imports
# In Modal, files are mounted to /root/src and /root/models
sys.path.insert(0, "/root/src")

# Create Modal app
app = App("pokemon-predict-api")

# Define the image with all dependencies
# Updated: dj_rotom onnx_ceiling raised to 0.8 for necrozma confidence issue
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "onnxruntime>=1.17.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "supabase>=2.0.0",
        "python-multipart==0.0.6"
    )
    .add_local_dir(
        Path(__file__).parent.parent.parent / "src",
        remote_path="/root/src"
    )
    .add_local_dir(
        Path(__file__).parent.parent.parent / "models",
        remote_path="/root/models",
        # local index backups aren't needed in the container
        ignore=lambda p: any(part.startswith("backups") for part in Path(p).parts),
    )
)

# Create FastAPI app
fastapi_app = FastAPI()

# Global recognizer instance
recognizer = None
_recognizer_lock = threading.Lock()

def get_recognizer():
    """Get or create recognizer instance."""
    global recognizer
    if recognizer is None:
        with _recognizer_lock:
            if recognizer is None:
                from recognition import PokemonRecognizer
                recognizer = PokemonRecognizer(
                    model_path="/root/models/pokemon_cnn_v2.onnx",
                    labels_path="/root/models/labels_v2.json",
                    embedding_index_path="/root/models/event_embedding_index.npz",
                    embedding_meta_path="/root/models/event_embedding_meta.json",
                    event_label_config_path="/root/models/event_label_config.json",
                    event_labels_path="/root/models/event_labels.json"
                )
    return recognizer

@fastapi_app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        from supabase_client import supabase_manager
        recognizer = await run_in_threadpool(get_recognizer)
        return {
            "status": "healthy",
            "model_loaded": True,
            "platform": "Modal",
            "supabase_connected": await run_in_threadpool(supabase_manager.health_check),
            "recognition_version": __import__("recognition").RECOGNITION_VERSION,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }, 500

@fastapi_app.post("/predict")
async def predict(request: Request):
    """Predict Pokemon from image bytes (accepts raw bytes or JSON)."""
    try:
        from recognition import PokemonRecognizer
        from supabase_client import supabase_manager

        recognizer = await run_in_threadpool(get_recognizer)

        # Get the body first (can only read once)
        body = await request.body()
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type and body:
            # JSON format
            import json
            request_data = json.loads(body.decode('utf-8'))
            image_bytes = request_data.get("image_bytes")
            if isinstance(image_bytes, str):
                import base64
                image_bytes = base64.b64decode(image_bytes)
        else:
            # Raw bytes format - this is what the bot sends
            image_bytes = body

        if not image_bytes or len(image_bytes) < 100:
            raise HTTPException(status_code=400, detail="No valid image data provided")

        # Check Supabase cache first
        cached_result = await run_in_threadpool(supabase_manager.get_cached_prediction, image_bytes)
        if cached_result:
            return {
                "pokemon": cached_result["pokemon"],
                "pokemon_id": recognizer.event_labels.get(cached_result["pokemon"], {}).get("id"),
                "confidence": f"{cached_result['confidence'] * 100:.2f}%",
                "confidence_raw": cached_result["confidence_raw"],
                "prediction_time_ms": cached_result["prediction_time_ms"],
                "cached": True
            }

        # Cache miss - make prediction
        start_time = time.time()
        pokemon_name, confidence = await run_in_threadpool(recognizer.predict, image_bytes)
        end_time = time.time()

        # Calculate prediction time
        prediction_time = end_time - start_time

        # Cache the result in Supabase
        await run_in_threadpool(supabase_manager.cache_prediction,
            image_bytes, pokemon_name, confidence, prediction_time * 1000
        )

        return {
            "pokemon": pokemon_name,
            "pokemon_id": recognizer.event_labels.get(pokemon_name, {}).get("id"),
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "prediction_time_ms": round(prediction_time * 1000, 2),
            "cached": False
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/predict/url")
async def predict_url(request: Request):
    """Predict Pokemon from image URL."""
    try:
        import requests
        from recognition import PokemonRecognizer
        from supabase_client import supabase_manager

        recognizer = await run_in_threadpool(get_recognizer)

        request_data = await request.json()
        url = request_data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL not provided")

        # Download image
        response = await run_in_threadpool(requests.get, url, timeout=10)
        response.raise_for_status()
        image_bytes = response.content

        # Check Supabase cache first
        cached_result = await run_in_threadpool(supabase_manager.get_cached_prediction, image_bytes)
        if cached_result:
            return {
                "pokemon": cached_result["pokemon"],
                "pokemon_id": recognizer.event_labels.get(cached_result["pokemon"], {}).get("id"),
                "confidence": f"{cached_result['confidence'] * 100:.2f}%",
                "confidence_raw": cached_result["confidence_raw"],
                "prediction_time_ms": cached_result["prediction_time_ms"],
                "cached": True
            }

        # Cache miss - make prediction
        start_time = time.time()
        pokemon_name, confidence = await run_in_threadpool(recognizer.predict, image_bytes)
        end_time = time.time()

        # Calculate prediction time
        prediction_time = end_time - start_time

        # Cache the result in Supabase
        await run_in_threadpool(supabase_manager.cache_prediction,
            image_bytes, pokemon_name, confidence, prediction_time * 1000
        )

        return {
            "pokemon": pokemon_name,
            "pokemon_id": recognizer.event_labels.get(pokemon_name, {}).get("id"),
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "prediction_time_ms": round(prediction_time * 1000, 2),
            "cached": False
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount the FastAPI app as an ASGI app
@app.function(image=image)
@asgi_app()
def asgi_app_entrypoint():
    """ASGI app entrypoint for FastAPI."""
    return fastapi_app

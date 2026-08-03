#!/usr/bin/env python3
"""
Distributed prediction worker for high-throughput Pokémon prediction.
Each worker runs as a separate process with its own ONNX session.
"""
import asyncio
import time
import os
import logging
from aiohttp import web
import onnxruntime as ort
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ONNX_PATH = "models/pokemon_cnn_v2.onnx"
LABELS_PATH = "models/labels_v2.json"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
_SCALE = np.float32(1.0 / 255.0)


class PredictionWorker:
    """Prediction worker with dedicated ONNX session."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        logger.info(f"Initializing worker {worker_id}")
        
        # Load ONNX session with optimized settings
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # Single thread per worker (multi-process scaling)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load labels
        with open(LABELS_PATH, 'r') as f:
            self.labels = json.load(f)
        
        # Thread pool for CPU-bound preprocessing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Worker {worker_id} initialized with providers: {providers}")

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image bytes for ONNX inference."""
        import cv2
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) * _SCALE
        img = (img - _MEAN) / _STD
        
        # Add batch dimension
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        return img

    def _predict_sync(self, image_bytes: bytes) -> dict:
        """Synchronous prediction (runs in thread pool)."""
        start = time.time()
        
        try:
            # Preprocess
            img = self._preprocess_image(image_bytes)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: img})
            predictions = outputs[0][0]
            
            # Get top prediction
            top_idx = np.argmax(predictions)
            confidence = float(predictions[top_idx])
            pokemon_name = self.labels[top_idx]
            
            latency_ms = (time.time() - start) * 1000
            
            return {
                "pokemon": pokemon_name,
                "confidence": f"{confidence * 100:.2f}%",
                "worker_id": self.worker_id,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.error(f"Prediction error in worker {self.worker_id}: {e}")
            raise

    async def predict(self, request: web.Request) -> web.Response:
        """Async prediction endpoint."""
        try:
            # Read image bytes
            image_bytes = await request.read()
            
            if len(image_bytes) < 1024:
                return web.json_response(
                    {"error": "Image too small"}, 
                    status=400
                )
            
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                return web.json_response(
                    {"error": "Image too large"}, 
                    status=413
                )
            
            # Run prediction in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._predict_sync, 
                image_bytes
            )
            
            logger.debug(f"Worker {self.worker_id}: {result['pokemon']} ({result['confidence']}) in {result['latency_ms']:.1f}ms")
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Prediction endpoint error in worker {self.worker_id}: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )

    async def health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "worker_id": self.worker_id,
            "status": "healthy",
            "providers": self.session.get_providers()
        })


async def create_app(worker_id: int) -> web.Application:
    """Create aiohttp application for prediction worker."""
    worker = PredictionWorker(worker_id)
    
    app = web.Application()
    app.add_routes([
        web.post('/predict', worker.predict),
        web.get('/health', worker.health),
    ])
    
    return app


if __name__ == "__main__":
    import json
    
    worker_id = int(os.environ.get("WORKER_ID", "0"))
    port = 8080 + worker_id
    
    app = asyncio.run(create_app(worker_id))
    
    logger.info(f"Starting prediction worker {worker_id} on port {port}")
    web.run_app(app, host="0.0.0.0", port=port)

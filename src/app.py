"""
Pokemon ONNX Prediction API
A standalone service for Pokemon spawn prediction using ONNX model.
Optimized for high-volume spawn processing with Poketwo reinforcement learning.
"""
import os
import sys
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
from PIL import Image
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from poketwo_feedback import PoketwoFeedback
from perceptual_cache import PerceptualCache

# Get the parent directory (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "pokemon_cnn_v2.onnx"))
LABELS_PATH = os.environ.get("LABELS_PATH", os.path.join(PROJECT_ROOT, "models", "labels_v2.json"))
EVENT_EMBEDDING_INDEX_PATH = os.environ.get("EVENT_EMBEDDING_INDEX_PATH", os.path.join(PROJECT_ROOT, "models", "event_embedding_index.npz"))
EVENT_EMBEDDING_META_PATH = os.environ.get("EVENT_EMBEDDING_META_PATH", os.path.join(PROJECT_ROOT, "models", "event_embedding_meta.json"))
EVENT_MANIFEST_PATH = os.environ.get("EVENT_MANIFEST_PATH", os.path.join(PROJECT_ROOT, "models", "event_labels.json"))
EVENT_LABEL_CONFIG_PATH = os.environ.get("EVENT_LABEL_CONFIG_PATH", os.path.join(PROJECT_ROOT, "models", "event_label_config.json"))
INPUT_SIZE = 224

# Performance settings
ENABLE_TTA = os.environ.get("ENABLE_TTA", "false").lower() in ("true", "1", "yes")
ENABLE_GPU = os.environ.get("ENABLE_GPU", "true").lower() in ("true", "1", "yes")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "true").lower() in ("true", "1", "yes")
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "1000"))
ENABLE_EVENT_EMBEDDING = os.environ.get("ENABLE_EVENT_EMBEDDING", "true").lower() in ("true", "1", "yes")

print(f"Loading ONNX model from {ONNX_MODEL_PATH}...")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')[:10]}")
print(f"Performance settings: TTA={ENABLE_TTA}, GPU={ENABLE_GPU}, Workers={MAX_WORKERS}, Cache={ENABLE_CACHE}, EventEmbedding={ENABLE_EVENT_EMBEDDING}")

# Set up execution providers
providers = ['CPUExecutionProvider']
if ENABLE_GPU:
    try:
        # Try CUDA provider first
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("GPU acceleration enabled (CUDA)")
        # Try TensorRT
        elif 'TensorrtExecutionProvider' in ort.get_available_providers():
            providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
            print("GPU acceleration enabled (TensorRT)")
        else:
            print("GPU requested but no GPU providers available, using CPU")
    except Exception as e:
        print(f"GPU setup failed: {e}, using CPU")

# Load ONNX model at module level (before app creation)
try:
    if os.path.exists(ONNX_MODEL_PATH):
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        print(f"Model loaded successfully with providers: {session.get_providers()}")
    else:
        print(f"ERROR: Model file not found at {ONNX_MODEL_PATH}")
        session = None
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    session = None

# Embedding Index class for event pokemon detection
class EmbeddingIndex:
    """Simple embedding index for event pokemon detection."""
    
    def __init__(self, index_path: str, meta_path: str):
        self._index_path = index_path
        self._meta_path = meta_path
        self._embeddings = None
        self._labels = None
        self._meta = {}
        self._loaded = False
        self._unique_labels = None
        self._inverse = None
        self._load()
    
    def _load(self):
        """Load index from disk."""
        if os.path.exists(self._index_path):
            try:
                with np.load(self._index_path, allow_pickle=True) as data:
                    self._embeddings = data["embeddings"].astype(np.float16)
                    self._labels = data["labels"].copy()
                print(f"Loaded event embedding index: {len(self._labels)} entries")
            except Exception as e:
                print(f"Failed to load event embedding index: {e}")
                self._embeddings = None
                self._labels = None
        
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "r") as f:
                    self._meta = json.load(f)
            except Exception:
                self._meta = {}
        
        self._rebuild_groupby()
        self._loaded = True
    
    def _rebuild_groupby(self):
        """Recompute groupby cache."""
        if self._labels is not None and len(self._labels) > 0:
            self._unique_labels, self._inverse = np.unique(
                self._labels, return_inverse=True
            )
        else:
            self._unique_labels = None
            self._inverse = None
    
    @property
    def size(self):
        return len(self._labels) if self._labels is not None else 0
    
    @property
    def unique_labels(self):
        return len(self._unique_labels) if self._unique_labels is not None else 0
    
    @property
    def label_counts(self):
        if self._unique_labels is not None and self._inverse is not None:
            counts = {}
            for label in self._unique_labels:
                counts[label] = np.sum(self._inverse == np.where(self._unique_labels == label)[0][0])
            return counts
        return {}
    
    def query_aggregated(self, query_vec: np.ndarray, top_k: int = 5) -> list:
        """Query the index and aggregate results by label."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        
        # Compute cosine similarity - normalize query vector first
        query_vec = query_vec.astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        # Compute dot product for cosine similarity (embeddings are already normalized)
        similarities = np.dot(self._embeddings.astype(np.float32), query_vec)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Aggregate by label
        label_scores = {}
        for idx in top_indices:
            label = self._labels[idx]
            score = float(similarities[idx])
            if label not in label_scores:
                label_scores[label] = []
            label_scores[label].append(score)
        
        # Average scores per label
        results = []
        for label, scores in label_scores.items():
            avg_score = sum(scores) / len(scores)
            results.append((label, avg_score))
        
        # Sort by average score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# Load event label config
event_label_config = {}
if os.path.exists(EVENT_LABEL_CONFIG_PATH):
    try:
        with open(EVENT_LABEL_CONFIG_PATH, "r") as f:
            raw_cfg = json.load(f)
            meta_keys = {"_comment", "_fields", "_strategy"}
            event_label_config = {k: v for k, v in raw_cfg.items() if k not in meta_keys and not k.startswith("_")}
        print(f"Loaded event label config: {len(event_label_config)} label overrides")
    except Exception as e:
        print(f"Failed to load event label config: {e}")

# Load event embedding index
event_embedding_index = None
event_labels = set()

try:
    if os.path.exists(EVENT_EMBEDDING_INDEX_PATH):
        event_embedding_index = EmbeddingIndex(EVENT_EMBEDDING_INDEX_PATH, EVENT_EMBEDDING_META_PATH)
        event_labels = set(event_embedding_index.label_counts.keys())
        print(f"Event embedding index loaded: {event_embedding_index.size} entries, {event_embedding_index.unique_labels} unique labels")
    else:
        print(f"Event embedding index not found at {EVENT_EMBEDDING_INDEX_PATH}")
except Exception as e:
    print(f"Failed to load event embedding index: {e}")
    event_embedding_index = None

# Load labels at module level
labels = []
label_to_index = {}
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    
    # Handle different label formats
    if isinstance(labels_data, list):
        labels = labels_data
        label_to_index = {label: i for i, label in enumerate(labels)}
    elif isinstance(labels_data, dict):
        if all(str(k).isdigit() for k in labels_data.keys()):
            labels = [labels_data[str(i)] for i in range(len(labels_data))]
            label_to_index = {label: i for i, label in enumerate(labels)}
        else:
            labels = sorted(labels_data.keys(), key=lambda k: labels_data[k].get("index", 0))
            label_to_index = {label: i for i, label in enumerate(labels)}
    
    print(f"Loaded {len(labels)} labels")
else:
    print(f"Warning: Labels file not found at {LABELS_PATH}")

app = Flask(__name__)
CORS(app)

# Performance optimizations
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Faster JSON responses
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize Poketwo feedback system
FEEDBACK_ENABLED = os.environ.get("FEEDBACK_ENABLED", "true").lower() in ("true", "1", "yes")
feedback_system = PoketwoFeedback() if FEEDBACK_ENABLED else None

if FEEDBACK_ENABLED:
    print("Poketwo feedback system enabled")

# Initialize perceptual cache
PERCEPTUAL_CACHE_ENABLED = os.environ.get("PERCEPTUAL_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
perceptual_cache = None

if PERCEPTUAL_CACHE_ENABLED:
    try:
        cache_file = os.path.join(PROJECT_ROOT, "perceptual_cache.pkl")
        perceptual_cache = PerceptualCache(
            cache_file=cache_file,
            canonical_size=64,
            hamming_threshold=5,
            max_cache_size=10000
        )
        print("Perceptual cache enabled")
    except Exception as e:
        print(f"Failed to initialize perceptual cache: {e}")
        perceptual_cache = None


def preprocess_image(image_bytes):
    """Preprocess image for ONNX model - optimized without TTA by default"""
    # Load image
    try:
        # Handle both bytes and BytesIO objects
        if isinstance(image_bytes, bytes):
            bio = io.BytesIO(image_bytes)
        elif hasattr(image_bytes, 'read'):
            bio = image_bytes
            bio.seek(0)
        else:
            raise ValueError(f"Unsupported type for image_bytes: {type(image_bytes)}")
        
        image = Image.open(bio)
    except Exception as e:
        print(f"Failed to open image: {e}")
        raise
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to input size
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    
    # Normalize setup
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    if ENABLE_TTA:
        # Forward view
        fwd_array = np.array(image, dtype=np.float32) / 255.0
        fwd = np.transpose((fwd_array - mean) / std, (2, 0, 1))
        
        # Flipped view
        flip_img = image.transpose(Image.FLIP_LEFT_RIGHT)
        flip_array = np.array(flip_img, dtype=np.float32) / 255.0
        flip = np.transpose((flip_array - mean) / std, (2, 0, 1))
        
        # Batch (2, 3, 224, 224)
        return np.stack([fwd, flip], axis=0).astype(np.float32)
    else:
        # Single forward pass only (much faster)
        array = np.array(image, dtype=np.float32) / 255.0
        normalized = np.transpose((array - mean) / std, (2, 0, 1))
        # Single batch (1, 3, 224, 224)
        return np.expand_dims(normalized, axis=0).astype(np.float32)

def is_event_label(label: str) -> bool:
    """Check if a label is an event pokemon."""
    return label.lower() in (l.lower() for l in event_labels)

def merge_onnx_and_event(
    onnx_name: str,
    onnx_conf: float,
    embed_vec: np.ndarray,
) -> tuple[str, float, bool]:
    """Merge ONNX prediction with event embedding index results."""
    if event_embedding_index is None or event_embedding_index.size == 0:
        return onnx_name, onnx_conf, False
    
    try:
        ev_results = event_embedding_index.query_aggregated(embed_vec, top_k=5)
    except Exception as e:
        return onnx_name, onnx_conf, False
    
    if not ev_results:
        return onnx_name, onnx_conf, False
    
    # Filter out ignored labels
    ignore_labels = {"backgrounds", "backgounds", "background", "bg"}
    ev_results = [(l, s) for l, s in ev_results if l.lower() not in ignore_labels]
    if not ev_results:
        return onnx_name, onnx_conf, False
    
    best_label, best_sim = ev_results[0]
    second_sim = ev_results[1][1] if len(ev_results) > 1 else 0.0
    margin = best_sim - second_sim
    
    cfg = event_label_config.get(best_label, {})
    min_sim = cfg.get("min_sim", 0.93)
    min_margin = cfg.get("min_margin", 0.005)
    onnx_ceiling = cfg.get("onnx_ceiling", 1.0)
    
    onnx_is_event = is_event_label(onnx_name)
    
    # If ONNX already predicts an event label, check if event index agrees
    if onnx_is_event:
        if best_sim >= min_sim and margin >= min_margin:
            if best_sim > onnx_conf:
                if best_sim >= 0.99:
                    boosted_conf = min(1.0, max(onnx_conf, best_sim * 0.9))
                    return best_label, boosted_conf, True
                return best_label, onnx_conf, True
        return onnx_name, onnx_conf, False
    
    # ONNX predicts non-event
    is_base_variant = onnx_name.lower() in best_label.lower()

    if best_sim >= min_sim and margin >= min_margin:
        if onnx_conf >= onnx_ceiling and not is_base_variant:
            return onnx_name, onnx_conf, False
        if best_sim >= 0.99:
            boosted_conf = min(1.0, max(onnx_conf, best_sim * 0.9))
            return best_label, boosted_conf, True
        return best_label, onnx_conf, True
    
    if best_sim >= 0.99 and margin >= min_margin:
        if onnx_conf >= onnx_ceiling and not is_base_variant:
            return onnx_name, onnx_conf, False
        if best_sim >= 0.99:
            boosted_conf = min(1.0, max(onnx_conf, best_sim * 0.9))
            return best_label, boosted_conf, True
        return best_label, onnx_conf, True

    return onnx_name, onnx_conf, False


def run_onnx_prediction(image_bytes):
    """Run ONNX prediction on image bytes (without caching)."""
    # Preprocess image (single pass by default for speed)
    input_data = preprocess_image(image_bytes)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_data})
    
    # Handle different batch sizes based on TTA setting
    if ENABLE_TTA:
        # Aggregate predictions (logits) across TTA batch
        logits = (outputs[0][0] + outputs[0][1]) * 0.5
    else:
        # Single forward pass
        logits = outputs[0][0]
    
    # Apply softmax to get probabilities
    exp_predictions = np.exp(logits - np.max(logits))
    probabilities = exp_predictions / np.sum(exp_predictions)
    
    # Get top prediction
    top_index = np.argmax(probabilities)
    confidence = float(probabilities[top_index])
    
    # Get label name
    if top_index < len(labels):
        pokemon_name = labels[top_index]
    else:
        pokemon_name = "unknown"
    
    # Apply confidence adjustment from feedback system
    if feedback_system:
        adjustment = feedback_system.get_confidence_adjustment(pokemon_name)
        confidence = max(0.0, min(1.0, confidence + adjustment))
    
    # Apply event pokemon detection if embedding index is available and enabled
    event_override = False
    if ENABLE_EVENT_EMBEDDING and event_embedding_index is not None and event_embedding_index.size > 0:
        # Use the raw logits as embedding vector (before softmax)
        embed_vec = logits.astype(np.float32)
        pokemon_name, confidence, event_override = merge_onnx_and_event(
            pokemon_name, confidence, embed_vec
        )
    
    return {
        "pokemon": pokemon_name,
        "confidence": f"{confidence * 100:.2f}%",
        "confidence_raw": confidence,
        "top_index": int(top_index),
        "event_override": event_override
    }


def download_image(url):
    """Download image from URL with connection pooling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)  # Reduced timeout
        if response.status_code == 200:
            content = response.content
            return content
        return None
    except Exception as e:
        return None

# Simple LRU cache for predictions
prediction_cache = {}
cache_lock = threading.Lock()
cache_hits = 0
cache_misses = 0

def get_cache_key(image_bytes):
    """Generate cache key from image bytes"""
    return hashlib.md5(image_bytes).hexdigest()

def get_cached_prediction(cache_key):
    """Get cached prediction if available"""
    global cache_hits, cache_misses
    if not ENABLE_CACHE:
        return None
    with cache_lock:
        result = prediction_cache.get(cache_key)
        if result is not None:
            cache_hits += 1
            return result
        cache_misses += 1
        return None

def set_cached_prediction(cache_key, result):
    """Cache prediction result"""
    if not ENABLE_CACHE:
        return
    with cache_lock:
        if len(prediction_cache) >= CACHE_SIZE:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(prediction_cache))
            del prediction_cache[oldest_key]
        prediction_cache[cache_key] = result

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    global cache_hits, cache_misses
    health_data = {
        "status": "ok",
        "model_loaded": session is not None,
        "num_labels": len(labels),
        "event_embedding_loaded": event_embedding_index is not None,
        "event_embedding_size": event_embedding_index.size if event_embedding_index else 0,
        "event_labels_count": len(event_labels),
        "performance": {
            "tta": ENABLE_TTA,
            "gpu": ENABLE_GPU,
            "workers": MAX_WORKERS,
            "cache": ENABLE_CACHE,
            "cache_size": CACHE_SIZE,
            "cache_entries": len(prediction_cache),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "event_embedding": ENABLE_EVENT_EMBEDDING
        },
        "feedback": {
            "enabled": FEEDBACK_ENABLED,
            "total_feedback": feedback_system.get_feedback_stats()["total_feedback"] if feedback_system else 0
        },
        "perceptual_cache": {
            "enabled": PERCEPTUAL_CACHE_ENABLED,
            "statistics": perceptual_cache.get_statistics() if perceptual_cache else None
        }
    }
    return jsonify(health_data)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict Pokemon from image bytes with perceptual caching"""
    if session is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Get image bytes from request
        image_bytes = request.get_data()
        
        if not image_bytes:
            return jsonify({"error": "No image data provided"}), 400
        
        # Try perceptual cache first (if enabled)
        if perceptual_cache:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Check if client wants to force model inference
                force_model = request.headers.get('X-Force-Model', '').lower() in ('true', '1', 'yes')
                
                prediction = perceptual_cache.predict(
                    image, 
                    onnx_predict_func=lambda img: run_onnx_prediction(image_bytes),
                    force_model=force_model
                )
                
                if prediction.get('cache_hit') and not force_model:
                    return jsonify(prediction)
            except Exception as e:
                print(f"Perceptual cache error: {e}")
                # Fall through to regular prediction
        
        # Check regular cache (hash-based)
        cache_key = get_cache_key(image_bytes)
        cached_result = get_cached_prediction(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Run ONNX prediction
        result = run_onnx_prediction(image_bytes)
        
        # Cache the result
        set_cached_prediction(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/url", methods=["POST"])
def predict_url():
    """Predict Pokemon from image URL (POST with JSON body) - optimized"""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL not provided"}), 400
    
    url = data["url"]
    image_bytes = download_image(url)
    
    if not image_bytes:
        return jsonify({"error": "Failed to download image"}), 400
    
    if session is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Check cache first
        cache_key = get_cache_key(image_bytes)
        cached_result = get_cached_prediction(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Preprocess image (single pass by default for speed)
        input_data = preprocess_image(image_bytes)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_data})
        
        # Handle different batch sizes based on TTA setting
        if ENABLE_TTA:
            # Aggregate predictions (logits) across TTA batch
            logits = (outputs[0][0] + outputs[0][1]) * 0.5
        else:
            # Single forward pass
            logits = outputs[0][0]
        
        # Apply softmax to get probabilities
        exp_predictions = np.exp(logits - np.max(logits))
        probabilities = exp_predictions / np.sum(exp_predictions)
        
        # Get top prediction
        top_index = np.argmax(probabilities)
        confidence = float(probabilities[top_index])
        
        # Get label name
        if top_index < len(labels):
            pokemon_name = labels[top_index]
        else:
            pokemon_name = "unknown"
        
        # Apply confidence adjustment from feedback system
        if feedback_system:
            adjustment = feedback_system.get_confidence_adjustment(pokemon_name)
            confidence = max(0.0, min(1.0, confidence + adjustment))
        
        # Apply event pokemon detection if embedding index is available and enabled
        event_override = False
        if ENABLE_EVENT_EMBEDDING and event_embedding_index is not None and event_embedding_index.size > 0:
            # Use the raw logits as embedding vector (before softmax)
            embed_vec = logits.astype(np.float32)
            pokemon_name, confidence, event_override = merge_onnx_and_event(
                pokemon_name, confidence, embed_vec
            )
        
        result = {
            "pokemon": pokemon_name,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "top_index": int(top_index),
            "event_override": event_override
        }
        
        # Cache the result
        set_cached_prediction(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_details = str(e)
        traceback_str = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        print(f"Traceback: {traceback_str}")
        return jsonify({"error": error_details, "traceback": traceback_str}), 500


@app.route("/api/predict", methods=["GET", "POST"])
def predict_api():
    """Predict Pokemon from image bytes (POST) or image URL (GET with url query param) - optimized"""
    if request.method == "POST":
        image_bytes = request.get_data()
        if not image_bytes:
            return jsonify({"error": "No image data provided"}), 400
    else:
        url = request.args.get("url")
        if not url:
            return jsonify({"error": "URL query parameter required"}), 400
        image_bytes = download_image(url)
        if not image_bytes:
            return jsonify({"error": "Failed to download image"}), 400

    if session is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Check cache first
        cache_key = get_cache_key(image_bytes)
        cached_result = get_cached_prediction(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Preprocess image (single pass by default for speed)
        input_data = preprocess_image(image_bytes)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_data})
        
        # Handle different batch sizes based on TTA setting
        if ENABLE_TTA:
            # Aggregate predictions (logits) across TTA batch
            logits = (outputs[0][0] + outputs[0][1]) * 0.5
        else:
            # Single forward pass
            logits = outputs[0][0]
        
        # Apply softmax to get probabilities
        exp_predictions = np.exp(logits - np.max(logits))
        probabilities = exp_predictions / np.sum(exp_predictions)
        top_index = np.argmax(probabilities)
        confidence = float(probabilities[top_index])

        if top_index < len(labels):
            pokemon_name = labels[top_index]
        else:
            pokemon_name = "unknown"
        
        # Apply confidence adjustment from feedback system
        if feedback_system:
            adjustment = feedback_system.get_confidence_adjustment(pokemon_name)
            confidence = max(0.0, min(1.0, confidence + adjustment))

        # Apply event pokemon detection if embedding index is available and enabled
        event_override = False
        if ENABLE_EVENT_EMBEDDING and event_embedding_index is not None and event_embedding_index.size > 0:
            # Use the raw logits as embedding vector (before softmax)
            embed_vec = logits.astype(np.float32)
            pokemon_name, confidence, event_override = merge_onnx_and_event(
                pokemon_name, confidence, embed_vec
            )
        
        result = {
            "pokemon": pokemon_name,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "top_index": int(top_index),
            "event_override": event_override
        }
        
        # Cache the result
        set_cached_prediction(cache_key, result)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """Receive Poketwo catch message feedback for reinforcement learning"""
    if not feedback_system:
        return jsonify({"error": "Feedback system disabled"}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        message = data.get("message")
        predicted_pokemon = data.get("predicted_pokemon")
        predicted_confidence = data.get("predicted_confidence")
        
        if not message or not predicted_pokemon:
            return jsonify({"error": "Missing required fields: message, predicted_pokemon"}), 400
        
        # Process the catch message
        result = feedback_system.process_catch_message(
            message, predicted_pokemon, predicted_confidence or 0.0
        )
        
        return jsonify({
            "status": "success",
            "feedback": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feedback/stats", methods=["GET"])
def feedback_stats():
    """Get feedback system statistics"""
    if not feedback_system:
        return jsonify({"error": "Feedback system disabled"}), 503
    
    stats = feedback_system.get_feedback_stats()
    return jsonify(stats)

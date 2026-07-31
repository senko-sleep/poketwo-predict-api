"""
Pokemon ONNX Prediction API
A standalone service for Pokemon spawn prediction using ONNX model.
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

# Configuration
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "pokemon_cnn_v2.onnx")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels_v2.json")
EVENT_EMBEDDING_INDEX_PATH = os.environ.get("EVENT_EMBEDDING_INDEX_PATH", "event_embedding_index.npz")
EVENT_EMBEDDING_META_PATH = os.environ.get("EVENT_EMBEDDING_META_PATH", "event_embedding_meta.json")
EVENT_MANIFEST_PATH = os.environ.get("EVENT_MANIFEST_PATH", "event_labels.json")
INPUT_SIZE = 224

print(f"Loading ONNX model from {ONNX_MODEL_PATH}...")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')[:10]}")

# Load ONNX model at module level (before app creation)
try:
    if os.path.exists(ONNX_MODEL_PATH):
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"Model loaded successfully")
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


def preprocess_image(image_bytes):
    """Preprocess image for ONNX model"""
    # Load image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to input size
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Transpose to NCHW format (batch, channels, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)

def is_event_label(label: str) -> bool:
    """Check if a label is an event pokemon."""
    return label.lower() in (l.lower() for l in event_labels)

def merge_onnx_and_event(
    onnx_name: str,
    onnx_conf: float,
    embed_vec: np.ndarray,
) -> tuple[str, float, bool]:
    """Merge ONNX prediction with event embedding index results."""
    print(f"DEBUG merge_onnx_and_event: onnx_name={onnx_name}, onnx_conf={onnx_conf}, embed_vec.shape={embed_vec.shape}")
    
    if event_embedding_index is None or event_embedding_index.size == 0:
        print(f"DEBUG: Event embedding index is None or empty")
        return onnx_name, onnx_conf, False
    
    try:
        ev_results = event_embedding_index.query_aggregated(embed_vec, top_k=5)
        print(f"DEBUG: Event embedding results: {ev_results[:3]}")
    except Exception as e:
        print(f"Event embedding check failed: {e}")
        return onnx_name, onnx_conf, False
    
    if not ev_results:
        print(f"DEBUG: No event embedding results")
        return onnx_name, onnx_conf, False
    
    # Filter out ignored labels
    ignore_labels = {"backgrounds", "backgounds", "background", "bg"}
    ev_results = [(l, s) for l, s in ev_results if l.lower() not in ignore_labels]
    if not ev_results:
        print(f"DEBUG: All results filtered as ignored labels")
        return onnx_name, onnx_conf, False
    
    best_label, best_sim = ev_results[0]
    second_sim = ev_results[1][1] if len(ev_results) > 1 else 0.0
    margin = best_sim - second_sim
    
    print(f"DEBUG: best_label={best_label}, best_sim={best_sim}, margin={margin}")
    
    # Event detection thresholds - extremely strict to prevent artificial predictions
    # Only override when there's clear, unambiguous evidence
    min_sim = 0.98  # Require extremely high similarity for event override
    min_margin = 0.20  # Require very clear margin over second best (embeddings are clustered)
    onnx_ceiling = 0.40  # ONNX confidence above this blocks event override (only override when ONNX is very uncertain)
    
    onnx_is_event = is_event_label(onnx_name)
    print(f"DEBUG: onnx_is_event={onnx_is_event}")
    
    # If ONNX already predicts an event label, check if event index agrees
    if onnx_is_event:
        if best_sim >= min_sim and margin >= min_margin:
            if best_sim > onnx_conf:
                print(f"Event refinement: {onnx_name}@{onnx_conf:.3f} -> {best_label}@{best_sim:.4f}")
                return best_label, onnx_conf, True
        return onnx_name, onnx_conf, False
    
    # ONNX predicts non-event, event index has match
    # Only override if event embedding has EXTREMELY high confidence (clear signal it's an event)
    # This catches cases where ONNX is confidently wrong but event embedding knows it's an event
    if best_sim >= 0.98 and margin >= 0.05:
        print(f"Event override (high confidence): {onnx_name}@{onnx_conf:.3f} -> {best_label}@{best_sim:.4f}")
        return best_label, onnx_conf, True
    
    print(f"Event override blocked: insufficient evidence (sim={best_sim:.4f}, margin={margin:.4f})")
    return onnx_name, onnx_conf, False


def download_image(url):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": session is not None,
        "num_labels": len(labels),
        "event_embedding_loaded": event_embedding_index is not None,
        "event_embedding_size": event_embedding_index.size if event_embedding_index else 0,
        "event_labels_count": len(event_labels)
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Predict Pokemon from image bytes"""
    if session is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Get image bytes from request
        image_bytes = request.get_data()
        
        if not image_bytes:
            return jsonify({"error": "No image data provided"}), 400
        
        # Preprocess image
        input_data = preprocess_image(image_bytes)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_data})
        
        # Get predictions (logits)
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
        
        # Apply event pokemon detection if embedding index is available
        event_override = False
        if event_embedding_index is not None and event_embedding_index.size > 0:
            # Use the raw logits as embedding vector (before softmax)
            embed_vec = logits.astype(np.float32)
            pokemon_name, confidence, event_override = merge_onnx_and_event(
                pokemon_name, confidence, embed_vec
            )
        
        return jsonify({
            "pokemon": pokemon_name,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "top_index": int(top_index),
            "event_override": event_override
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/url", methods=["POST"])
def predict_url():
    """Predict Pokemon from image URL (POST with JSON body)"""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL not provided"}), 400
    
    url = data["url"]
    image_bytes = download_image(url)
    
    if not image_bytes:
        return jsonify({"error": "Failed to download image"}), 400
    
    # Use the same prediction logic
    return predict()


@app.route("/api/predict", methods=["GET", "POST"])
def predict_api():
    """Predict Pokemon from image bytes (POST) or image URL (GET with url query param)"""
    if request.method == "POST":
        image_bytes = request.get_data()
        if not image_bytes:
            return jsonify({"error": "No image data provided"}), 400
        try:
            input_data = preprocess_image(image_bytes)
        except Exception as e:
            return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 400
    else:
        url = request.args.get("url")
        if not url:
            return jsonify({"error": "URL query parameter required"}), 400
        image_bytes = download_image(url)
        if not image_bytes:
            return jsonify({"error": "Failed to download image"}), 400
        try:
            input_data = preprocess_image(image_bytes)
        except Exception as e:
            return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 400

    if session is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_data})

        logits = outputs[0][0]
        exp_predictions = np.exp(logits - np.max(logits))
        probabilities = exp_predictions / np.sum(exp_predictions)
        top_index = np.argmax(probabilities)
        confidence = float(probabilities[top_index])

        if top_index < len(labels):
            pokemon_name = labels[top_index]
        else:
            pokemon_name = "unknown"

        # Apply event pokemon detection if embedding index is available
        event_override = False
        if event_embedding_index is not None and event_embedding_index.size > 0:
            # Use the raw logits as embedding vector (before softmax)
            embed_vec = logits.astype(np.float32)
            pokemon_name, confidence, event_override = merge_onnx_and_event(
                pokemon_name, confidence, embed_vec
            )

        return jsonify({
            "pokemon": pokemon_name,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "top_index": int(top_index),
            "event_override": event_override
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Verify model loaded
    if session is None:
        print("ERROR: Model failed to load. Server cannot start.")
        sys.exit(1)
    
    print("Model loaded successfully, starting server...")
    
    # Run server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

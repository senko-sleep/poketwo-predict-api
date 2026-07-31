"""
Pokemon ONNX Prediction API
A standalone service for Pokemon spawn prediction using ONNX model.
"""
import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
from PIL import Image
import requests

# Configuration
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "pokemon_cnn_v2.onnx")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels_v2.json")
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

# Load labels at module level
labels = []
label_to_index = {}
import json
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
        "num_labels": len(labels)
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
        
        # Get predictions
        predictions = outputs[0][0]
        
        # Apply softmax to get probabilities
        exp_predictions = np.exp(predictions - np.max(predictions))
        probabilities = exp_predictions / np.sum(exp_predictions)
        
        # Get top prediction
        top_index = np.argmax(probabilities)
        confidence = float(probabilities[top_index])
        
        # Get label name
        if top_index < len(labels):
            pokemon_name = labels[top_index]
        else:
            pokemon_name = "unknown"
        
        return jsonify({
            "pokemon": pokemon_name,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "top_index": int(top_index)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/url", methods=["POST"])
def predict_url():
    """Predict Pokemon from image URL"""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL not provided"}), 400
    
    url = data["url"]
    image_bytes = download_image(url)
    
    if not image_bytes:
        return jsonify({"error": "Failed to download image"}), 400
    
    # Use the same prediction logic
    from flask import Response
    return Response(predict().get_data(), mimetype='application/json')


if __name__ == "__main__":
    # Verify model loaded
    if session is None:
        print("ERROR: Model failed to load. Server cannot start.")
        sys.exit(1)
    
    print("Model loaded successfully, starting server...")
    
    # Run server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

"""
Simple Pokemon Prediction API
Clean REST API for Pokemon image recognition.
"""
import os
import io
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from recognition import PokemonRecognizer

app = Flask(__name__)
CORS(app)

# Initialize recognizer
recognizer = None

# Statistics tracking
prediction_stats = {
    "total_predictions": 0,
    "prediction_times": []
}

def get_recognizer():
    """Get or create recognizer instance."""
    global recognizer
    if recognizer is None:
        recognizer = PokemonRecognizer()
    return recognizer

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with statistics."""
    global prediction_stats
    
    avg_time = 0
    if prediction_stats["prediction_times"]:
        avg_time = sum(prediction_stats["prediction_times"]) / len(prediction_stats["prediction_times"])
    
    return jsonify({
        "status": "healthy",
        "model_loaded": recognizer is not None,
        "total_predictions": prediction_stats["total_predictions"],
        "average_prediction_time_ms": round(avg_time * 1000, 2),
        "prediction_count": len(prediction_stats["prediction_times"])
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict Pokemon from image bytes."""
    global prediction_stats
    
    try:
        recognizer = get_recognizer()
        
        # Get image data
        image_bytes = request.get_data()
        if not image_bytes:
            return jsonify({"error": "No image data provided"}), 400
        
        # Make prediction with timing
        start_time = time.time()
        pokemon_name, confidence = recognizer.predict(image_bytes)
        end_time = time.time()
        
        # Update statistics
        prediction_time = end_time - start_time
        prediction_stats["total_predictions"] += 1
        prediction_stats["prediction_times"].append(prediction_time)
        
        # Keep only last 100 prediction times for memory
        if len(prediction_stats["prediction_times"]) > 100:
            prediction_stats["prediction_times"] = prediction_stats["prediction_times"][-100:]
        
        return jsonify({
            "pokemon": pokemon_name,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "prediction_time_ms": round(prediction_time * 1000, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/url', methods=['POST'])
def predict_url():
    """Predict Pokemon from image URL."""
    try:
        import requests
        
        recognizer = get_recognizer()
        
        # Get URL from request
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "URL not provided"}), 400
        
        url = data['url']
        
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_bytes = response.content
        
        # Make prediction with timing
        start_time = time.time()
        pokemon_name, confidence = recognizer.predict(image_bytes)
        end_time = time.time()
        
        # Update statistics
        prediction_time = end_time - start_time
        prediction_stats["total_predictions"] += 1
        prediction_stats["prediction_times"].append(prediction_time)
        
        # Keep only last 100 prediction times for memory
        if len(prediction_stats["prediction_times"]) > 100:
            prediction_stats["prediction_times"] = prediction_stats["prediction_times"][-100:]
        
        return jsonify({
            "pokemon": pokemon_name,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "prediction_time_ms": round(prediction_time * 1000, 2)
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Vercel serverless entry point
app_handler = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Pokemon Recognition API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
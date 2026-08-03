FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with new structure
COPY src/ ./src/
COPY models/ ./models/
COPY run.py .

# Verify files exist
RUN ls -lh models/pokemon_cnn_v2.onnx models/labels_v2.json models/event_embedding_index.npz models/event_embedding_meta.json models/event_labels.json models/event_label_config.json

# Expose port
EXPOSE 8080

# Run with gunicorn for production (preload to load model before forking workers)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "60", "--preload", "--access-logfile", "-", "--error-logfile", "-", "run:app"]

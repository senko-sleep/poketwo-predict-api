FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY pokemon_cnn_v2.onnx .
COPY labels_v2.json .

# Verify files exist
RUN ls -lh pokemon_cnn_v2.onnx labels_v2.json

# Expose port
EXPOSE 8080

# Run with gunicorn for production (single worker for free tier memory limits)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-", "app:app"]

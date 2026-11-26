FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Minimal system libraries needed by OpenCV/TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, model weights, and visualization assets
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY notebook/ ./notebook/
COPY start.sh ./start.sh

RUN chmod +x /app/start.sh && \
    mkdir -p /app/data/retrain_uploads /app/models/backups

EXPOSE 8501 8000

# Starts both FastAPI (8000) and Streamlit (8501) in the same container
CMD ["./start.sh"]
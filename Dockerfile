# Multi-stage build for MLOps Platform
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs

# Expose ports
EXPOSE 8000 5000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command (can be overridden)
CMD ["python", "serving/api.py"]

# ===== docker-compose.yml =====
# FILE: docker-compose.yml
# version: '3.8'
# 
# services:
#   mlflow:
#     image: python:3.10-slim
#     container_name: mlops-mlflow
#     working_dir: /app
#     volumes:
#       - ./mlruns:/app/mlruns
#       - ./mlartifacts:/app/mlartifacts
#     ports:
#       - "5000:5000"
#     command: >
#       bash -c "pip install mlflow &&
#                mlflow server 
#                --backend-store-uri sqlite:///mlflow.db
#                --default-artifact-root /app/mlartifacts
#                --host 0.0.0.0"
#     networks:
#       - mlops-network
# 
#   api:
#     build: .
#     container_name: mlops-api
#     ports:
#       - "8000:8000"
#     environment:
#       - MLFLOW_TRACKING_URI=http://mlflow:5000
#       - SERVING_HOST=0.0.0.0
#       - SERVING_PORT=8000
#     depends_on:
#       - mlflow
#     volumes:
#       - ./models:/app/models
#       - ./data:/app/data
#     networks:
#       - mlops-network
#     command: python serving/api.py
# 
#   prometheus:
#     image: prom/prometheus:latest
#     container_name: mlops-prometheus
#     ports:
#       - "9090:9090"
#     volumes:
#       - ./prometheus.yml:/etc/prometheus/prometheus.yml
#       - prometheus_data:/prometheus
#     command:
#       - '--config.file=/etc/prometheus/prometheus.yml'
#       - '--storage.tsdb.path=/prometheus'
#     networks:
#       - mlops-network
# 
#   grafana:
#     image: grafana/grafana:latest
#     container_name: mlops-grafana
#     ports:
#       - "3000:3000"
#     environment:
#       - GF_SECURITY_ADMIN_PASSWORD=admin
#     volumes:
#       - grafana_data:/var/lib/grafana
#     depends_on:
#       - prometheus
#     networks:
#       - mlops-network
# 
# networks:
#   mlops-network:
#     driver: bridge
# 
# volumes:
#   prometheus_data:
#   grafana_data:

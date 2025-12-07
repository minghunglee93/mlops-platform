# End-to-End MLOps Platform

A production-ready MLOps platform for the complete machine learning lifecycle: training, experiment tracking, model registry, serving, and monitoring.

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Training       │────▶│  Model Registry  │────▶│  Model Serving  │
│  Pipeline       │     │  (Versioning)    │     │  (FastAPI)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Experiment     │     │  MLflow Model    │     │  Prometheus     │
│  Tracking       │     │  Registry        │     │  Metrics        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## ✨ Features

### Phase 1: Training & Tracking (✅ Complete)
- **Training Pipeline**: Automated model training with MLflow integration
- **Experiment Tracking**: Track parameters, metrics, and artifacts
- **Model Registry**: Version control and lifecycle management
- **Model Serving**: REST API with FastAPI
- **Monitoring**: Prometheus metrics for predictions

### Phase 2: Feature Store (✅ Complete)
- **Feature Management**: Centralized feature definitions with Feast
- **Feature Engineering**: Automated feature transformations
- **Online Serving**: Low-latency feature retrieval for inference
- **Offline Storage**: Point-in-time correct features for training
- **Feature Monitoring**: Track feature drift and quality

### Phase 3: Drift Detection (✅ Complete)
- **Data Drift Detection**: Monitor feature distribution changes
- **Target Drift Detection**: Track target variable shifts
- **Performance Monitoring**: Detect model degradation
- **Automated Testing**: Pass/fail criteria for drift
- **Visual Reports**: Interactive HTML dashboards with Evidently
- **Alert System**: Threshold-based notifications

### Phase 4: A/B Testing (✅ Complete)
- **Champion/Challenger Testing**: Compare model versions safely
- **Traffic Splitting**: Multiple strategies (fixed, epsilon-greedy, Thompson sampling, UCB)
- **Statistical Testing**: Automated significance tests
- **Multi-Armed Bandits**: Intelligent traffic allocation
- **Automated Promotion**: Data-driven winner selection
- **Performance Tracking**: Real-time metrics and rewards

### Phase 5: Automated Retraining (✅ Complete)
- **Performance-Based**: Triggers on degradation
- **Drift-Based**: Responds to data/concept drift
- **Scheduled**: Periodic model updates
- **Manual Triggers**: On-demand retraining
- **Auto-Promotion**: Automatic deployment
- **Job Tracking**: Complete audit trail

### Coming Soon
- Kubernetes deployment
- CI/CD pipelines
- Web UI dashboard

## 📁 Project Structure

```
mlops-platform/
├── config.py                 # Configuration management
├── training/
│   └── pipeline.py          # Training pipeline with MLflow
├── registry/
│   └── model_registry.py    # Model versioning & lifecycle
├── serving/
│   └── api.py              # FastAPI serving endpoint
├── feature_store/          # NEW - Feature management
│   ├── store.py           # Feast wrapper
│   ├── features.py        # Feature definitions
│   └── engineering.py     # Feature engineering
├── monitoring/
│   └── drift_detection.py  # Data drift monitoring (Phase 3)
├── retraining/        # NEW - Automated retraining
│   └── scheduler.py
├── examples/
│   ├── train_example.py
│   ├── feature_store_example.py
│   ├── drift_detection_example.py
│   ├── ab_testing_example.py
│   └── retraining_example.py  # NEW
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## 🚀 Quick Start

> **TL;DR**: Run `python run_training.py` after installing dependencies!

### Automated Setup (Recommended)

**Linux/Mac:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

**Windows:**
```batch
quickstart.bat
```

### Manual Installation

```bash
# Clone repository
git clone <repo-url>
cd mlops-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

**Important**: Install with `pip install -e .` to make all modules properly importable!

For detailed installation troubleshooting, see [INSTALL.md](INSTALL.md).

### 1. Train Your First Model

```bash
# Activate virtual environment first!
source venv/bin/activate  # Windows: venv\Scripts\activate

# Method 1: Use the wrapper script (Easiest!)
python run_training.py

# Method 2: Install package first, then run
pip install -e .
python examples/train_example.py

# Method 3: Run as module
python -m examples.train_example
```

```bash
# Clone repository
git clone <repo-url>
cd mlops-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional)
```

### 3. Train Your First Model

```bash
# Run the example training script
python examples/train_example.py
```

This will:
- Train multiple models (Random Forest, Logistic Regression)
- Track experiments in MLflow
- Compare model performance
- Register the best model

### 4. View Experiments

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

### 5. Start Model Serving

```bash
# Start the API server
python serving/api.py

# API will be available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Make Predictions

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
                  11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]],
    "model_name": "classifier_model"
  }'

# Using Python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]],
        "model_name": "classifier_model"
    }
)
print(response.json())
```

## 📚 Detailed Usage

### Training Pipeline

```python
from training.pipeline import TrainingPipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Initialize pipeline
pipeline = TrainingPipeline(
    experiment_name="my_experiment",
    model_type="sklearn"
)

# Prepare data
X_train, X_test, y_train, y_test = pipeline.prepare_data(
    data=df,
    target_column="target"
)

# Train model
model = RandomForestClassifier()
trained_model, metrics = pipeline.train(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    hyperparameters={"n_estimators": 100},
    tags={"experiment": "baseline"}
)
```

### Model Registry

```python
from registry.model_registry import ModelRegistry

registry = ModelRegistry()

# Register model
metadata = registry.register_model(
    model_name="my_model",
    run_id="<mlflow_run_id>",
    description="Production model v1"
)

# Promote to production
registry.promote_model(
    model_name="my_model",
    version="1",
    stage="Production"
)

# Load production model
model = registry.get_production_model("my_model")
```

### Model Serving API

The serving API provides these endpoints:

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /models` - List all models
- `GET /models/{name}` - Get model info
- `POST /models/{name}/load` - Preload model
- `DELETE /models/{name}/unload` - Unload model
- `GET /metrics` - Prometheus metrics

## 🔧 Configuration

### Environment Variables

```bash
# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=default-experiment

# Model Serving
SERVING_HOST=0.0.0.0
SERVING_PORT=8000

# Training
TRAIN_BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_EPOCHS=10

# Monitoring
ENABLE_MONITORING=true
MONITORING_PORT=9090
```

### Settings in `config.py`

All configuration is centralized in `config.py` using Pydantic BaseSettings, which supports:
- Environment variables
- `.env` files
- Default values
- Type validation

## 📊 Monitoring

### Prometheus Metrics

The serving API exposes Prometheus metrics:

```
# Total predictions
model_predictions_total{model_name="my_model",version="1",status="success"}

# Prediction latency
model_prediction_latency_seconds{model_name="my_model",version="1"}
```

Access metrics at: `http://localhost:8000/metrics`

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 🐳 Docker Deployment (Coming Soon)

```bash
# Build image
docker build -t mlops-platform .

# Run container
docker run -p 8000:8000 mlops-platform
```

## 🔄 CI/CD Integration (Phase 2)

Integration with:
- GitHub Actions
- GitLab CI
- Jenkins
- Azure DevOps

## 📈 Roadmap

- [x] Training pipeline with experiment tracking
- [x] Model registry and versioning
- [x] Model serving API
- [x] Basic monitoring with Prometheus
- [x] Feature store with Feast
- [x] Feature engineering utilities
- [x] Data drift detection (Evidently)
- [x] A/B testing framework
- [x] Automated retraining
- [ ] Kubernetes deployment
- [ ] CI/CD pipelines
- [ ] Web UI dashboard

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- MLflow for experiment tracking
- FastAPI for API framework
- Prometheus for monitoring
- scikit-learn and PyTorch for ML frameworks

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review examples in `examples/`

---

Built with ❤️ for ML Engineers and Data Scientists
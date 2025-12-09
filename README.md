# End-to-End MLOps Platform

A production-ready MLOps platform for the complete machine learning lifecycle: training, experiment tracking, model registry, feature store, serving, monitoring, and automated retraining.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Ingress (NGINX + TLS)                   │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┼────────┬────────────┬──────────┐
       ▼       ▼        ▼            ▼          ▼
   ┌─────┐ ┌──────┐ ┌──────┐  ┌─────────┐ ┌──────┐
   │ API │ │MLflow│ │Prom. │  │ Grafana │ │ HPA  │
   │ Pods│ │      │ │      │  │         │ │      │
   └──┬──┘ └───┬──┘ └──────┘  └─────────┘ └──────┘
      │        │
      └────┬───┘
           ▼
      ┌─────────┐        ┌──────────────┐
      │PostgreSQL│        │ Persistent   │
      │         │◄──────►│ Volumes      │
      └─────────┘        └──────────────┘
```

## ✨ Features

### Phase 1: Training & Tracking ✅
- **Training Pipeline**: Automated model training with MLflow integration
- **Experiment Tracking**: Track parameters, metrics, and artifacts
- **Model Registry**: Version control and lifecycle management
- **Model Serving**: REST API with FastAPI
- **Monitoring**: Prometheus metrics for predictions

### Phase 2: Feature Store ✅
- **Feature Management**: Centralized feature definitions with Feast
- **Feature Engineering**: Automated feature transformations
- **Online Serving**: Low-latency feature retrieval for inference
- **Offline Storage**: Point-in-time correct features for training
- **Feature Monitoring**: Track feature drift and quality

### Phase 3: Drift Detection ✅
- **Data Drift Detection**: Monitor feature distribution changes
- **Target Drift Detection**: Track target variable shifts
- **Performance Monitoring**: Detect model degradation
- **Automated Testing**: Pass/fail criteria for drift
- **Visual Reports**: Interactive HTML dashboards with Evidently
- **Alert System**: Threshold-based notifications

### Phase 4: A/B Testing ✅
- **Champion/Challenger Testing**: Compare model versions safely
- **Traffic Splitting**: Multiple strategies (fixed, epsilon-greedy, Thompson sampling, UCB)
- **Statistical Testing**: Automated significance tests
- **Multi-Armed Bandits**: Intelligent traffic allocation
- **Automated Promotion**: Data-driven winner selection
- **Performance Tracking**: Real-time metrics and rewards

### Phase 5: Automated Retraining ✅
- **Performance-Based**: Triggers on degradation
- **Drift-Based**: Responds to data/concept drift
- **Scheduled**: Periodic model updates
- **Manual Triggers**: On-demand retraining
- **Auto-Promotion**: Automatic deployment
- **Job Tracking**: Complete audit trail

### Phase 6: Kubernetes Deployment ✅
- **Helm Charts**: Production-ready deployment
- **Auto-scaling**: HPA based on CPU/memory (3-10 replicas)
- **High Availability**: Multi-replica deployments with anti-affinity
- **Persistent Storage**: PVCs for models, data, and artifacts
- **Ingress & TLS**: External access with cert-manager
- **Monitoring**: Prometheus + Grafana integration
- **Resource Management**: Quotas, limits, and PodDisruptionBudget

### Coming Soon
- CI/CD Pipelines
- Web UI Dashboard

## 📁 Project Structure

```
mlops-platform/
├── config.py                    # Configuration management
├── Dockerfile                   # Container image
├── Makefile                     # Convenient commands
├── requirements.txt             # Python dependencies
│
├── training/
│   ├── __init__.py
│   └── pipeline.py             # Training pipeline with MLflow
│
├── registry/
│   ├── __init__.py
│   └── model_registry.py       # Model versioning & lifecycle
│
├── serving/
│   ├── __init__.py
│   └── api.py                  # FastAPI serving endpoint
│
├── feature_store/
│   ├── __init__.py
│   ├── store.py                # Feast wrapper
│   ├── features.py             # Feature definitions
│   └── engineering.py          # Feature engineering
│
├── monitoring/
│   ├── __init__.py
│   └── drift_detector.py       # Data drift monitoring
│
├── ab_testing/
│   ├── __init__.py
│   └── experiment.py           # A/B testing framework
│
├── retraining/
│   ├── __init__.py
│   └── scheduler.py            # Automated retraining
│
├── kubernetes/                  # K8s manifests
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── services.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   └── pvc.yaml
│
├── helm/                        # Helm chart
│   └── mlops-platform/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│
├── scripts/
│   ├── deploy-k8s.sh           # Kubernetes deployment
│   └── deploy-helm.sh          # Helm deployment
│
├── examples/
│   ├── train_example.py
│   ├── feature_store_example.py
│   ├── drift_detection_example.py
│   ├── ab_testing_example.py
│   └── retraining_example.py
│
└── tests/
    └── test_*.py               # Unit tests
```

## 🚀 Quick Start

### Local Development

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# Or use Makefile
make install

# Train model
make train

# Start serving
make serve

# Run examples
make run-examples
```

### Docker

```bash
# Build
make docker-build

# Run
make docker-run

# Access at http://localhost:8000
```

### Kubernetes

```bash
# Deploy with kubectl
make k8s-deploy

# Or with Helm
make helm-deploy

# Check status
make k8s-status

# View logs
make k8s-logs

# Port forward (development)
kubectl port-forward svc/mlops-api-service 8000:8000 -n mlops
```

## 📚 Documentation

- [Quick Start Guide](QUICKSTART_GUIDE.md)
- [Installation](INSTALL.md)
- [Feature Store](FEATURE_STORE.md)
- [Automated Retraining](AUTOMATED_RETRAINING.md)
- [Kubernetes Deployment](KUBERNETES_DEPLOYMENT.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## 🔧 Configuration

### Environment Variables

Create `.env` file:

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
DRIFT_DETECTION_THRESHOLD=0.05
```

### Kubernetes Configuration

Edit `helm/mlops-platform/values.yaml`:

```yaml
replicaCount: 3

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10

resources:
  limits:
    cpu: 2000m
    memory: 2Gi

ingress:
  enabled: true
  hosts:
    - host: api.mlops.example.com
```

## 💻 Usage Examples

### Training Pipeline

```python
from training.pipeline import TrainingPipeline
from sklearn.ensemble import RandomForestClassifier

# Initialize
pipeline = TrainingPipeline("my_experiment", "sklearn")

# Prepare data
X_train, X_test, y_train, y_test = pipeline.prepare_data(df, "target")

# Train
model, metrics = pipeline.train(
    model=RandomForestClassifier(),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    hyperparameters={"n_estimators": 100}
)
```

### Feature Store

```python
from feature_store import MLOpsFeatureStore

store = MLOpsFeatureStore()

# Get online features (serving)
features = store.get_online_features(
    entity_rows=[{"user_id": "user_123"}],
    features=["user_features:age", "user_features:income"]
)

# Get historical features (training)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:age", "user_features:income"]
)
```

### Automated Retraining

```python
from retraining.scheduler import RetrainingScheduler, RetrainingConfig

config = RetrainingConfig(
    model_name="my_model",
    performance_threshold=0.05,
    drift_threshold=0.1,
    schedule_enabled=True,
    auto_promote=True
)

scheduler = RetrainingScheduler(config, data_loader, pipeline)
scheduler.initialize_monitors(ref_data, baseline_metrics)
scheduler.start_monitoring()
```

### A/B Testing

```python
from ab_testing.experiment import ABExperiment, ModelVariant

variants = [
    ModelVariant("champion", "v1.0", traffic_weight=0.7),
    ModelVariant("challenger", "v2.0", traffic_weight=0.3)
]

experiment = ABExperiment("comparison", variants)

# Select variant and record results
variant = experiment.select_variant()
experiment.record_result(variant, success=True, reward=1.0)
```

## 🌐 API Endpoints

### Model Serving

```bash
# Health check
GET /health

# List models
GET /models

# Make prediction
POST /predict
{
  "features": [[1.0, 2.0, 3.0, ...]],
  "model_name": "my_model"
}

# Batch prediction
POST /predict/batch
```

### Drift Detection

```bash
# Initialize detector
POST /drift/initialize

# Detect drift
POST /drift/detect

# Get reports
GET /drift/reports
```

### A/B Testing

```bash
# Create experiment
POST /ab/experiments

# Predict with A/B test
POST /ab/predict/{experiment_name}

# Get results
GET /ab/experiments/{experiment_name}

# Statistical test
POST /ab/experiments/{experiment_name}/test
```

### Automated Retraining

```bash
# Configure retraining
POST /retraining/configure

# Trigger retraining
POST /retraining/{model_name}/trigger

# Check if needed
GET /retraining/{model_name}/check

# Job history
GET /retraining/{model_name}/jobs
```

## 📊 Monitoring

### Prometheus Metrics

```
# Total predictions
model_predictions_total{model_name, version, status}

# Prediction latency
model_prediction_latency_seconds{model_name, version}

# Drift detected
drift_detected_total{model_name}

# A/B test requests
ab_test_requests_total{experiment_name, variant}
```

Access metrics: `http://localhost:8000/metrics`

### Grafana Dashboards

- Model Performance
- Request Latency
- Error Rates
- Drift Metrics
- A/B Test Results

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes 1.24+
- kubectl configured
- Helm 3 (optional)
- 8GB RAM, 4 CPUs minimum

### Deploy

```bash
# Option 1: kubectl
bash scripts/deploy-k8s.sh

# Option 2: Helm
helm install mlops-platform ./helm/mlops-platform \
  --namespace mlops \
  --create-namespace

# Option 3: Makefile
make k8s-deploy
```

### Cloud Providers

**AWS EKS:**
```bash
eksctl create cluster --name mlops --region us-west-2
make k8s-deploy
```

**GCP GKE:**
```bash
gcloud container clusters create mlops --zone us-central1-a
make k8s-deploy
```

**Azure AKS:**
```bash
az aks create --name mlops --resource-group mlops-rg
az aks get-credentials --name mlops --resource-group mlops-rg
make k8s-deploy
```

### Access Services

```bash
# Port forwarding (development)
kubectl port-forward svc/mlops-api-service 8000:8000 -n mlops
kubectl port-forward svc/grafana-service 3000:3000 -n mlops
kubectl port-forward svc/mlflow-service 5000:5000 -n mlops

# Production (via Ingress)
# https://api.mlops.example.com
# https://mlflow.mlops.example.com
# https://grafana.mlops.example.com
```

## 🔧 Makefile Commands

```bash
make help              # Show all commands
make install          # Install dependencies
make test             # Run tests
make train            # Train example model
make serve            # Start API server
make docker-build     # Build Docker image
make docker-run       # Run in Docker
make k8s-deploy       # Deploy to Kubernetes
make helm-deploy      # Deploy with Helm
make k8s-status       # Check deployment status
make k8s-logs         # View API logs
make run-examples     # Run all examples
```

## 🧪 Testing

```bash
# Run all tests
make test

# Or manually
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

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
- [x] Kubernetes deployment with Helm
- [ ] CI/CD pipelines
- [ ] Web UI dashboard
- [ ] Service mesh integration
- [ ] Multi-region deployment

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🐛 Troubleshooting

### ModuleNotFoundError

```bash
# Install in editable mode
pip install -e .

# Or use wrapper scripts
python run_training.py
```

### Port Already in Use

```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>
```

### Kubernetes Pods Not Starting

```bash
# Check pod status
kubectl get pods -n mlops
kubectl describe pod <pod-name> -n mlops
kubectl logs <pod-name> -n mlops
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more details.

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- MLflow for experiment tracking
- FastAPI for API framework
- Feast for feature store
- Evidently for drift detection
- Prometheus for monitoring
- Kubernetes for orchestration

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review examples in `examples/`

---

**Built with ❤️ for ML Engineers and Data Scientists**
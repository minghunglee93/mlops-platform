"""
Model Serving API with FastAPI
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from registry.model_registry import ModelRegistry
from monitoring.drift_detector import DriftDetector
from ab_testing.experiment import ABExperiment, ModelVariant, TrafficSplitStrategy
from retraining.scheduler import (
    RetrainingScheduler,
    RetrainingConfig,
    RetrainingTrigger
)

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'model_predictions_total',
    'Total prediction requests',
    ['model_name', 'version', 'status']
)

REQUEST_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name', 'version']
)

DRIFT_DETECTED = Counter(
    'drift_detected_total',
    'Total drift detections',
    ['model_name']
)

AB_TEST_REQUESTS = Counter(
    'ab_test_requests_total',
    'Total A/B test requests',
    ['experiment_name', 'variant']
)

# FastAPI app
app = FastAPI(
    title="MLOps Model Serving API",
    description="Production-grade model serving with monitoring, drift detection, and A/B testing",
    version=settings.VERSION
)

# Model registry
registry = ModelRegistry()

# Loaded models cache
loaded_models: Dict[str, Any] = {}

# Drift detector (initialized with reference data)
drift_detector: Optional[DriftDetector] = None
reference_data_buffer: List[Dict] = []  # Store recent predictions for drift detection

# A/B testing experiments
active_experiments: Dict[str, ABExperiment] = {}

# Retraining Schedulers
active_schedulers: Dict[str, RetrainingScheduler] = {}

# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]] = Field(..., description="Input features for prediction")
    model_name: str = Field(..., description="Name of the model to use")
    version: Optional[str] = Field(None, description="Specific model version (optional)")

    class Config:
        schema_extra = {
            "example": {
                "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "model_name": "my_model",
                "version": "1"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Any]
    model_name: str
    model_version: str
    timestamp: str
    latency_ms: float


class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    versions: List[Dict]
    description: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    loaded_models: int


class RetrainingConfigRequest(BaseModel):
    """Request model for retraining configuration."""
    model_name: str
    performance_threshold: float = 0.05
    drift_threshold: float = 0.1
    schedule_enabled: bool = False
    schedule_interval_days: int = 7
    auto_promote: bool = False


class RetrainingTriggerRequest(BaseModel):
    """Request model for triggering retraining."""
    trigger_type: str = Field(..., description="manual, performance, drift, or scheduled")
    reason: str = Field(..., description="Reason for retraining")


# Helper Functions
def load_model(model_name: str, version: Optional[str] = None) -> Any:
    """
    Load model from registry with caching.

    Args:
        model_name: Name of the model
        version: Specific version or None for latest

    Returns:
        Loaded model
    """
    cache_key = f"{model_name}_{version or 'latest'}"

    if cache_key in loaded_models:
        logger.info(f"Using cached model: {cache_key}")
        return loaded_models[cache_key]

    try:
        logger.info(f"Loading model: {cache_key}")
        if version:
            model = registry.get_model(model_name, version=version)
        else:
            model = registry.get_production_model(model_name)
            if model is None:
                model = registry.get_model(model_name)

        loaded_models[cache_key] = model
        return model

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")


# API Endpoints
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint."""
    return {
        "message": "MLOps Model Serving API",
        "version": settings.VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        loaded_models=len(loaded_models)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Make predictions using a registered model.

    Args:
        request: Prediction request with features and model info
        background_tasks: FastAPI background tasks

    Returns:
        Predictions with metadata
    """
    start_time = time.time()

    try:
        # Load model
        model = load_model(request.model_name, request.version)

        # Convert features to numpy array
        features = np.array(request.features)

        # Make predictions
        predictions = model.predict(features)

        # Convert to list for JSON serialization
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        # Calculate latency
        latency = (time.time() - start_time) * 1000  # Convert to ms

        # Get model version
        version = request.version or "latest"

        # Store prediction for drift detection (in background)
        if settings.ENABLE_MONITORING:
            background_tasks.add_task(
                store_prediction_for_drift_detection,
                features=request.features,
                predictions=predictions
            )

        # Update metrics
        REQUEST_COUNT.labels(
            model_name=request.model_name,
            version=version,
            status="success"
        ).inc()

        REQUEST_LATENCY.labels(
            model_name=request.model_name,
            version=version
        ).observe(latency / 1000)

        logger.info(f"Prediction successful: {request.model_name} v{version}, latency: {latency:.2f}ms")

        return PredictionResponse(
            predictions=predictions,
            model_name=request.model_name,
            model_version=version,
            timestamp=datetime.now().isoformat(),
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        REQUEST_COUNT.labels(
            model_name=request.model_name,
            version=request.version or "latest",
            status="error"
        ).inc()

        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all registered models.

    Returns:
        List of models with versions
    """
    try:
        models = registry.list_models()
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Model information
    """
    try:
        versions = registry.get_model_versions(model_name)
        if not versions:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        return ModelInfo(
            name=model_name,
            versions=versions,
            description=None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/load")
async def load_model_endpoint(
    model_name: str,
    version: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Preload a model into cache.

    Args:
        model_name: Name of the model
        version: Specific version (optional)

    Returns:
        Success message
    """
    try:
        load_model(model_name, version)
        return {
            "message": f"Model {model_name} loaded successfully",
            "version": version or "latest"
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}/unload")
async def unload_model(model_name: str, version: Optional[str] = None):
    """
    Unload a model from cache.

    Args:
        model_name: Name of the model
        version: Specific version (optional)

    Returns:
        Success message
    """
    cache_key = f"{model_name}_{version or 'latest'}"

    if cache_key in loaded_models:
        del loaded_models[cache_key]
        logger.info(f"Unloaded model: {cache_key}")
        return {"message": f"Model {model_name} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model not loaded: {model_name}")


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type="text/plain")


# Drift Detection Endpoints

def store_prediction_for_drift_detection(features: List[List[float]], predictions: List[Any]):
    """Store prediction data for drift detection (background task)."""
    global reference_data_buffer

    for feature_row, pred in zip(features, predictions):
        reference_data_buffer.append({
            'timestamp': datetime.now().isoformat(),
            'features': feature_row,
            'prediction': pred
        })

    # Keep only recent N samples
    max_buffer_size = 1000
    if len(reference_data_buffer) > max_buffer_size:
        reference_data_buffer = reference_data_buffer[-max_buffer_size:]


@app.post("/drift/initialize")
async def initialize_drift_detection(reference_file: Optional[str] = None):
    """
    Initialize drift detector with reference data.

    Args:
        reference_file: Path to reference data CSV (optional)

    Returns:
        Status message
    """
    global drift_detector

    try:
        if reference_file:
            # Load from file
            reference_df = pd.read_csv(reference_file)
        elif reference_data_buffer:
            # Use buffered predictions
            reference_df = pd.DataFrame([
                {'feature_' + str(i): val for i, val in enumerate(item['features'])}
                for item in reference_data_buffer[-500:]  # Use last 500
            ])
        else:
            raise HTTPException(
                status_code=400,
                detail="No reference data available. Make some predictions first or provide reference_file"
            )

        drift_detector = DriftDetector(
            reference_data=reference_df,
            drift_threshold=settings.DRIFT_DETECTION_THRESHOLD
        )

        logger.info(f"Drift detector initialized with {len(reference_df)} reference samples")

        return {
            "status": "success",
            "message": f"Drift detector initialized with {len(reference_df)} samples"
        }

    except Exception as e:
        logger.error(f"Error initializing drift detector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift/detect")
async def detect_drift():
    """
    Detect drift in recent production data.

    Returns:
        Drift detection results
    """
    global drift_detector, reference_data_buffer

    if drift_detector is None:
        raise HTTPException(
            status_code=400,
            detail="Drift detector not initialized. Call /drift/initialize first"
        )

    if len(reference_data_buffer) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough production data. Have {len(reference_data_buffer)}, need at least 50"
        )

    try:
        # Create current data from recent predictions
        current_df = pd.DataFrame([
            {'feature_' + str(i): val for i, val in enumerate(item['features'])}
            for item in reference_data_buffer[-200:]  # Use last 200
        ])

        # Detect drift
        drift_summary = drift_detector.detect_data_drift(current_df)

        # Check if alert needed
        if drift_detector.should_alert(drift_summary):
            DRIFT_DETECTED.labels(model_name="current_model").inc()
            logger.warning("🚨 DRIFT ALERT TRIGGERED")

        return {
            "status": "success",
            "drift_detected": drift_summary['dataset_drift'],
            "drift_share": drift_summary['drift_share'],
            "drifted_columns": drift_summary['number_of_drifted_columns'],
            "report_path": drift_summary['report_path'],
            "alert": drift_detector.should_alert(drift_summary),
            "timestamp": drift_summary['timestamp']
        }

    except Exception as e:
        logger.error(f"Error detecting drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/report/{report_name}")
async def get_drift_report(report_name: str):
    """
    Retrieve a drift detection report.

    Args:
        report_name: Name of the report file

    Returns:
        HTML report
    """
    report_path = Path("drift_reports") / report_name

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename=report_name
    )


@app.get("/drift/reports")
async def list_drift_reports():
    """
    List all available drift reports.

    Returns:
        List of report files
    """
    reports_dir = Path("drift_reports")

    if not reports_dir.exists():
        return {"reports": []}

    reports = [
        {
            "name": report.name,
            "created": datetime.fromtimestamp(report.stat().st_mtime).isoformat(),
            "size_kb": report.stat().st_size / 1024
        }
        for report in sorted(reports_dir.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True)
    ]

    return {"reports": reports, "total": len(reports)}


@app.get("/drift/status")
async def drift_detection_status():
    """
    Get drift detection system status.

    Returns:
        Status information
    """
    return {
        "initialized": drift_detector is not None,
        "buffer_size": len(reference_data_buffer),
        "reference_samples": len(drift_detector.reference_data) if drift_detector else 0,
        "drift_threshold": settings.DRIFT_DETECTION_THRESHOLD if drift_detector else None,
        "monitoring_enabled": settings.ENABLE_MONITORING
    }


# A/B Testing Endpoints

@app.post("/ab/experiments")
async def create_ab_experiment(
    experiment_name: str,
    variant_names: List[str],
    variant_versions: List[str],
    strategy: str = "fixed",
    traffic_weights: Optional[List[float]] = None
):
    """
    Create a new A/B testing experiment.

    Args:
        experiment_name: Name of the experiment
        variant_names: Names of variants (e.g., ["champion", "challenger"])
        variant_versions: Model versions for each variant
        strategy: Traffic split strategy (fixed, epsilon_greedy, thompson_sampling, ucb)
        traffic_weights: Traffic weights for each variant (for fixed strategy)

    Returns:
        Experiment details
    """
    global active_experiments

    if experiment_name in active_experiments:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment '{experiment_name}' already exists"
        )

    if len(variant_names) != len(variant_versions):
        raise HTTPException(
            status_code=400,
            detail="Number of variant names must match number of versions"
        )

    # Create variants
    variants = []
    if traffic_weights is None:
        traffic_weights = [1.0 / len(variant_names)] * len(variant_names)

    for name, version, weight in zip(variant_names, variant_versions, traffic_weights):
        variants.append(ModelVariant(
            name=name,
            model_version=version,
            traffic_weight=weight
        ))

    # Create experiment
    try:
        strategy_enum = TrafficSplitStrategy(strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {strategy}. Must be one of: fixed, epsilon_greedy, thompson_sampling, ucb"
        )

    experiment = ABExperiment(
        experiment_name=experiment_name,
        variants=variants,
        strategy=strategy_enum
    )

    active_experiments[experiment_name] = experiment

    logger.info(f"Created A/B experiment: {experiment_name}")

    return {
        "status": "success",
        "experiment_name": experiment_name,
        "variants": [v.to_dict() for v in variants],
        "strategy": strategy
    }


@app.post("/ab/predict/{experiment_name}")
async def ab_predict(
    experiment_name: str,
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Make prediction using A/B test variant selection.

    Args:
        experiment_name: Name of the experiment
        request: Prediction request
        background_tasks: FastAPI background tasks

    Returns:
        Prediction with variant information
    """
    if experiment_name not in active_experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    experiment = active_experiments[experiment_name]

    start_time = time.time()

    try:
        # Select variant
        variant_name = experiment.select_variant()
        variant = experiment.variants[variant_name]

        # Load model for this variant
        model = load_model(request.model_name, variant.model_version)

        # Convert features
        features = np.array(request.features)

        # Make predictions
        predictions = model.predict(features)

        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        latency = (time.time() - start_time) * 1000

        # Record in experiment (assume success for now)
        # In production, you'd record actual success based on user feedback
        experiment.record_result(
            variant_name=variant_name,
            success=True,
            reward=1.0
        )

        # Update metrics
        AB_TEST_REQUESTS.labels(
            experiment_name=experiment_name,
            variant=variant_name
        ).inc()

        REQUEST_COUNT.labels(
            model_name=request.model_name,
            version=variant.model_version,
            status="success"
        ).inc()

        logger.info(f"A/B prediction: experiment={experiment_name}, variant={variant_name}")

        return {
            "predictions": predictions,
            "model_name": request.model_name,
            "model_version": variant.model_version,
            "experiment_name": experiment_name,
            "variant": variant_name,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round(latency, 2)
        }

    except Exception as e:
        logger.error(f"A/B prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ab/experiments")
async def list_ab_experiments():
    """
    List all active A/B experiments.

    Returns:
        List of experiments
    """
    experiments = []
    for name, exp in active_experiments.items():
        results = exp.get_results()
        experiments.append(results)

    return {"experiments": experiments, "total": len(experiments)}


@app.get("/ab/experiments/{experiment_name}")
async def get_ab_experiment_results(experiment_name: str):
    """
    Get results for a specific experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Experiment results
    """
    if experiment_name not in active_experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    experiment = active_experiments[experiment_name]
    return experiment.get_results()


@app.post("/ab/experiments/{experiment_name}/test")
async def run_ab_statistical_test(
    experiment_name: str,
    variant_a: str,
    variant_b: str,
    metric: str = "success_rate"
):
    """
    Run statistical significance test between variants.

    Args:
        experiment_name: Name of the experiment
        variant_a: First variant name
        variant_b: Second variant name
        metric: Metric to compare

    Returns:
        Statistical test results
    """
    if experiment_name not in active_experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    experiment = active_experiments[experiment_name]

    try:
        test_result = experiment.run_statistical_test(variant_a, variant_b, metric)
        return test_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ab/experiments/{experiment_name}/promote")
async def promote_challenger(
    experiment_name: str,
    champion: str,
    challenger: str,
    min_improvement: float = 0.01
):
    """
    Check if challenger should be promoted to champion.

    Args:
        experiment_name: Name of the experiment
        champion: Current champion variant
        challenger: Challenger variant
        min_improvement: Minimum improvement required

    Returns:
        Promotion recommendation
    """
    if experiment_name not in active_experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    experiment = active_experiments[experiment_name]

    should_promote, reason = experiment.should_promote_challenger(
        champion, challenger, min_improvement
    )

    return {
        "should_promote": should_promote,
        "reason": reason,
        "champion": champion,
        "challenger": challenger,
        "experiment_name": experiment_name
    }


@app.delete("/ab/experiments/{experiment_name}")
async def stop_ab_experiment(experiment_name: str, save_results: bool = True):
    """
    Stop an A/B experiment.

    Args:
        experiment_name: Name of the experiment
        save_results: Whether to save results before stopping

    Returns:
        Status message
    """
    if experiment_name not in active_experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    experiment = active_experiments[experiment_name]

    results_file = None
    report_file = None

    if save_results:
        results_file = experiment.save_results()
        report_file = experiment.generate_report()

    del active_experiments[experiment_name]

    logger.info(f"Stopped A/B experiment: {experiment_name}")

    return {
        "status": "success",
        "message": f"Experiment '{experiment_name}' stopped",
        "results_saved": save_results,
        "results_file": results_file,
        "report_file": report_file
    }


@app.post("/ab/experiments/{experiment_name}/record_feedback")
async def record_ab_feedback(
    experiment_name: str,
    variant_name: str,
    success: bool = True,
    reward: float = 1.0
):
    """
    Record user feedback for a prediction.

    Args:
        experiment_name: Name of the experiment
        variant_name: Variant that made the prediction
        success: Whether prediction was successful
        reward: Reward value

    Returns:
        Status message
    """
    if experiment_name not in active_experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    experiment = active_experiments[experiment_name]
    experiment.record_result(variant_name, success, reward)

    return {
        "status": "success",
        "message": "Feedback recorded",
        "variant": variant_name
    }


# Batch prediction endpoint
class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features: List[List[float]]
    model_name: str
    version: Optional[str] = None
    batch_size: int = Field(default=32, description="Batch size for processing")


@app.post("/predict/batch", response_model=PredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions with optimized processing.

    Args:
        request: Batch prediction request

    Returns:
        Batch predictions with metadata
    """
    start_time = time.time()

    try:
        model = load_model(request.model_name, request.version)
        features = np.array(request.features)

        # Process in batches
        predictions = []
        for i in range(0, len(features), request.batch_size):
            batch = features[i:i + request.batch_size]
            batch_preds = model.predict(batch)
            predictions.extend(batch_preds.tolist() if isinstance(batch_preds, np.ndarray) else batch_preds)

        latency = (time.time() - start_time) * 1000
        version = request.version or "latest"

        REQUEST_COUNT.labels(
            model_name=request.model_name,
            version=version,
            status="success"
        ).inc()

        logger.info(f"Batch prediction: {len(features)} samples in {latency:.2f}ms")

        return PredictionResponse(
            predictions=predictions,
            model_name=request.model_name,
            model_version=version,
            timestamp=datetime.now().isoformat(),
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retraining/configure")
async def configure_retraining(config_req: RetrainingConfigRequest):
    """
    Configure automated retraining for a model.

    Args:
        config_req: Retraining configuration

    Returns:
        Configuration status
    """
    global active_schedulers

    try:
        # Create config
        config = RetrainingConfig(
            model_name=config_req.model_name,
            performance_threshold=config_req.performance_threshold,
            drift_threshold=config_req.drift_threshold,
            schedule_enabled=config_req.schedule_enabled,
            schedule_interval_days=config_req.schedule_interval_days,
            auto_promote=config_req.auto_promote
        )

        # Data loader (in production, load from your data source)
        def data_loader():
            # Placeholder - replace with actual data loading
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=2000, n_features=20)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
            df['target'] = y
            return df

        # Create training pipeline
        pipeline = TrainingPipeline(
            experiment_name=f"retraining_{config_req.model_name}",
            model_type="sklearn"
        )

        # Initialize scheduler
        scheduler = RetrainingScheduler(
            config=config,
            data_loader=data_loader,
            training_pipeline=pipeline
        )

        # Store scheduler
        active_schedulers[config_req.model_name] = scheduler

        # Setup scheduled retraining if enabled
        if config.schedule_enabled:
            scheduler.schedule_periodic_retraining()

        logger.info(f"Retraining configured for model: {config_req.model_name}")

        return {
            "status": "success",
            "message": f"Retraining configured for {config_req.model_name}",
            "config": config.to_dict()
        }

    except Exception as e:
        logger.error(f"Error configuring retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retraining/{model_name}/initialize")
async def initialize_retraining_monitors(
        model_name: str,
        reference_data_path: Optional[str] = None
):
    """
    Initialize drift detector and performance monitor for a model.

    Args:
        model_name: Name of the model
        reference_data_path: Path to reference data CSV (optional)

    Returns:
        Initialization status
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]

    try:
        # Load reference data
        if reference_data_path:
            ref_data = pd.read_csv(reference_data_path)
        else:
            # Use recent production data from buffer
            if not reference_data_buffer:
                raise HTTPException(
                    status_code=400,
                    detail="No reference data available. Provide reference_data_path or make some predictions first"
                )
            ref_data = pd.DataFrame([
                {'feature_' + str(i): val for i, val in enumerate(item['features'])}
                for item in reference_data_buffer[-1000:]
            ])

        # Get baseline metrics from model registry
        model_versions = registry.get_model_versions(model_name)
        if not model_versions:
            raise HTTPException(
                status_code=404,
                detail=f"No versions found for model: {model_name}"
            )

        # Use metrics from latest version
        baseline_metrics = {
            'accuracy': 0.85,  # Placeholder - should load from registry
            'f1_score': 0.83
        }

        # Initialize monitors
        scheduler.initialize_monitors(ref_data, baseline_metrics)

        logger.info(f"Monitors initialized for model: {model_name}")

        return {
            "status": "success",
            "message": f"Monitors initialized for {model_name}",
            "reference_samples": len(ref_data),
            "baseline_metrics": baseline_metrics
        }

    except Exception as e:
        logger.error(f"Error initializing monitors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retraining/{model_name}/trigger")
async def trigger_retraining(model_name: str, request: RetrainingTriggerRequest):
    """
    Manually trigger retraining for a model.

    Args:
        model_name: Name of the model
        request: Trigger request with type and reason

    Returns:
        Retraining job details
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]

    try:
        # Map trigger type to enum
        trigger_map = {
            'manual': RetrainingTrigger.MANUAL,
            'performance': RetrainingTrigger.PERFORMANCE_DEGRADATION,
            'drift': RetrainingTrigger.DATA_DRIFT,
            'scheduled': RetrainingTrigger.SCHEDULED
        }

        trigger = trigger_map.get(request.trigger_type.lower())
        if not trigger:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid trigger type: {request.trigger_type}"
            )

        # Trigger retraining
        job = scheduler.trigger_retraining(
            trigger=trigger,
            reason=request.reason
        )

        logger.info(f"Retraining triggered for {model_name}: {job.job_id}")

        return {
            "status": "success",
            "job": job.to_dict()
        }

    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retraining/{model_name}/check")
async def check_retraining_needed(model_name: str):
    """
    Check if retraining is needed based on current data.

    Args:
        model_name: Name of the model

    Returns:
        Retraining recommendation
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]

    try:
        # Get recent production data
        if not reference_data_buffer:
            return {
                "needs_retraining": False,
                "reason": "Insufficient production data for analysis"
            }

        current_data = pd.DataFrame([
            {'feature_' + str(i): val for i, val in enumerate(item['features'])}
            for item in reference_data_buffer[-200:]
        ])

        # Check retraining needed
        needs_retraining, trigger, reason = scheduler.check_retraining_needed(current_data)

        return {
            "needs_retraining": needs_retraining,
            "trigger": trigger.value if trigger else None,
            "reason": reason,
            "samples_analyzed": len(current_data)
        }

    except Exception as e:
        logger.error(f"Error checking retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retraining/{model_name}/jobs")
async def get_retraining_jobs(model_name: str, limit: int = 10):
    """
    Get retraining job history for a model.

    Args:
        model_name: Name of the model
        limit: Number of recent jobs to return

    Returns:
        List of retraining jobs
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]
    job_history = scheduler.get_job_history(limit=limit)

    return {
        "model_name": model_name,
        "total_jobs": len(job_history),
        "jobs": job_history
    }


@app.get("/retraining/{model_name}/jobs/{job_id}")
async def get_retraining_job_status(model_name: str, job_id: str):
    """
    Get status of a specific retraining job.

    Args:
        model_name: Name of the model
        job_id: Job identifier

    Returns:
        Job status details
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]
    job_status = scheduler.get_job_status(job_id)

    if not job_status:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )

    return job_status


@app.get("/retraining/{model_name}/config")
async def get_retraining_config(model_name: str):
    """
    Get retraining configuration for a model.

    Args:
        model_name: Name of the model

    Returns:
        Retraining configuration
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]

    return {
        "model_name": model_name,
        "config": scheduler.config.to_dict(),
        "monitors_initialized": scheduler.drift_detector is not None
    }


@app.post("/retraining/{model_name}/start-monitoring")
async def start_monitoring(model_name: str, check_interval_seconds: int = 3600):
    """
    Start continuous monitoring for a model.

    Args:
        model_name: Name of the model
        check_interval_seconds: Monitoring interval in seconds

    Returns:
        Status message
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]

    try:
        scheduler.start_monitoring(check_interval_seconds=check_interval_seconds)

        return {
            "status": "success",
            "message": f"Monitoring started for {model_name}",
            "check_interval_seconds": check_interval_seconds
        }

    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retraining/{model_name}/stop-monitoring")
async def stop_monitoring(model_name: str):
    """
    Stop continuous monitoring for a model.

    Args:
        model_name: Name of the model

    Returns:
        Status message
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]

    try:
        scheduler.stop_monitoring()

        return {
            "status": "success",
            "message": f"Monitoring stopped for {model_name}"
        }

    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retraining/{model_name}/report")
async def generate_retraining_report(model_name: str):
    """
    Generate retraining summary report.

    Args:
        model_name: Name of the model

    Returns:
        Report file path
    """
    if model_name not in active_schedulers:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduler configured for model: {model_name}"
        )

    scheduler = active_schedulers[model_name]

    try:
        report_path = scheduler.generate_report()

        return {
            "status": "success",
            "report_path": report_path,
            "message": "Report generated successfully"
        }

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retraining/status")
async def retraining_system_status():
    """
    Get overall retraining system status.

    Returns:
        System status summary
    """
    schedulers_status = []

    for model_name, scheduler in active_schedulers.items():
        job_history = scheduler.get_job_history(limit=100)

        schedulers_status.append({
            "model_name": model_name,
            "monitoring_active": scheduler.is_monitoring,
            "total_jobs": len(job_history),
            "successful_jobs": sum(1 for j in job_history if j['status'] == 'completed'),
            "failed_jobs": sum(1 for j in job_history if j['status'] == 'failed'),
            "schedule_enabled": scheduler.config.schedule_enabled,
            "auto_promote": scheduler.config.auto_promote
        })

    return {
        "total_models": len(active_schedulers),
        "schedulers": schedulers_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.SERVING_HOST,
        port=settings.SERVING_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
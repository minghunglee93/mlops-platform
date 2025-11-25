"""
Model Serving API with FastAPI
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from datetime import datetime
import mlflow
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

from config import settings
from registry.model_registry import ModelRegistry

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

# FastAPI app
app = FastAPI(
    title="MLOps Model Serving API",
    description="Production-grade model serving with monitoring",
    version=settings.VERSION
)

# Model registry
registry = ModelRegistry()

# Loaded models cache
loaded_models: Dict[str, Any] = {}


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
async def predict(request: PredictionRequest):
    """
    Make predictions using a registered model.
    
    Args:
        request: Prediction request with features and model info
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.SERVING_HOST,
        port=settings.SERVING_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

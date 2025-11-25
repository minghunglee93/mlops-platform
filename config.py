"""
Configuration management for MLOps Platform
"""
import os
from pathlib import Path
from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Project
    PROJECT_NAME: str = "mlops-platform"
    VERSION: str = "1.0.0"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = Field(
        default="sqlite:///mlflow.db",
        env="MLFLOW_TRACKING_URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = "default-experiment"
    MLFLOW_ARTIFACT_ROOT: str = "./mlartifacts"
    
    # Model Registry
    MODEL_REGISTRY_URI: str = Field(
        default="sqlite:///model_registry.db",
        env="MODEL_REGISTRY_URI"
    )
    
    # Training
    TRAIN_BATCH_SIZE: int = 32
    EVAL_BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 10
    EARLY_STOPPING_PATIENCE: int = 3
    
    # Model Serving
    SERVING_HOST: str = "0.0.0.0"
    SERVING_PORT: int = 8000
    MODEL_NAME: str = "default_model"
    MODEL_VERSION: str = "latest"
    
    # Monitoring
    ENABLE_MONITORING: bool = True
    MONITORING_PORT: int = 9090
    DRIFT_DETECTION_THRESHOLD: float = 0.05
    
    # Data Processing
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Cloud (Optional)
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    AWS_BUCKET: str = Field(default="", env="AWS_BUCKET")
    GCP_PROJECT: str = Field(default="", env="GCP_PROJECT")
    GCP_BUCKET: str = Field(default="", env="GCP_BUCKET")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / "raw").mkdir(exist_ok=True)
            (dir_path / "processed").mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
settings.create_directories()


class ExperimentConfig:
    """Configuration for ML experiments."""
    
    def __init__(
        self,
        experiment_name: str,
        model_type: str,
        hyperparameters: Dict[str, Any]
    ):
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.hyperparameters = hyperparameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            experiment_name=config_dict["experiment_name"],
            model_type=config_dict["model_type"],
            hyperparameters=config_dict["hyperparameters"]
        )

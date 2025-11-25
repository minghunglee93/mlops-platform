"""
Model Registry for versioning and lifecycle management
"""
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    name: str
    version: str
    stage: str
    created_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    tags: Dict[str, str]
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetadata":
        return cls(**data)


class ModelRegistry:
    """
    Manages model lifecycle: registration, versioning, promotion, archival.
    """
    
    def __init__(self):
        """Initialize model registry."""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.registry_path = settings.MODEL_DIR / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load local registry metadata."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.local_registry = json.load(f)
        else:
            self.local_registry = {}
    
    def _save_registry(self):
        """Save local registry metadata."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.local_registry, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        run_id: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ModelMetadata:
        """
        Register a new model version.
        
        Args:
            model_name: Name for the model
            run_id: MLflow run ID containing the model
            description: Model description
            tags: Additional tags
            
        Returns:
            ModelMetadata object
        """
        try:
            # Register in MLflow
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Get run info
            run = self.client.get_run(run_id)
            
            # Create metadata
            metadata = ModelMetadata(
                name=model_name,
                version=str(model_version.version),
                stage="None",
                created_at=datetime.now().isoformat(),
                metrics=run.data.metrics,
                hyperparameters=run.data.params,
                tags=tags or {},
                description=description
            )
            
            # Update model version description
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            # Add tags
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=value
                    )
            
            # Save to local registry
            registry_key = f"{model_name}_v{model_version.version}"
            self.local_registry[registry_key] = metadata.to_dict()
            self._save_registry()
            
            logger.info(f"Registered model: {model_name} v{model_version.version}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str
    ) -> bool:
        """
        Promote model to a specific stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            
        Returns:
            True if successful
        """
        valid_stages = ["Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of {valid_stages}")
        
        try:
            # Transition in MLflow
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True if stage == "Production" else False
            )
            
            # Update local registry
            registry_key = f"{model_name}_v{version}"
            if registry_key in self.local_registry:
                self.local_registry[registry_key]["stage"] = stage
                self._save_registry()
            
            logger.info(f"Promoted {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Any:
        """
        Load a model from registry.
        
        Args:
            model_name: Name of the model
            version: Specific version (optional)
            stage: Stage to load from (optional)
            
        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def list_models(self) -> List[Dict]:
        """
        List all registered models.
        
        Returns:
            List of model information
        """
        try:
            models = self.client.search_registered_models()
            model_list = []
            
            for model in models:
                latest_versions = model.latest_versions
                model_info = {
                    "name": model.name,
                    "description": model.description,
                    "versions": []
                }
                
                for version in latest_versions:
                    model_info["versions"].append({
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "creation_timestamp": version.creation_timestamp
                    })
                
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_model_versions(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all versions of a model.
        
        Args:
            model_name: Name of the model
            stage: Filter by stage (optional)
            
        Returns:
            List of model versions
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                model = self.client.get_registered_model(model_name)
                versions = model.latest_versions
            
            version_list = []
            for version in versions:
                version_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                    "description": version.description,
                    "run_id": version.run_id
                }
                version_list.append(version_info)
            
            return version_list
            
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_name: Name of the model
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        try:
            # Get metadata from local registry
            key1 = f"{model_name}_v{version1}"
            key2 = f"{model_name}_v{version2}"
            
            if key1 not in self.local_registry or key2 not in self.local_registry:
                logger.warning("One or both versions not found in local registry")
                return {}
            
            metadata1 = self.local_registry[key1]
            metadata2 = self.local_registry[key2]
            
            comparison = {
                "version_1": {
                    "version": version1,
                    "metrics": metadata1.get("metrics", {}),
                    "stage": metadata1.get("stage", "Unknown")
                },
                "version_2": {
                    "version": version2,
                    "metrics": metadata2.get("metrics", {}),
                    "stage": metadata2.get("stage", "Unknown")
                },
                "metric_differences": {}
            }
            
            # Calculate differences
            metrics1 = metadata1.get("metrics", {})
            metrics2 = metadata2.get("metrics", {})
            
            for metric in set(metrics1.keys()) | set(metrics2.keys()):
                val1 = metrics1.get(metric, 0)
                val2 = metrics2.get(metric, 0)
                comparison["metric_differences"][metric] = val2 - val1
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {}
    
    def delete_model_version(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_model_version(model_name, version)
            
            # Remove from local registry
            registry_key = f"{model_name}_v{version}"
            if registry_key in self.local_registry:
                del self.local_registry[registry_key]
                self._save_registry()
            
            logger.info(f"Deleted {model_name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
            return False
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """
        Get the production version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Production model or None
        """
        try:
            return self.get_model(model_name, stage="Production")
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None

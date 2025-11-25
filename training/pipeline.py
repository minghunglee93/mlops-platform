"""
Training Pipeline with MLflow Integration
"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from pathlib import Path
import logging
import json
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the training process with experiment tracking."""

    def __init__(
        self,
        experiment_name: str,
        model_type: str = "sklearn"
    ):
        """
        Initialize training pipeline.

        Args:
            experiment_name: Name for MLflow experiment
            model_type: Type of model ('sklearn' or 'pytorch')
        """
        self.experiment_name = experiment_name
        self.model_type = model_type

        # Setup MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        logger.info(f"Initialized training pipeline for {experiment_name}")

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            data: Input dataframe
            target_column: Name of target column
            test_size: Test split ratio

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or settings.TEST_SIZE

        X = data.drop(columns=[target_column]).values
        y = data[target_column].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=settings.RANDOM_STATE
        )

        logger.info(f"Data prepared: Train={len(X_train)}, Test={len(X_test)}")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hyperparameters: Dict[str, Any] = None,
        tags: Dict[str, str] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train model with MLflow tracking.

        Args:
            model: Model instance to train
            X_train, y_train: Training data
            X_test, y_test: Test data
            hyperparameters: Model hyperparameters
            tags: Additional tags for experiment

        Returns:
            Tuple of (trained_model, metrics)
        """
        hyperparameters = hyperparameters or {}
        tags = tags or {}

        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(hyperparameters)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            # Log tags
            for key, value in tags.items():
                mlflow.set_tag(key, value)

            # Train model
            logger.info("Starting model training...")
            start_time = datetime.now()

            if self.model_type == "sklearn":
                model.fit(X_train, y_train)
            elif self.model_type == "pytorch":
                model = self._train_pytorch(model, X_train, y_train, hyperparameters)

            training_time = (datetime.now() - start_time).total_seconds()
            mlflow.log_metric("training_time_seconds", training_time)

            # Evaluate
            logger.info("Evaluating model...")
            metrics = self.evaluate(model, X_test, y_test)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            if self.model_type == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif self.model_type == "pytorch":
                mlflow.pytorch.log_model(model, "model")

            # Save model locally
            model_path = settings.MODEL_DIR / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            if self.model_type == "sklearn":
                import joblib
                joblib.dump(model, model_path)
            elif self.model_type == "pytorch":
                torch.save(model.state_dict(), model_path)

            mlflow.log_artifact(str(model_path))

            logger.info(f"Training complete. Metrics: {metrics}")

            return model, metrics

    def _train_pytorch(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Dict[str, Any]
    ) -> nn.Module:
        """Train PyTorch model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.LongTensor(y_train).to(device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparameters.get("learning_rate", settings.LEARNING_RATE)
        )

        epochs = hyperparameters.get("epochs", settings.NUM_EPOCHS)
        batch_size = hyperparameters.get("batch_size", settings.TRAIN_BATCH_SIZE)

        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(X_tensor) / batch_size)
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return model

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test, y_test: Test data

        Returns:
            Dictionary of metrics
        """
        if self.model_type == "sklearn":
            y_pred = model.predict(X_test)
        elif self.model_type == "pytorch":
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                outputs = model(X_tensor)
                _, y_pred = torch.max(outputs, 1)
                y_pred = y_pred.numpy()

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        return metrics

    def register_model(
        self,
        model_name: str,
        run_id: str,
        stage: str = "None"
    ):
        """
        Register model in MLflow Model Registry.

        Args:
            model_name: Name for registered model
            run_id: MLflow run ID
            stage: Model stage (None, Staging, Production, Archived)
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)

            # Transition to stage
            if stage != "None":
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=stage
                )

            logger.info(f"Model registered: {model_name} v{model_version.version} (stage: {stage})")
            return model_version
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise


class ExperimentTracker:
    """Track and compare experiments."""

    def __init__(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.client = mlflow.tracking.MlflowClient()

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "f1_score"
    ) -> Optional[mlflow.entities.Run]:
        """
        Get best run from experiment based on metric.

        Args:
            experiment_name: Name of experiment
            metric: Metric to optimize

        Returns:
            Best run or None
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment {experiment_name} not found")
            return None

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )

        return runs[0] if runs else None

    def compare_runs(
        self,
        experiment_name: str,
        metric: str = "f1_score",
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Compare top runs from experiment.

        Args:
            experiment_name: Name of experiment
            metric: Metric to compare
            top_n: Number of top runs to return

        Returns:
            DataFrame with run comparisons
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            return pd.DataFrame()

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n
        )

        data = []
        for run in runs:
            data.append({
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
                **run.data.params,
                **run.data.metrics
            })

        return pd.DataFrame(data)
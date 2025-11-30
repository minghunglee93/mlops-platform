"""
Example: Training a model with the MLOps platform
"""
import sys
sys.path.append('..')

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

from training.pipeline import TrainingPipeline, ExperimentTracker
from registry.model_registry import ModelRegistry

def create_sample_data():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df['target'] = y
    
    return df


def train_random_forest():
    """Train a Random Forest model."""
    print("=" * 50)
    print("Training Random Forest Model")
    print("=" * 50)
    
    # Create data
    data = create_sample_data()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        experiment_name="classification_experiment",
        model_type="sklearn"
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        data=data,
        target_column="target"
    )
    
    # Define model and hyperparameters
    model = RandomForestClassifier(random_state=42)
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    
    # Train
    trained_model, metrics = pipeline.train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        hyperparameters=hyperparameters,
        tags={"model_type": "random_forest", "experiment": "baseline"}
    )
    
    print(f"\nTraining Complete!")
    print(f"Metrics: {metrics}")
    
    return trained_model, metrics


def train_logistic_regression():
    """Train a Logistic Regression model."""
    print("\n" + "=" * 50)
    print("Training Logistic Regression Model")
    print("=" * 50)
    
    # Create data
    data = create_sample_data()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        experiment_name="classification_experiment",
        model_type="sklearn"
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        data=data,
        target_column="target"
    )
    
    # Define model and hyperparameters
    model = LogisticRegression(random_state=42, max_iter=1000)
    hyperparameters = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": 42
    }
    
    # Train
    trained_model, metrics = pipeline.train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        hyperparameters=hyperparameters,
        tags={"model_type": "logistic_regression", "experiment": "baseline"}
    )
    
    print(f"\nTraining Complete!")
    print(f"Metrics: {metrics}")
    
    return trained_model, metrics


def compare_experiments():
    """Compare experiment results."""
    print("\n" + "=" * 50)
    print("Comparing Experiments")
    print("=" * 50)
    
    tracker = ExperimentTracker()
    
    # Get best run
    best_run = tracker.get_best_run(
        experiment_name="classification_experiment",
        metric="f1_score"
    )
    
    if best_run:
        print(f"\nBest Run ID: {best_run.info.run_id}")
        print(f"Metrics: {best_run.data.metrics}")
        print(f"Parameters: {best_run.data.params}")
    
    # Compare top runs
    comparison_df = tracker.compare_runs(
        experiment_name="classification_experiment",
        metric="f1_score",
        top_n=5
    )
    
    if not comparison_df.empty:
        print("\nTop 5 Runs Comparison:")
        print(comparison_df[['run_id', 'accuracy', 'f1_score', 'precision', 'recall']].to_string())


def register_best_model():
    """Register the best model to the model registry."""
    print("\n" + "=" * 50)
    print("Registering Best Model")
    print("=" * 50)
    
    # Get best run
    tracker = ExperimentTracker()
    best_run = tracker.get_best_run(
        experiment_name="classification_experiment",
        metric="f1_score"
    )
    
    if not best_run:
        print("No runs found!")
        return
    
    # Register model
    registry = ModelRegistry()
    metadata = registry.register_model(
        model_name="classifier_model",
        run_id=best_run.info.run_id,
        description="Best performing classification model",
        tags={"use_case": "binary_classification", "dataset": "synthetic"}
    )
    
    print(f"\nModel Registered!")
    print(f"Name: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Metrics: {metadata.metrics}")
    
    # Promote to staging
    success = registry.promote_model(
        model_name="classifier_model",
        version=metadata.version,
        stage="Staging"
    )
    
    if success:
        print(f"\nModel promoted to Staging!")


def main():
    """Run the complete example."""
    # Train multiple models
    train_random_forest()
    train_logistic_regression()
    
    # Compare results
    compare_experiments()
    
    # Register best model
    register_best_model()
    
    print("\n" + "=" * 50)
    print("Example Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. View experiments in MLflow UI: mlflow ui")
    print("2. Start the serving API: python serving/api.py")
    print("3. Test predictions: curl -X POST http://localhost:8000/predict ...")


if __name__ == "__main__":
    main()

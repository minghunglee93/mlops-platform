"""
Feature Store Example - Complete Workflow

Demonstrates:
1. Feature engineering
2. Feature store setup
3. Feature ingestion
4. Online feature retrieval (serving)
5. Offline feature retrieval (training)
6. Integration with training pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

from feature_store.store import MLOpsFeatureStore
from feature_store.engineering import generate_sample_data_with_features
from training.pipeline import TrainingPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_feature_store():
    """
    Step 1: Initialize and setup the feature store.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Setting up Feature Store")
    logger.info("=" * 60)

    store = MLOpsFeatureStore(repo_path="./feature_repo")
    logger.info("✓ Feature store initialized\n")

    return store


def engineer_features():
    """
    Step 2: Engineer features from raw data.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Engineering Features")
    logger.info("=" * 60)

    # Generate sample data
    logger.info("Generating sample data...")
    df = generate_sample_data_with_features(n_samples=2000)

    logger.info(f"✓ Generated {len(df)} samples with {len(df.columns)} features")
    logger.info(f"  Features: {df.columns.tolist()[:10]}...")

    # Save for feature store
    feature_data_path = Path("./feature_repo/data")
    feature_data_path.mkdir(parents=True, exist_ok=True)

    # Prepare data for feature store (needs specific format)
    user_features = df[[
        'user_id',
        'event_timestamp',
        'age',
        'income',
        'amount',
        'hour',
        'day_of_week',
        'is_weekend'
    ]].copy()

    # Save to parquet
    parquet_path = feature_data_path / "user_features.parquet"
    user_features.to_parquet(parquet_path, index=False)
    logger.info(f"✓ Saved features to {parquet_path}")

    # Verify the file exists and is readable
    if parquet_path.exists():
        test_df = pd.read_parquet(parquet_path)
        logger.info(f"✓ Verified: {len(test_df)} rows, {len(test_df.columns)} columns")
        logger.info(f"  Columns: {test_df.columns.tolist()}")
    else:
        logger.error(f"✗ File not created: {parquet_path}")

    logger.info("")

    return df, user_features


def create_feature_definitions(store: MLOpsFeatureStore):
    """
    Step 3: Define features in the feature store.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Creating Feature Definitions")
    logger.info("=" * 60)

    from feast import Entity, FeatureView, Field, FileSource
    from feast.types import Float32, Int64, String
    from datetime import timedelta

    # Define entity
    user_entity = Entity(
        name="user",
        join_keys=["user_id"],
        description="User entity"
    )

    # Define data source
    user_source = FileSource(
        path="feature_repo/data/user_features.parquet",
        timestamp_field="event_timestamp",
    )

    # Define feature view
    user_features_view = FeatureView(
        name="user_features",
        entities=[user_entity],
        schema=[
            Field(name="age", dtype=Int64),
            Field(name="income", dtype=Float32),
            Field(name="amount", dtype=Float32),
            Field(name="hour", dtype=Int64),
            Field(name="day_of_week", dtype=Int64),
            Field(name="is_weekend", dtype=Int64),
        ],
        source=user_source,
        ttl=timedelta(days=365),
        online=True,
    )

    # Write feature definitions to file
    definitions_path = Path("./feature_repo/features_def.py")
    with open(definitions_path, 'w') as f:
        f.write("""
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

user_entity = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity"
)

user_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)

user_features = FeatureView(
    name="user_features",
    entities=[user_entity],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
        Field(name="amount", dtype=Float32),
        Field(name="hour", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
    ],
    source=user_source,
    ttl=timedelta(days=365),
    online=True,
)
""")

    logger.info(f"✓ Created feature definitions at {definitions_path}")

    # Apply definitions
    try:
        import subprocess
        result = subprocess.run(
            ["feast", "apply"],
            cwd="./feature_repo",
            capture_output=True,
            text=True
        )
        logger.info("✓ Applied feature definitions to Feast")
    except Exception as e:
        logger.warning(f"Could not apply with feast CLI: {e}")
        logger.info("  You can manually run: cd feature_repo && feast apply")

    logger.info("")


def materialize_features_to_online_store(store: MLOpsFeatureStore):
    """
    Step 4: Materialize features to online store for low-latency serving.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Materializing Features to Online Store")
    logger.info("=" * 60)

    try:
        from feast import FeatureStore as FeastStore
        import os
        import pytz

        # Change to feature repo directory
        original_dir = os.getcwd()
        os.chdir(store.repo_path)

        try:
            # Verify data file exists before materializing
            data_file = Path("data/user_features.parquet")
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_file.absolute()}")

            # Read and verify data
            test_df = pd.read_parquet(data_file)
            logger.info(f"✓ Data file verified: {len(test_df)} rows")
            logger.info(f"  Columns: {test_df.columns.tolist()}")

            # Get date range from data and ensure timezone-aware
            # Feast needs timezone-aware timestamps for materialization
            end_date = pd.Timestamp(test_df['event_timestamp'].max())
            start_date = pd.Timestamp(test_df['event_timestamp'].min())

            # Make timezone-aware if they aren't already
            if end_date.tz is None:
                end_date = end_date.tz_localize('UTC')
            if start_date.tz is None:
                start_date = start_date.tz_localize('UTC')

            logger.info(f"  Date range: {start_date} to {end_date}")
            logger.info(f"Materializing features...")

            # Materialize all feature views
            fs = FeastStore(repo_path=".")
            fs.materialize(start_date=start_date, end_date=end_date)

            logger.info("✓ Features materialized to online store")
            logger.info("  Features are now available for low-latency serving")

        finally:
            os.chdir(original_dir)

    except Exception as e:
        logger.error(f"Could not materialize: {e}")
        import traceback
        traceback.print_exc()

    logger.info("")


def retrieve_online_features(store: MLOpsFeatureStore):
    """
    Step 5: Retrieve features for online inference (serving).
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Retrieving Online Features (for Serving)")
    logger.info("=" * 60)

    try:
        from feast import FeatureStore as FeastStore
        import os

        # Change to feature repo directory
        original_dir = os.getcwd()
        os.chdir(store.repo_path)

        try:
            fs = FeastStore(repo_path=".")

            # Example: Get features for specific users
            entity_rows = [
                {"user_id": "user_0"},
                {"user_id": "user_1"},
                {"user_id": "user_2"},
            ]

            features = [
                "user_features:age",
                "user_features:income",
                "user_features:amount",
                "user_features:hour",
            ]

            # Get online features
            feature_vector = fs.get_online_features(
                features=features,
                entity_rows=entity_rows
            )

            online_features = feature_vector.to_df()

            logger.info("✓ Retrieved online features:")
            print(online_features)
            logger.info("")

        finally:
            os.chdir(original_dir)

    except Exception as e:
        logger.warning(f"Could not retrieve online features: {e}")
        import traceback
        traceback.print_exc()
        logger.info("  Make sure features are materialized first\n")


def retrieve_historical_features(store: MLOpsFeatureStore, df: pd.DataFrame):
    """
    Step 6: Retrieve historical features for training (point-in-time correct).
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Retrieving Historical Features (for Training)")
    logger.info("=" * 60)

    try:
        from feast import FeatureStore as FeastStore
        import os

        # Change to feature repo directory
        original_dir = os.getcwd()
        os.chdir(store.repo_path)

        try:
            fs = FeastStore(repo_path=".")

            # Create entity dataframe with timestamps
            entity_df = df[['user_id', 'event_timestamp']].head(100).copy()

            features = [
                "user_features:age",
                "user_features:income",
                "user_features:amount",
                "user_features:hour",
                "user_features:day_of_week",
                "user_features:is_weekend",
            ]

            # Get historical features
            training_job = fs.get_historical_features(
                entity_df=entity_df,
                features=features
            )

            training_df = training_job.to_df()

            logger.info(f"✓ Retrieved {len(training_df)} historical feature rows")
            logger.info("  Sample data:")
            print(training_df.head())
            logger.info("")

            return training_df

        finally:
            os.chdir(original_dir)

    except Exception as e:
        logger.warning(f"Could not retrieve historical features: {e}")
        import traceback
        traceback.print_exc()
        logger.info("  Continuing with generated features\n")
        return None


def train_model_with_features(df: pd.DataFrame):
    """
    Step 7: Train a model using the engineered features.
    """
    logger.info("=" * 60)
    logger.info("STEP 7: Training Model with Features")
    logger.info("=" * 60)

    # Select features for training
    feature_cols = [
        'age', 'income', 'amount', 'hour', 'day_of_week',
        'is_weekend', 'amount_zscore', 'age_zscore'
    ]

    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['target']

    # Initialize pipeline
    logger.info("Initializing Pipeline")
    pipeline = TrainingPipeline(
        experiment_name="feature_store_experiment",
        model_type="sklearn"
    )
    logger.info("Pipeline Initialized")

    logger.info("Preparing data")
    # Prepare train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    trained_model, metrics = pipeline.train(
        model=model,
        X_train=X_train.values,
        y_train=y_train.values,
        X_test=X_test.values,
        y_test=y_test.values,
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "features": feature_cols
        },
        tags={
            "model_type": "random_forest",
            "feature_store": "feast",
            "experiment": "feature_engineering"
        }
    )

    logger.info("✓ Model trained successfully!")
    logger.info(f"  Metrics: {metrics}")
    logger.info("")

    return trained_model, metrics


def demonstrate_feature_store_benefits():
    """
    Step 8: Demonstrate feature store benefits.
    """
    logger.info("=" * 60)
    logger.info("STEP 8: Feature Store Benefits")
    logger.info("=" * 60)

    benefits = """
    ✓ CONSISTENCY: Same features used in training and serving
    ✓ REUSABILITY: Features defined once, used by multiple models
    ✓ MONITORING: Track feature drift and quality
    ✓ GOVERNANCE: Centralized feature management and versioning
    ✓ PERFORMANCE: Low-latency online serving
    ✓ CORRECTNESS: Point-in-time correctness for training data
    ✓ COLLABORATION: Share features across teams
    """

    logger.info(benefits)
    logger.info("")


def main():
    """
    Run the complete feature store workflow.
    """
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE STORE EXAMPLE - COMPLETE WORKFLOW")
    logger.info("=" * 60 + "\n")

    try:
        # Step 1: Setup
        store = setup_feature_store()

        # Step 2: Engineer features
        df, user_features = engineer_features()

        # Step 3: Create feature definitions
        create_feature_definitions(store)

        # Step 4: Materialize to online store
        materialize_features_to_online_store(store)

        # Step 5: Retrieve online features
        retrieve_online_features(store)

        # Step 6: Retrieve historical features
        historical_df = retrieve_historical_features(store, df)

        # Step 7: Train model
        model, metrics = train_model_with_features(df)

        if model is None:
            logger.error("Training failed! Skipping remaining steps.")
        else:
            # Step 8: Show benefits
            demonstrate_feature_store_benefits()

        logger.info("=" * 60)
        logger.info("FEATURE STORE WORKFLOW COMPLETE! 🎉")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("  1. View experiments: mlflow ui")
        logger.info("  2. Explore features: cd feature_repo && feast feature-views list")
        logger.info("  3. Check feature registry: cd feature_repo && feast registry describe")
        logger.info("")

    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
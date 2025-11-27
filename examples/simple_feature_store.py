"""
Simple Feature Store Example - Focuses on core functionality

This is a simplified version that demonstrates the essential feature store workflow
without complex error handling. Use this to understand the basics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Simple feature store workflow."""

    logger.info("=" * 60)
    logger.info("SIMPLE FEATURE STORE EXAMPLE")
    logger.info("=" * 60)

    # Step 1: Create feature repo directory
    logger.info("\n1. Setting up feature repository...")
    repo_path = Path("./feature_repo_simple")
    repo_path.mkdir(exist_ok=True)
    (repo_path / "data").mkdir(exist_ok=True)

    # Create Feast config
    config_content = """
project: simple_features
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: file
entity_key_serialization_version: 2
"""

    with open(repo_path / "feature_store.yaml", 'w') as f:
        f.write(config_content)

    logger.info("✓ Feature repository created")

    # Step 2: Create sample data
    logger.info("\n2. Creating sample data...")

    n_samples = 100
    base_time = datetime.now()

    data = pd.DataFrame({
        'user_id': [f'user_{i % 10}' for i in range(n_samples)],
        'event_timestamp': [base_time - timedelta(days=i) for i in range(n_samples)],
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'transaction_count': np.random.randint(0, 50, n_samples),
    })

    # Ensure timestamps are timezone-aware (UTC)
    data['event_timestamp'] = pd.to_datetime(data['event_timestamp']).dt.tz_localize('UTC')

    # Save to parquet
    data_path = repo_path / "data" / "user_features.parquet"
    data.to_parquet(data_path, index=False)

    logger.info(f"✓ Created {len(data)} sample records")
    logger.info(f"  Saved to: {data_path}")

    # Step 3: Define features
    logger.info("\n3. Defining features...")

    features_definition = '''
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Define entity
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity"
)

# Define data source
user_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view
user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
        Field(name="transaction_count", dtype=Int64),
    ],
    source=user_source,
    ttl=timedelta(days=365),
    online=True,
)
'''

    with open(repo_path / "features.py", 'w') as f:
        f.write(features_definition)

    logger.info("✓ Feature definitions written to features.py")

    # Step 4: Apply feature definitions
    logger.info("\n4. Applying feature definitions...")

    original_dir = os.getcwd()
    os.chdir(repo_path)

    try:
        from feast import FeatureStore

        fs = FeatureStore(repo_path=".")

        # Import the definitions
        import importlib.util
        spec = importlib.util.spec_from_file_location("features", "features.py")
        features_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(features_module)

        # Apply them
        fs.apply([features_module.user, features_module.user_features])

        logger.info("✓ Feature definitions applied")

        # Verify
        views = fs.list_feature_views()
        logger.info(f"✓ Registered views: {[v.name for v in views]}")

        # Step 5: Materialize to online store
        logger.info("\n5. Materializing features to online store...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        fs.materialize(start_date=start_date, end_date=end_date)

        logger.info("✓ Features materialized")

        # Step 6: Get online features
        logger.info("\n6. Retrieving online features...")

        entity_rows = [
            {"user_id": "user_0"},
            {"user_id": "user_1"},
            {"user_id": "user_2"},
        ]

        features_to_fetch = [
            "user_features:age",
            "user_features:income",
            "user_features:transaction_count",
        ]

        feature_vector = fs.get_online_features(
            features=features_to_fetch,
            entity_rows=entity_rows
        )

        result_df = feature_vector.to_df()

        logger.info("✓ Online features retrieved:")
        print("\n", result_df)

        # Step 7: Get historical features
        logger.info("\n7. Retrieving historical features...")

        entity_df = data[['user_id', 'event_timestamp']].head(10)

        training_job = fs.get_historical_features(
            entity_df=entity_df,
            features=features_to_fetch
        )

        training_df = training_job.to_df()

        logger.info(f"✓ Historical features retrieved ({len(training_df)} rows):")
        print("\n", training_df.head())

        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS! Feature store is working correctly! 🎉")
        logger.info("=" * 60)
        logger.info("\nKey capabilities demonstrated:")
        logger.info("  ✓ Feature registration")
        logger.info("  ✓ Online serving (low latency)")
        logger.info("  ✓ Offline retrieval (training)")
        logger.info("  ✓ Point-in-time correctness")

        logger.info("\nNext steps:")
        logger.info("  1. Explore: cd feature_repo_simple && feast feature-views list")
        logger.info("  2. Add your own features in features.py")
        logger.info("  3. Integrate with training pipeline")

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        logger.info("\nTroubleshooting:")
        logger.info("  - Make sure Feast is installed: pip install feast")
        logger.info("  - Check that data files exist in feature_repo_simple/data/")
        logger.info("  - Try running from project root directory")

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
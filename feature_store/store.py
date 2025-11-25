"""
Feature Store implementation using Feast

Manages feature definitions, ingestion, and serving for both
training (offline) and inference (online) workloads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feast import FeatureStore, FeatureView, Entity, Field, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class MLOpsFeatureStore:
    """
    Wrapper around Feast for feature management.

    Provides high-level interface for:
    - Feature registration
    - Feature ingestion
    - Online feature retrieval (for serving)
    - Offline feature retrieval (for training)
    """

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize feature store.

        Args:
            repo_path: Path to Feast repository (default: ./feature_repo)
        """
        self.repo_path = Path(repo_path or "./feature_repo")
        self.repo_path.mkdir(parents=True, exist_ok=True)

        # Initialize Feast only if feature_store.yaml exists
        self.store = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or load Feast feature store."""
        feature_store_yaml = self.repo_path / "feature_store.yaml"

        if not feature_store_yaml.exists():
            logger.info("Creating new Feast repository...")
            self._create_feast_repo()

        try:
            self.store = FeatureStore(repo_path=str(self.repo_path))
            logger.info(f"Feature store initialized at {self.repo_path}")
        except Exception as e:
            logger.error(f"Error initializing feature store: {e}")
            raise

    def _create_feast_repo(self):
        """Create Feast repository configuration."""
        config_content = """
project: mlops_features
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: file
entity_key_serialization_version: 2
"""
        feature_store_yaml = self.repo_path / "feature_store.yaml"
        with open(feature_store_yaml, 'w') as f:
            f.write(config_content)

        # Create data directory
        (self.repo_path / "data").mkdir(exist_ok=True)

        logger.info(f"Created Feast configuration at {feature_store_yaml}")

    def register_entity(
            self,
            name: str,
            description: str,
            join_keys: List[str]
    ) -> Entity:
        """
        Register an entity (e.g., user, product).

        Args:
            name: Entity name
            description: Entity description
            join_keys: Keys used for joining

        Returns:
            Feast Entity object
        """
        entity = Entity(
            name=name,
            join_keys=join_keys,
            description=description
        )

        logger.info(f"Registered entity: {name}")
        return entity

    def create_feature_view(
            self,
            name: str,
            entities: List[Entity],
            schema: List[Field],
            source: FileSource,
            ttl: timedelta = timedelta(days=365)
    ) -> FeatureView:
        """
        Create a feature view definition.

        Args:
            name: Feature view name
            entities: List of entities
            schema: Feature schema
            source: Data source
            ttl: Time-to-live for features

        Returns:
            FeatureView object
        """
        feature_view = FeatureView(
            name=name,
            entities=entities,
            schema=schema,
            source=source,
            ttl=ttl
        )

        logger.info(f"Created feature view: {name}")
        return feature_view

    def ingest_features(
            self,
            data: pd.DataFrame,
            feature_view_name: str
    ):
        """
        Ingest features into the feature store.

        Args:
            data: DataFrame with features
            feature_view_name: Name of feature view to ingest to
        """
        try:
            # Save data to parquet (required by Feast)
            data_path = self.repo_path / "data" / f"{feature_view_name}.parquet"
            data.to_parquet(data_path, index=False)

            logger.info(f"Ingested {len(data)} rows to {feature_view_name}")

            # Materialize features to online store
            self.materialize_features(feature_view_name)

        except Exception as e:
            logger.error(f"Error ingesting features: {e}")
            raise

    def materialize_features(
            self,
            feature_view_name: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ):
        """
        Materialize features from offline to online store.

        Args:
            feature_view_name: Specific feature view (None for all)
            start_date: Start date for materialization
            end_date: End date for materialization
        """
        try:
            end_date = end_date or datetime.now()
            start_date = start_date or (end_date - timedelta(days=365))

            if feature_view_name:
                self.store.materialize(
                    start_date=start_date,
                    end_date=end_date,
                    feature_views=[feature_view_name]
                )
            else:
                self.store.materialize(
                    start_date=start_date,
                    end_date=end_date
                )

            logger.info(f"Materialized features: {feature_view_name or 'all'}")

        except Exception as e:
            logger.error(f"Error materializing features: {e}")
            raise

    def get_online_features(
            self,
            entity_rows: List[Dict[str, Any]],
            features: List[str]
    ) -> pd.DataFrame:
        """
        Get features for online inference (low latency).

        Args:
            entity_rows: List of entity identifiers
            features: List of feature references (e.g., 'view:feature')

        Returns:
            DataFrame with requested features
        """
        try:
            feature_vector = self.store.get_online_features(
                features=features,
                entity_rows=entity_rows
            )

            df = feature_vector.to_df()
            logger.info(f"Retrieved {len(df)} online feature vectors")
            return df

        except Exception as e:
            logger.error(f"Error getting online features: {e}")
            raise

    def get_historical_features(
            self,
            entity_df: pd.DataFrame,
            features: List[str]
    ) -> pd.DataFrame:
        """
        Get historical features for training (point-in-time correct).

        Args:
            entity_df: DataFrame with entity IDs and timestamps
            features: List of feature references

        Returns:
            DataFrame with historical features
        """
        try:
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=features
            ).to_df()

            logger.info(f"Retrieved {len(training_df)} historical feature rows")
            return training_df

        except Exception as e:
            logger.error(f"Error getting historical features: {e}")
            raise

    def list_feature_views(self) -> List[str]:
        """
        List all registered feature views.

        Returns:
            List of feature view names
        """
        try:
            views = self.store.list_feature_views()
            return [view.name for view in views]
        except Exception as e:
            logger.error(f"Error listing feature views: {e}")
            return []

    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        """
        Get a specific feature view.

        Args:
            name: Feature view name

        Returns:
            FeatureView object or None
        """
        try:
            return self.store.get_feature_view(name)
        except Exception as e:
            logger.warning(f"Feature view not found: {name}")
            return None

    def apply_feature_definitions(self, definitions_path: Optional[Path] = None):
        """
        Apply feature definitions from Python files.

        Args:
            definitions_path: Path to definitions (default: feature_repo/)
        """
        try:
            self.store.apply(objects=[], objects_path=str(definitions_path or self.repo_path))
            logger.info("Applied feature definitions")
        except Exception as e:
            logger.error(f"Error applying definitions: {e}")
            raise


def create_sample_features() -> pd.DataFrame:
    """
    Create sample feature data for demonstration.

    Returns:
        DataFrame with sample features
    """
    from datetime import datetime, timedelta
    import numpy as np

    # Generate sample data
    n_samples = 1000
    base_time = datetime.now()

    data = {
        'entity_id': [f'entity_{i}' for i in range(n_samples)],
        'event_timestamp': [base_time - timedelta(days=np.random.randint(0, 365))
                            for _ in range(n_samples)],
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randint(0, 10, n_samples),
        'feature_4': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    logger.info("Initializing feature store...")

    store = MLOpsFeatureStore()

    # Create sample data
    logger.info("Creating sample features...")
    sample_data = create_sample_features()

    logger.info(f"Sample data shape: {sample_data.shape}")
    logger.info(f"Columns: {sample_data.columns.tolist()}")
    logger.info("\nSample data preview:")
    print(sample_data.head())

    logger.info("\nFeature store ready!")
"""
Feature engineering utilities

Provides functions to compute and transform features for the feature store.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for creating ML features.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_transformers = {}

    def create_time_features(
            self,
            df: pd.DataFrame,
            timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Create time-based features.

        Args:
            df: Input dataframe
            timestamp_col: Timestamp column name

        Returns:
            DataFrame with time features
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)

        logger.info("Created time features")
        return df

    def create_aggregation_features(
            self,
            df: pd.DataFrame,
            group_by: str,
            value_col: str,
            windows: List[int] = [7, 30, 90]
    ) -> pd.DataFrame:
        """
        Create rolling aggregation features.

        Args:
            df: Input dataframe
            group_by: Column to group by
            value_col: Column to aggregate
            windows: List of window sizes in days

        Returns:
            DataFrame with aggregation features
        """
        df = df.copy()
        df = df.sort_values('event_timestamp')

        for window in windows:
            # Count
            df[f'{value_col}_count_{window}d'] = df.groupby(group_by)[value_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).count()
            )

            # Sum
            df[f'{value_col}_sum_{window}d'] = df.groupby(group_by)[value_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )

            # Mean
            df[f'{value_col}_mean_{window}d'] = df.groupby(group_by)[value_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            # Std
            df[f'{value_col}_std_{window}d'] = df.groupby(group_by)[value_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

        logger.info(f"Created aggregation features for windows: {windows}")
        return df

    def create_categorical_features(
            self,
            df: pd.DataFrame,
            categorical_cols: List[str],
            method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df: Input dataframe
            categorical_cols: List of categorical columns
            method: Encoding method ('onehot' or 'label')

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        if method == "onehot":
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        elif method == "label":
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

        logger.info(f"Encoded {len(categorical_cols)} categorical features using {method}")
        return df

    def create_interaction_features(
            self,
            df: pd.DataFrame,
            feature_pairs: List[tuple]
    ) -> pd.DataFrame:
        """
        Create interaction features between pairs of features.

        Args:
            df: Input dataframe
            feature_pairs: List of (feature1, feature2) tuples

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]

                # Division (avoid division by zero)
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)

        logger.info(f"Created {len(feature_pairs)} interaction features")
        return df

    def create_statistical_features(
            self,
            df: pd.DataFrame,
            numeric_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create statistical features from numeric columns.

        Args:
            df: Input dataframe
            numeric_cols: List of numeric columns

        Returns:
            DataFrame with statistical features
        """
        df = df.copy()

        # Z-score normalization
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            df[f'{col}_zscore'] = (df[col] - mean) / (std + 1e-6)

        # Min-max normalization
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val + 1e-6)

        # Log transform
        for col in numeric_cols:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

        logger.info(f"Created statistical features for {len(numeric_cols)} columns")
        return df

    def compute_feature_importance(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            method: str = "random_forest"
    ) -> pd.DataFrame:
        """
        Compute feature importance scores.

        Args:
            X: Feature matrix
            y: Target variable
            method: Method to use ('random_forest' or 'mutual_info')

        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif

        if method == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importance = model.feature_importances_
        elif method == "mutual_info":
            importance = mutual_info_classif(X, y, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info(f"Computed feature importance using {method}")
        return importance_df


def generate_sample_data_with_features(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample data with engineered features.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with features
    """
    np.random.seed(42)

    # Generate base data
    base_time = datetime.now()

    data = {
        'user_id': [f'user_{i % 100}' for i in range(n_samples)],
        'event_timestamp': [base_time - timedelta(days=np.random.randint(0, 365))
                            for _ in range(n_samples)],
        'amount': np.random.exponential(50, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU'], n_samples),
    }

    df = pd.DataFrame(data)

    # Apply feature engineering
    engineer = FeatureEngineer()

    # Time features
    df = engineer.create_time_features(df, 'event_timestamp')

    # Statistical features
    df = engineer.create_statistical_features(df, ['amount', 'age', 'income'])

    # Interaction features
    df = engineer.create_interaction_features(df, [('amount', 'age'), ('income', 'age')])

    # Add target variable
    df['target'] = (
            (df['amount'] > df['amount'].median()) &
            (df['age'] > df['age'].median())
    ).astype(int)

    logger.info(f"Generated {len(df)} samples with {len(df.columns)} features")

    return df


if __name__ == "__main__":
    # Example usage
    logger.info("Generating sample data with features...")
    df = generate_sample_data_with_features(1000)

    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info("\nSample data:")
    print(df.head())

    # Compute feature importance
    engineer = FeatureEngineer()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'target']

    X = df[numeric_cols].fillna(0)
    y = df['target']

    importance = engineer.compute_feature_importance(X, y)
    logger.info("\nTop 10 important features:")
    print(importance.head(10))
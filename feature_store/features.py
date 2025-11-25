"""
Feature definitions for Feast feature store

This file defines:
- Entities (e.g., user, transaction)
- Feature views (collections of features)
- Data sources
"""
from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Int64, String, Bool
from datetime import timedelta
from pathlib import Path


# ===== Data Sources =====

# Example: User features from parquet file
user_source = FileSource(
    path="feature_repo/data/user_features.parquet",
    timestamp_field="event_timestamp",
)

# Example: Transaction features
transaction_source = FileSource(
    path="feature_repo/data/transaction_features.parquet",
    timestamp_field="event_timestamp",
)


# ===== Entities =====

# User entity
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity for ML features"
)

# Transaction entity
transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    description="Transaction entity"
)


# ===== Feature Views =====

# User demographic features
user_demographics = FeatureView(
    name="user_demographics",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
        Field(name="account_age_days", dtype=Int64),
        Field(name="country", dtype=String),
        Field(name="is_premium", dtype=Bool),
    ],
    source=user_source,
    ttl=timedelta(days=365),
    online=True,
    description="User demographic features"
)

# User activity features
user_activity = FeatureView(
    name="user_activity",
    entities=[user],
    schema=[
        Field(name="login_count_7d", dtype=Int64),
        Field(name="login_count_30d", dtype=Int64),
        Field(name="transaction_count_7d", dtype=Int64),
        Field(name="transaction_count_30d", dtype=Int64),
        Field(name="total_spent_7d", dtype=Float32),
        Field(name="total_spent_30d", dtype=Float32),
        Field(name="avg_transaction_value", dtype=Float32),
    ],
    source=user_source,
    ttl=timedelta(days=7),  # Fresher TTL for activity features
    online=True,
    description="User activity and engagement features"
)

# User features
user_features = FeatureView(
    name="user_features",
    entities=[user],
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

# Transaction features
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="merchant_category", dtype=String),
        Field(name="is_international", dtype=Bool),
        Field(name="device_type", dtype=String),
        Field(name="risk_score", dtype=Float32),
    ],
    source=transaction_source,
    ttl=timedelta(days=30),
    online=True,
    description="Transaction-level features"
)


# ===== Feature Service =====
# Group related features for specific ML use cases

from feast import FeatureService

# Feature service for fraud detection model
fraud_detection_service = FeatureService(
    name="fraud_detection_v1",
    features=[
        user_demographics,
        user_activity,
        transaction_features
    ],
    description="Features for fraud detection model"
)

# Feature service for credit scoring
credit_scoring_service = FeatureService(
    name="credit_scoring_v1",
    features=[
        user_demographics[["age", "income", "account_age_days"]],
        user_activity[["transaction_count_30d", "total_spent_30d", "avg_transaction_value"]]
    ],
    description="Features for credit scoring model"
)
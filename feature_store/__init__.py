"""
Feature Store module for MLOps platform

Provides centralized feature management with:
- Feature registration and versioning
- Online serving (low latency)
- Offline retrieval (training)
- Point-in-time correctness
"""
from .store import MLOpsFeatureStore, create_sample_features

__all__ = ['MLOpsFeatureStore', 'create_sample_features']
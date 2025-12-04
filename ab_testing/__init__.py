"""
A/B Testing module for MLOps platform

Provides experiment management, traffic splitting, and statistical testing.
"""
from .experiment import ABExperiment, ModelVariant, TrafficSplitStrategy

__all__ = ['ABExperiment', 'ModelVariant', 'TrafficSplitStrategy']

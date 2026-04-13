"""Utility modules for rehabilitation RL."""

from .callbacks import DebugCallback
from .feature_extractors import (
    MLPOnlyExtractor,
    BiResidualGatedExtractor,
    FiLMFeatureExtractor,
    LSTMExtractor,
    AuxLSTMExtractor,
    EXTRACTOR_REGISTRY,
)
from .ablation_callbacks import (
    AuxTrainingCallback,
    PPOAuxTrainingCallback,
    SaveMetricsCallback,
)

__all__ = [
    'DebugCallback',
    'MLPOnlyExtractor',
    'BiResidualGatedExtractor',
    'FiLMFeatureExtractor',
    'LSTMExtractor',
    'AuxLSTMExtractor',
    'EXTRACTOR_REGISTRY',
    'AuxTrainingCallback',
    'PPOAuxTrainingCallback',
    'SaveMetricsCallback',
]

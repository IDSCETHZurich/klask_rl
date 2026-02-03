"""Backward compatibility module - imports from experiment_config.

This module provides backward compatibility for scripts that still import
from train_config. New code should use experiment_config.ExperimentConfig directly.
"""

from experiment_config import ExperimentConfig

# Backward compatibility alias
TrainConfig = ExperimentConfig

__all__ = ["TrainConfig", "ExperimentConfig"]

"""
DEPRECATED: This module is deprecated.

Use soilspec.training.deep_learning_trainer instead.

The term "PINN" (Physics-Informed Neural Networks) is misleading in this
context. Traditional PINNs solve partial differential equations, which is
not what we do in soil spectroscopy. We use physics-informed FEATURE
ENGINEERING (spectral band assignments), not physics-informed LEARNING.

This module is provided for backwards compatibility only and will be
removed in v0.2.0.
"""

import warnings

warnings.warn(
    "Module 'soilspec.training.pinn_trainer' is deprecated and will be removed in v0.2.0. "
    "Use 'soilspec.training.deep_learning_trainer' instead. "
    "The term 'PINN' is misleading for standard neural networks with physics-informed features.",
    DeprecationWarning,
    stacklevel=2
)

# Import for backwards compatibility
from soilspec.training.deep_learning_trainer import PINNTrainer

__all__ = ['PINNTrainer']

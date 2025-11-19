"""
Prediction and uncertainty quantification module.

Provides unified prediction interface with support for ensemble,
conformal, and dropout-based uncertainty quantification.
"""

from soilspec_pinn.prediction.predictor import SpectralPredictor
from soilspec_pinn.prediction.uncertainty import (
    EnsembleUncertainty,
    ConformalPrediction,
    DropoutUncertainty,
)

__all__ = [
    "SpectralPredictor",
    "EnsembleUncertainty",
    "ConformalPrediction",
    "DropoutUncertainty",
]

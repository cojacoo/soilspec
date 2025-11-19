"""
Traditional machine learning models for soil spectroscopy.

Strong models that outperform simple PLS/Random Forest:
- MBL (Memory-Based Learning): Local modeling, excellent for transfer learning
- Cubist: OSSL standard, rule-based + local linear regression
"""

from soilspec.models.traditional.mbl import MBLRegressor
from soilspec.models.traditional.cubist_wrapper import (
    CubistRegressor,
    OSSLCubistPredictor,
)

__all__ = [
    # Memory-Based Learning
    "MBLRegressor",
    # Cubist (OSSL standard)
    "CubistRegressor",
    "OSSLCubistPredictor",
]

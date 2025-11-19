"""
SoilSpec: Evidence-Based Soil Spectroscopy Package

A comprehensive Python package for analyzing soil mid-infrared spectra using
domain knowledge (spectral_bands.csv), traditional chemometrics (PLS, MBL, Cubist),
and interpretable deep learning with physics-guided attention.

Combines 150+ literature-referenced peak assignments with modern machine learning
for accurate, interpretable soil property prediction.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from soilspec import (
    io,
    preprocessing,
    knowledge,
    features,
    models,
    training,
    prediction,
    integration,
    utils,
    datasets,
)

__all__ = [
    "io",
    "preprocessing",
    "knowledge",
    "features",
    "models",
    "training",
    "prediction",
    "integration",
    "utils",
    "datasets",
    "__version__",
]

"""
SoilSpec-PINN: Physics-Informed Neural Networks for Soil Spectroscopy

A comprehensive Python package for analyzing soil mid-infrared spectra using
physics-informed neural networks (PINNs), message passing neural networks (MPNNs),
and traditional machine learning methods.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from soilspec_pinn import (
    io,
    preprocessing,
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
    "models",
    "training",
    "prediction",
    "integration",
    "utils",
    "datasets",
    "__version__",
]

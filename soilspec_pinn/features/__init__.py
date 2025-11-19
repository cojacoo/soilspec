"""
Feature extraction module for physics-informed spectral analysis.

Provides tools to extract chemically meaningful features from soil spectra
based on domain knowledge from spectral_bands.csv.

Example:
    >>> from soilspec_pinn.features import PhysicsInformedFeatures
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from sklearn.pipeline import Pipeline
    >>>
    >>> # Enhanced PLS with physics features
    >>> model = Pipeline([
    ...     ('features', PhysicsInformedFeatures()),
    ...     ('scaler', StandardScaler()),
    ...     ('pls', PLSRegression(n_components=10))
    ... ])
    >>>
    >>> model.fit(spectra, soil_properties, features__wavenumbers=wavenumbers)
"""

from .peak_integration import PeakIntegrator, PeakHeightExtractor
from .ratios import SpectralRatios, SpectralIndices
from .transformers import (
    PhysicsInformedFeatures,
    CompactFeatures,
    ExtensiveFeatures
)

__all__ = [
    # Peak-based features
    'PeakIntegrator',
    'PeakHeightExtractor',

    # Ratio and index features
    'SpectralRatios',
    'SpectralIndices',

    # Combined transformers
    'PhysicsInformedFeatures',
    'CompactFeatures',
    'ExtensiveFeatures',
]

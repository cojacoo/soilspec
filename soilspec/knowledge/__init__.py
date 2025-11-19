"""
Knowledge module for soil spectroscopy domain knowledge.

Provides access to spectral band assignments, chemical constraints,
and visualization tools based on literature-referenced peak assignments.

Example:
    >>> from soilspec.knowledge import (
    ...     SpectralBandDatabase,
    ...     ChemicalConstraints,
    ...     SpectralPlotter
    ... )
    >>> # Load spectral band database
    >>> bands = SpectralBandDatabase()
    >>> print(bands.summarize())
    >>>
    >>> # Get organic bands
    >>> organic_bands = bands.get_bands(type='org')
    >>>
    >>> # Validate predictions
    >>> constraints = ChemicalConstraints()
    >>> result = constraints.validate_prediction({'SOC': 2.5, 'clay': 25})
"""

from .band_parser import SpectralBandDatabase
from .constraints import ChemicalConstraints
from .visualization import SpectralPlotter, plot_band_summary

__all__ = [
    'SpectralBandDatabase',
    'ChemicalConstraints',
    'SpectralPlotter',
    'plot_band_summary',
]

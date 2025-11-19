"""
Utility functions for spectral processing and analysis.

Includes spectral processing utilities, validation tools,
and visualization functions.
"""

from soilspec_pinn.utils.spectral import interpolate_spectrum, baseline_als
from soilspec_pinn.utils.validation import train_test_split_spectral, cross_validate
from soilspec_pinn.utils.visualization import plot_spectrum, plot_predictions

__all__ = [
    "interpolate_spectrum",
    "baseline_als",
    "train_test_split_spectral",
    "cross_validate",
    "plot_spectrum",
    "plot_predictions",
]

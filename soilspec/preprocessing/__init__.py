"""
Spectral preprocessing module.

Provides scikit-learn compatible transformers for baseline correction,
derivatives, smoothing, and other spectral preprocessing operations.

All transformers wrap scipy/pywavelets for proven signal processing.
"""

from soilspec.preprocessing.baseline import (
    SNVTransformer,
    MSCTransformer,
    DetrendTransformer,
)
from soilspec.preprocessing.derivatives import (
    SavitzkyGolayDerivative,
    GapSegmentDerivative,
)
from soilspec.preprocessing.smoothing import (
    SavitzkyGolaySmoother,
    WaveletDenoiser,
    MovingAverageSmoother,
)
from soilspec.preprocessing.resample import (
    SpectralResample,
    TrimSpectrum,
)

__all__ = [
    # Baseline correction
    "SNVTransformer",
    "MSCTransformer",
    "DetrendTransformer",
    # Derivatives
    "SavitzkyGolayDerivative",
    "GapSegmentDerivative",
    # Smoothing
    "SavitzkyGolaySmoother",
    "WaveletDenoiser",
    "MovingAverageSmoother",
    # Resampling
    "SpectralResample",
    "TrimSpectrum",
]

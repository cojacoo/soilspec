"""
Spectral preprocessing module.

Provides scikit-learn compatible transformers for baseline correction,
derivatives, smoothing, and other spectral preprocessing operations.
"""

from soilspec.preprocessing.baseline import SNVTransformer, MSCTransformer
from soilspec.preprocessing.derivatives import SavitzkyGolayDerivative
from soilspec.preprocessing.smoothing import WaveletDenoise, SavGolSmooth
from soilspec.preprocessing.transforms import ToAbsorbance, SpectralResample, TrimTransformer
from soilspec.preprocessing.pipeline import create_preprocessing_pipeline

__all__ = [
    "SNVTransformer",
    "MSCTransformer",
    "SavitzkyGolayDerivative",
    "WaveletDenoise",
    "SavGolSmooth",
    "ToAbsorbance",
    "SpectralResample",
    "TrimTransformer",
    "create_preprocessing_pipeline",
]

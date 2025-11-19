"""
Spectral preprocessing module.

Provides scikit-learn compatible transformers for baseline correction,
derivatives, smoothing, and other spectral preprocessing operations.
"""

from soilspec_pinn.preprocessing.baseline import SNVTransformer, MSCTransformer
from soilspec_pinn.preprocessing.derivatives import SavitzkyGolayDerivative
from soilspec_pinn.preprocessing.smoothing import WaveletDenoise, SavGolSmooth
from soilspec_pinn.preprocessing.transforms import ToAbsorbance, SpectralResample, TrimTransformer
from soilspec_pinn.preprocessing.pipeline import create_preprocessing_pipeline

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

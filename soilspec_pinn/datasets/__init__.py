"""
Dataset loaders and augmentation utilities.

Provides loaders for common soil spectroscopy datasets
and spectral augmentation techniques.
"""

from soilspec_pinn.datasets.loaders import OSSLDataset, LUCASDataset
from soilspec_pinn.datasets.augmentation import SpectralAugmenter, add_noise, shift_baseline

__all__ = [
    "OSSLDataset",
    "LUCASDataset",
    "SpectralAugmenter",
    "add_noise",
    "shift_baseline",
]

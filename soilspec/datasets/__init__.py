"""
Dataset loaders and augmentation utilities.

Provides loaders for common soil spectroscopy datasets (OSSL, LUCAS)
and spectral augmentation techniques for improving model robustness.

**Data Loaders:**

* OSSLDataset: Open Soil Spectral Library (100k+ global samples)
* LUCASDataset: LUCAS European topsoil database (not yet implemented)

**Augmentation:**

* SpectralAugmenter: Combined augmentation (noise, baseline, scaling)
* Individual augmentation functions for custom workflows

Example Usage
-------------
>>> from soilspec.datasets import OSSLDataset, SpectralAugmenter
>>>
>>> # Load OSSL data
>>> ossl = OSSLDataset()
>>> X, y, ids = ossl.load_mir(target='soc', wmin=600, wmax=4000)
>>> print(f"Loaded {X.shape[0]} samples")
>>>
>>> # Split dataset
>>> splits = ossl.split_dataset(X, y, ids, test_size=0.2, val_size=0.1)
>>>
>>> # Augment training data
>>> aug = SpectralAugmenter(noise_level=0.01, p=0.3)
>>> X_train_aug = aug.fit_transform(splits['X_train'])

Notes
-----
Requires ``pip install soilspecdata`` for OSSL data access.

See Albinet et al. (2024) for soilspecdata package details:
https://github.com/franckalbinet/soilspecdata
"""

from soilspec.datasets.loaders import OSSLDataset, LUCASDataset
from soilspec.datasets.augmentation import (
    SpectralAugmenter,
    add_noise,
    shift_baseline,
    scale_spectrum,
    wavelength_shift_augment,
    mixup_spectra
)

__all__ = [
    # Data loaders
    "OSSLDataset",
    "LUCASDataset",
    # Augmentation
    "SpectralAugmenter",
    "add_noise",
    "shift_baseline",
    "scale_spectrum",
    "wavelength_shift_augment",
    "mixup_spectra",
]

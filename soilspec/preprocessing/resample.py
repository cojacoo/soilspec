"""
Spectral resampling using scipy.interpolate.

Wraps scipy interpolation functions for sklearn compatibility.
"""

import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union


class SpectralResample(BaseEstimator, TransformerMixin):
    """
    Resample spectra to target wavenumbers using interpolation.

    Common need when combining spectra from different instruments or
    standardizing to OSSL wavenumber grid.

    Args:
        target_wavenumbers: Target wavenumber array (cm⁻¹)
        kind: Interpolation type ('linear', 'cubic', 'quadratic')
        fill_value: How to handle extrapolation ('extrapolate' or value)

    Example:
        >>> from soilspec.preprocessing import SpectralResample
        >>> # Resample to 2 cm⁻¹ spacing
        >>> target_wn = np.arange(600, 4001, 2)
        >>> resampler = SpectralResample(target_wavenumbers=target_wn)
        >>> resampler.fit(spectra, wavenumbers=original_wn)
        >>> resampled = resampler.transform(spectra)

    Reference:
        Common in prospectr R package and ADDRESS workflow
    """

    def __init__(
        self,
        target_wavenumbers: Optional[np.ndarray] = None,
        kind: str = 'linear',
        fill_value: Union[str, float] = 'extrapolate'
    ):
        """
        Initialize resampler.

        Args:
            target_wavenumbers: Target wavenumber grid
            kind: Interpolation method
            fill_value: Extrapolation handling
        """
        self.target_wavenumbers = target_wavenumbers
        self.kind = kind
        self.fill_value = fill_value
        self.original_wavenumbers_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, wavenumbers=None):
        """
        Fit resampler by storing original wavenumbers.

        Args:
            X: Spectra array (n_samples, n_features)
            y: Target values (ignored)
            wavenumbers: Original wavenumber array (required)

        Returns:
            self
        """
        if wavenumbers is None:
            raise ValueError(
                "wavenumbers must be provided during fit. "
                "Pass as: resampler.fit(X, wavenumbers=wavenumber_array)"
            )

        self.original_wavenumbers_ = np.array(wavenumbers)

        # Validate target wavenumbers
        if self.target_wavenumbers is None:
            raise ValueError(
                "target_wavenumbers must be set during __init__ or provided during fit"
            )

        self.target_wavenumbers = np.array(self.target_wavenumbers)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Resample spectra to target wavenumbers.

        Args:
            X: Spectra array (n_samples, n_features)

        Returns:
            Resampled spectra (n_samples, n_target_features)
        """
        if self.original_wavenumbers_ is None:
            raise ValueError("Transformer must be fitted first")

        X = np.asarray(X)
        n_samples = X.shape[0]
        n_target = len(self.target_wavenumbers)

        X_resampled = np.zeros((n_samples, n_target))

        for i in range(n_samples):
            # Create interpolator for this spectrum
            interpolator = interp1d(
                self.original_wavenumbers_,
                X[i, :],
                kind=self.kind,
                bounds_error=False,
                fill_value=self.fill_value
            )

            # Interpolate to target wavenumbers
            X_resampled[i, :] = interpolator(self.target_wavenumbers)

        return X_resampled

    def set_target_wavenumbers(self, wavenumbers: np.ndarray):
        """
        Set target wavenumbers after initialization.

        Args:
            wavenumbers: New target wavenumber array
        """
        self.target_wavenumbers = np.array(wavenumbers)
        return self


class TrimSpectrum(BaseEstimator, TransformerMixin):
    """
    Trim spectrum to specified wavenumber range.

    Args:
        min_wavenumber: Minimum wavenumber to keep
        max_wavenumber: Maximum wavenumber to keep

    Example:
        >>> from soilspec.preprocessing import TrimSpectrum
        >>> # Keep only fingerprint region
        >>> trimmer = TrimSpectrum(min_wavenumber=600, max_wavenumber=1500)
        >>> trimmer.fit(spectra, wavenumbers=wavenumbers)
        >>> trimmed = trimmer.transform(spectra)
    """

    def __init__(
        self,
        min_wavenumber: Optional[float] = None,
        max_wavenumber: Optional[float] = None
    ):
        """
        Initialize trimmer.

        Args:
            min_wavenumber: Minimum wavenumber
            max_wavenumber: Maximum wavenumber
        """
        self.min_wavenumber = min_wavenumber
        self.max_wavenumber = max_wavenumber
        self.mask_ = None
        self.trimmed_wavenumbers_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, wavenumbers=None):
        """
        Fit by determining which wavelengths to keep.

        Args:
            X: Spectra array (n_samples, n_features)
            y: Target values (ignored)
            wavenumbers: Wavenumber array (required)

        Returns:
            self
        """
        if wavenumbers is None:
            raise ValueError("wavenumbers must be provided during fit")

        wavenumbers = np.array(wavenumbers)

        # Create mask
        mask = np.ones(len(wavenumbers), dtype=bool)

        if self.min_wavenumber is not None:
            mask &= (wavenumbers >= self.min_wavenumber)

        if self.max_wavenumber is not None:
            mask &= (wavenumbers <= self.max_wavenumber)

        self.mask_ = mask
        self.trimmed_wavenumbers_ = wavenumbers[mask]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Trim spectra to selected range.

        Args:
            X: Spectra array (n_samples, n_features)

        Returns:
            Trimmed spectra (n_samples, n_selected_features)
        """
        if self.mask_ is None:
            raise ValueError("Transformer must be fitted first")

        return X[:, self.mask_]

    def get_trimmed_wavenumbers(self) -> np.ndarray:
        """Get wavenumbers after trimming."""
        if self.trimmed_wavenumbers_ is None:
            raise ValueError("Transformer must be fitted first")

        return self.trimmed_wavenumbers_

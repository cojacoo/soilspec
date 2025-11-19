"""
Baseline correction transformers for spectral data.

Implements scikit-learn compatible transformers for SNV, MSC,
and other baseline correction methods.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class SNVTransformer(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) transformation.

    SNV centers and scales each spectrum individually to have mean=0 and std=1.
    This removes multiplicative scatter effects.

    Attributes:
        with_mean: Whether to center the data (default: True)
        with_std: Whether to scale the data (default: True)

    Example:
        >>> from soilspec_pinn.preprocessing import SNVTransformer
        >>> snv = SNVTransformer()
        >>> spectra_corrected = snv.fit_transform(spectra)
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Initialize SNV transformer.

        Args:
            with_mean: Whether to center spectra to mean=0
            with_std: Whether to scale spectra to std=1
        """
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SNVTransformer":
        """
        Fit the transformer (no-op for SNV, included for sklearn compatibility).

        Args:
            X: Input spectra array of shape (n_samples, n_features)
            y: Target values (ignored)

        Returns:
            self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply SNV transformation to spectra.

        Args:
            X: Input spectra array of shape (n_samples, n_features)

        Returns:
            Transformed spectra array
        """
        X = np.asarray(X)
        X_transformed = X.copy()

        # Apply row-wise standardization
        for i in range(X_transformed.shape[0]):
            spectrum = X_transformed[i, :]

            if self.with_mean:
                spectrum = spectrum - np.mean(spectrum)

            if self.with_std:
                std = np.std(spectrum)
                if std > 1e-10:  # Avoid division by zero
                    spectrum = spectrum / std

            X_transformed[i, :] = spectrum

        return X_transformed


class MSCTransformer(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction (MSC) transformation.

    MSC corrects for additive and multiplicative scatter effects by
    regressing each spectrum against a reference spectrum (typically
    the mean spectrum).

    Attributes:
        reference_spectrum: Reference spectrum for correction

    Example:
        >>> from soilspec_pinn.preprocessing import MSCTransformer
        >>> msc = MSCTransformer()
        >>> spectra_corrected = msc.fit_transform(spectra)
    """

    def __init__(self, reference: Optional[np.ndarray] = None):
        """
        Initialize MSC transformer.

        Args:
            reference: Optional reference spectrum. If None, mean spectrum is used.
        """
        self.reference = reference
        self.reference_spectrum_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MSCTransformer":
        """
        Fit the transformer by computing reference spectrum.

        Args:
            X: Input spectra array of shape (n_samples, n_features)
            y: Target values (ignored)

        Returns:
            self
        """
        X = np.asarray(X)

        if self.reference is not None:
            self.reference_spectrum_ = self.reference
        else:
            # Use mean spectrum as reference
            self.reference_spectrum_ = np.mean(X, axis=0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply MSC transformation to spectra.

        For each spectrum, fits: spectrum = a + b * reference
        Then corrects: corrected = (spectrum - a) / b

        Args:
            X: Input spectra array of shape (n_samples, n_features)

        Returns:
            Transformed spectra array
        """
        if self.reference_spectrum_ is None:
            raise ValueError("Transformer must be fitted before transform")

        X = np.asarray(X)
        X_transformed = np.zeros_like(X)

        for i in range(X.shape[0]):
            spectrum = X[i, :]

            # Fit linear regression: spectrum = a + b * reference
            coef = np.polyfit(self.reference_spectrum_, spectrum, deg=1)
            b, a = coef[0], coef[1]

            # Correct spectrum
            if abs(b) > 1e-10:  # Avoid division by zero
                corrected = (spectrum - a) / b
            else:
                corrected = spectrum - a

            X_transformed[i, :] = corrected

        return X_transformed


class DetrendTransformer(BaseEstimator, TransformerMixin):
    """
    Detrending transformer for baseline correction.

    Removes linear or polynomial baseline trends from spectra.

    Attributes:
        degree: Degree of polynomial for detrending (1=linear, 2=quadratic, etc.)

    Example:
        >>> from soilspec_pinn.preprocessing import DetrendTransformer
        >>> detrend = DetrendTransformer(degree=1)
        >>> spectra_corrected = detrend.fit_transform(spectra)
    """

    def __init__(self, degree: int = 1):
        """
        Initialize detrending transformer.

        Args:
            degree: Polynomial degree (1=linear, 2=quadratic, etc.)
        """
        self.degree = degree

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DetrendTransformer":
        """
        Fit the transformer (no-op, included for sklearn compatibility).

        Args:
            X: Input spectra array of shape (n_samples, n_features)
            y: Target values (ignored)

        Returns:
            self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply detrending to spectra.

        Args:
            X: Input spectra array of shape (n_samples, n_features)

        Returns:
            Detrended spectra array
        """
        X = np.asarray(X)
        X_transformed = np.zeros_like(X)

        n_features = X.shape[1]
        x_axis = np.arange(n_features)

        for i in range(X.shape[0]):
            spectrum = X[i, :]

            # Fit polynomial baseline
            poly_coef = np.polyfit(x_axis, spectrum, deg=self.degree)
            baseline = np.polyval(poly_coef, x_axis)

            # Remove baseline
            X_transformed[i, :] = spectrum - baseline

        return X_transformed

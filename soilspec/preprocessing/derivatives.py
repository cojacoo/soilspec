"""
Spectral derivative calculations using scipy.signal.

Wraps scipy.signal.savgol_filter for sklearn compatibility.
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class SavitzkyGolayDerivative(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay smoothing and derivative calculation.

    Wraps scipy.signal.savgol_filter with sklearn API for spectral preprocessing.
    Common in soil spectroscopy (prospectr R package, ADDRESS workflow).

    Args:
        window_length: Length of filter window (must be odd, >= polyorder + 2)
        polyorder: Order of polynomial used to fit samples
        deriv: Order of derivative (0=smoothing, 1=first derivative, 2=second)
        delta: Spacing of samples (for proper derivative scaling)

    Example:
        >>> from soilspec.preprocessing import SavitzkyGolayDerivative
        >>> from sklearn.pipeline import Pipeline
        >>>
        >>> pipeline = Pipeline([
        ...     ('snv', SNVTransformer()),
        ...     ('derivative', SavitzkyGolayDerivative(
        ...         window_length=11, polyorder=2, deriv=1
        ...     ))
        ... ])
        >>> preprocessed = pipeline.fit_transform(spectra)

    Reference:
        Savitzky, A. & Golay, M.J.E. (1964). Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical Chemistry, 36(8).
    """

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 2,
        deriv: int = 0,
        delta: float = 1.0
    ):
        """
        Initialize Savitzky-Golay filter.

        Args:
            window_length: Filter window size (must be odd)
            polyorder: Polynomial order for fitting
            deriv: Derivative order (0, 1, or 2)
            delta: Sample spacing for derivative scaling
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit transformer (no-op for Savitzky-Golay).

        Args:
            X: Spectra array (n_samples, n_features)
            y: Target values (ignored)

        Returns:
            self
        """
        # Validate parameters
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if self.window_length < self.polyorder + 2:
            raise ValueError(
                f"window_length ({self.window_length}) must be >= "
                f"polyorder + 2 ({self.polyorder + 2})"
            )
        if self.deriv < 0 or self.deriv > 2:
            raise ValueError("deriv must be 0, 1, or 2")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to spectra.

        Args:
            X: Spectra array (n_samples, n_features)

        Returns:
            Filtered/differentiated spectra
        """
        X = np.asarray(X)

        # Apply scipy's savgol_filter along wavelength axis
        X_transformed = savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=1  # Apply along features (wavelengths)
        )

        return X_transformed


class GapSegmentDerivative(BaseEstimator, TransformerMixin):
    """
    Gap-segment derivative calculation.

    Alternative to Savitzky-Golay that computes derivatives using segments
    separated by a gap. Less smoothing than Savitzky-Golay.

    Args:
        segment_size: Number of points in each segment
        gap_size: Number of points in gap between segments
        deriv: Derivative order (1 or 2)

    Example:
        >>> from soilspec.preprocessing import GapSegmentDerivative
        >>> gap_deriv = GapSegmentDerivative(segment_size=5, gap_size=10)
        >>> derivatives = gap_deriv.fit_transform(spectra)

    Reference:
        prospectr R package implementation
    """

    def __init__(
        self,
        segment_size: int = 1,
        gap_size: int = 1,
        deriv: int = 1
    ):
        """
        Initialize gap-segment derivative.

        Args:
            segment_size: Segment width
            gap_size: Gap width between segments
            deriv: Derivative order (1 or 2)
        """
        self.segment_size = segment_size
        self.gap_size = gap_size
        self.deriv = deriv

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit (no-op)."""
        if self.deriv not in [1, 2]:
            raise ValueError("deriv must be 1 or 2")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute gap-segment derivatives.

        Args:
            X: Spectra array (n_samples, n_features)

        Returns:
            Derivative spectra
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if self.deriv == 1:
            # First derivative
            step = self.gap_size + self.segment_size
            n_points = n_features - step

            if n_points <= 0:
                raise ValueError(
                    f"Spectrum too short ({n_features} points) for "
                    f"gap_size={self.gap_size} + segment_size={self.segment_size}"
                )

            X_deriv = np.zeros((n_samples, n_points))

            for i in range(n_samples):
                left_seg = X[i, :n_points]
                right_seg = X[i, step:step+n_points]

                # Average difference across segments
                X_deriv[i, :] = (right_seg - left_seg) / step

            return X_deriv

        else:  # deriv == 2
            # Second derivative (apply gap-segment twice)
            first_deriv = self.transform(X)  # Recursive call with deriv=1
            # Apply again
            temp_deriv = self.deriv
            self.deriv = 1
            second_deriv = self.transform(first_deriv)
            self.deriv = temp_deriv

            return second_deriv

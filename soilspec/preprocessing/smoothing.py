"""
Spectral smoothing using scipy and pywavelets.

Wraps proven signal processing libraries for sklearn compatibility.
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional
import warnings

# Optional wavelet support
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


class SavitzkyGolaySmoother(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay smoothing (no derivative).

    Convenience wrapper for smoothing-only (deriv=0) Savitzky-Golay filter.

    Args:
        window_length: Filter window size (must be odd)
        polyorder: Polynomial order

    Example:
        >>> from soilspec.preprocessing import SavitzkyGolaySmoother
        >>> smoother = SavitzkyGolaySmoother(window_length=11, polyorder=2)
        >>> smoothed = smoother.fit_transform(noisy_spectra)
    """

    def __init__(self, window_length: int = 11, polyorder: int = 2):
        """Initialize smoother."""
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit (no-op)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply smoothing."""
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=0,  # Smoothing only
            axis=1
        )


class WaveletDenoiser(BaseEstimator, TransformerMixin):
    """
    Wavelet-based denoising for spectral data.

    Uses discrete wavelet transform (DWT) for noise reduction.
    Common in spectroscopy when noise is non-uniform across spectrum.

    Args:
        wavelet: Wavelet family ('sym8', 'db4', 'coif4', etc.)
        level: Decomposition level (None = automatic)
        threshold: Thresholding method ('soft', 'hard', 'garrote')
        mode: Signal extension mode ('symmetric', 'periodic', 'zero', etc.)

    Example:
        >>> from soilspec.preprocessing import WaveletDenoiser
        >>> denoiser = WaveletDenoiser(wavelet='sym8', level=3)
        >>> denoised = denoiser.fit_transform(noisy_spectra)

    Reference:
        Donoho, D.L. & Johnstone, I.M. (1994). Ideal spatial adaptation by
        wavelet shrinkage. Biometrika, 81(3).
    """

    def __init__(
        self,
        wavelet: str = 'sym8',
        level: Optional[int] = None,
        threshold: str = 'soft',
        mode: str = 'symmetric'
    ):
        """
        Initialize wavelet denoiser.

        Args:
            wavelet: Wavelet name (see pywt.wavelist())
            level: Decomposition level
            threshold: 'soft', 'hard', or 'garrote'
            mode: Signal extension mode
        """
        if not PYWT_AVAILABLE:
            raise ImportError(
                "pywavelets is required for WaveletDenoiser. "
                "Install with: pip install pywavelets"
            )

        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
        self.mode = mode

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit (no-op for wavelets)."""
        # Validate wavelet
        if self.wavelet not in pywt.wavelist():
            raise ValueError(
                f"Unknown wavelet: {self.wavelet}. "
                f"Choose from: {pywt.wavelist()}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply wavelet denoising.

        Args:
            X: Spectra array (n_samples, n_features)

        Returns:
            Denoised spectra
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Determine level if not specified
        if self.level is None:
            level = pywt.dwt_max_level(n_features, self.wavelet)
        else:
            level = self.level

        X_denoised = np.zeros_like(X)

        for i in range(n_samples):
            signal = X[i, :]

            # Decompose
            coeffs = pywt.wavedec(signal, self.wavelet, level=level, mode=self.mode)

            # Estimate noise level from finest scale
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745

            # Threshold calculation (universal threshold)
            threshold = sigma * np.sqrt(2 * np.log(n_features))

            # Apply thresholding to detail coefficients
            coeffs_thresh = [coeffs[0]]  # Keep approximation
            for detail in coeffs[1:]:
                if self.threshold == 'soft':
                    thresh_detail = pywt.threshold(detail, threshold, mode='soft')
                elif self.threshold == 'hard':
                    thresh_detail = pywt.threshold(detail, threshold, mode='hard')
                elif self.threshold == 'garrote':
                    thresh_detail = pywt.threshold(detail, threshold, mode='garrote')
                else:
                    raise ValueError(f"Unknown threshold mode: {self.threshold}")

                coeffs_thresh.append(thresh_detail)

            # Reconstruct
            X_denoised[i, :] = pywt.waverec(coeffs_thresh, self.wavelet, mode=self.mode)

            # Handle any length mismatch from wavelet transform
            if len(X_denoised[i, :]) != n_features:
                X_denoised[i, :] = X_denoised[i, :n_features]

        return X_denoised


class MovingAverageSmoother(BaseEstimator, TransformerMixin):
    """
    Simple moving average smoothing.

    Fast, simple smoothing by averaging neighboring points.

    Args:
        window_size: Number of points to average (must be odd)

    Example:
        >>> from soilspec.preprocessing import MovingAverageSmoother
        >>> smoother = MovingAverageSmoother(window_size=5)
        >>> smoothed = smoother.fit_transform(spectra)
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize moving average smoother.

        Args:
            window_size: Window size (must be odd)
        """
        self.window_size = window_size

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit (no-op)."""
        if self.window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing.

        Args:
            X: Spectra array (n_samples, n_features)

        Returns:
            Smoothed spectra
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Create averaging kernel
        kernel = np.ones(self.window_size) / self.window_size

        X_smoothed = np.zeros_like(X)

        for i in range(n_samples):
            # Use numpy's convolve with 'same' mode
            X_smoothed[i, :] = np.convolve(X[i, :], kernel, mode='same')

        return X_smoothed

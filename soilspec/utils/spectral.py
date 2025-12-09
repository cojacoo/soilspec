"""
Spectral processing utilities.

Provides basic spectral processing functions that complement the preprocessing
transformers in soilspec.preprocessing.

Scientific Background
---------------------
Baseline correction is essential for removing additive background signals that
can interfere with quantitative analysis. The Asymmetric Least Squares (ALS)
method is particularly useful for baseline drift removal without distorting
peak shapes.

References
----------
.. [1] Eilers, P.H.C. & Boelens, H.F.M. (2005). Baseline correction with
       asymmetric least squares smoothing. Leiden University Medical Centre
       Report.
.. [2] Whittaker, E.T. (1922). On a new method of graduation. Proceedings of
       the Edinburgh Mathematical Society 41:63-75.
.. [3] Rinnan, Å., van den Berg, F., Engelsen, S.B. (2009). Review of the most
       common pre-processing techniques for near-infrared spectra.
       TrAC Trends in Analytical Chemistry 28(10):1201-1222.
       DOI: 10.1016/j.trac.2009.07.007
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional


def baseline_als(
    spectrum: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10
) -> np.ndarray:
    """
    Asymmetric Least Squares (ALS) baseline correction.

    Fits a smooth baseline to spectral data using penalized least squares
    with asymmetric weighting. Points below the baseline (valleys) are
    weighted more heavily than points above (peaks), allowing the baseline
    to follow the lower envelope of the spectrum.

    **Algorithm:**

    The baseline z is found by minimizing:

    .. math::

        \\sum_i w_i (y_i - z_i)^2 + \\lambda \\sum_i (\\Delta^2 z_i)^2

    where:
    * w_i are asymmetric weights (higher for y_i < z_i)
    * λ controls smoothness (higher = smoother)
    * Δ² is the second difference penalty (penalizes curvature)

    **Physical interpretation:**

    * Baseline drift: Caused by scattering, instrument drift, sample fluorescence
    * ALS removes: Additive background while preserving peak shapes
    * Asymmetry: p → 0 follows lower envelope, p → 1 follows upper envelope

    Parameters
    ----------
    spectrum : ndarray of shape (n_wavelengths,)
        Input spectrum (1D array)
    lam : float, default=1e5
        Smoothness parameter (λ). Higher values → smoother baseline.
        Typical range: 1e2 (rough) to 1e7 (very smooth)
    p : float, default=0.01
        Asymmetry parameter. Lower values → baseline follows valleys.
        Typical range: 0.001 (strict lower envelope) to 0.1 (moderate)
    niter : int, default=10
        Number of iterations. Usually converges in <10 iterations.

    Returns
    -------
    baseline : ndarray of shape (n_wavelengths,)
        Estimated baseline

    Examples
    --------
    **Basic usage:**

    >>> from soilspec.utils.spectral import baseline_als
    >>> import numpy as np
    >>>
    >>> # Generate spectrum with baseline drift
    >>> wavenumbers = np.arange(600, 4001, 2)
    >>> spectrum = np.sin(wavenumbers / 100) + 0.5 * wavenumbers / 4000
    >>>
    >>> # Remove baseline
    >>> baseline = baseline_als(spectrum, lam=1e5, p=0.01)
    >>> corrected = spectrum - baseline
    >>>
    >>> # Visualize
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(wavenumbers, spectrum, label='Original')
    >>> plt.plot(wavenumbers, baseline, label='Baseline')
    >>> plt.plot(wavenumbers, corrected, label='Corrected')
    >>> plt.legend()
    >>> plt.show()

    **Batch processing:**

    >>> # Apply to multiple spectra
    >>> X = np.random.randn(100, 1801)  # 100 spectra
    >>> X_corrected = np.array([spectrum - baseline_als(spectrum) for spectrum in X])

    **Integration with preprocessing:**

    >>> from soilspec.preprocessing import SNVTransformer
    >>> from sklearn.pipeline import Pipeline
    >>>
    >>> # Custom transformer for ALS baseline correction
    >>> class BaselineALS(BaseEstimator, TransformerMixin):
    ...     def __init__(self, lam=1e5, p=0.01):
    ...         self.lam = lam
    ...         self.p = p
    ...     def fit(self, X, y=None):
    ...         return self
    ...     def transform(self, X):
    ...         return np.array([x - baseline_als(x, self.lam, self.p) for x in X])
    >>>
    >>> pipeline = Pipeline([
    ...     ('baseline', BaselineALS()),
    ...     ('snv', SNVTransformer())
    ... ])

    Notes
    -----
    **When to use baseline correction:**

    * MIR spectra: Often needed for removing scattering effects
    * NIR/VISNIR: Less common (SNV + detrending usually sufficient)
    * After other preprocessing: Apply before normalization (SNV, MSC)

    **Parameter tuning:**

    * **lam too small** → Baseline follows peaks (overfitting)
    * **lam too large** → Baseline too smooth (underfitting)
    * **p too small** → Aggressive baseline removal (may remove real features)
    * **p too large** → Baseline stays above spectrum (ineffective)

    **Typical values:**

    * MIR soil spectra: lam=1e5-1e6, p=0.01-0.05
    * NIR spectra: lam=1e4-1e5, p=0.01
    * Noisy spectra: Higher lam for smoothness

    **Computational cost:**

    * Sparse matrix operations: O(n) per iteration
    * Typical runtime: <1ms per spectrum on modern hardware
    * Suitable for large datasets (100k+ spectra)

    References
    ----------
    See [1]_ for the original ALS algorithm, [2]_ for the Whittaker smoother,
    and [3]_ for a review of spectral preprocessing methods.

    See Also
    --------
    soilspec.preprocessing.SNVTransformer : Standard Normal Variate normalization
    soilspec.preprocessing.DetrendTransformer : Polynomial detrending
    """
    n = len(spectrum)

    # Build sparse difference matrix (D) for second derivative penalty
    # D is (n-2) × n matrix: D @ z computes second differences
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))

    # Precompute D^T @ D (constant across iterations)
    D_dot_D = D.T @ D

    # Initialize weights (all equal initially)
    w = np.ones(n)

    # Iterative reweighting
    for i in range(niter):
        # Build diagonal weight matrix
        W = sparse.diags(w, 0, shape=(n, n))

        # Solve: (W + λD^TD)z = Wy
        # This minimizes: ||W^(1/2)(y-z)||² + λ||Dz||²
        Z = W + lam * D_dot_D
        z = spsolve(Z, w * spectrum)

        # Update weights asymmetrically
        # Points below baseline: w → p (small weight, baseline can rise)
        # Points above baseline: w → 1-p (large weight, baseline must stay below)
        w = p * (spectrum > z) + (1 - p) * (spectrum < z)

    return z


def interpolate_spectrum(
    spectrum: np.ndarray,
    old_wavelengths: np.ndarray,
    new_wavelengths: np.ndarray,
    kind: str = 'linear'
) -> np.ndarray:
    """
    Interpolate spectrum to new wavelength grid.

    **Note:** For production use, prefer soilspec.preprocessing.SpectralResample
    which provides sklearn-compatible interface with fit/transform methods.

    This function is provided for quick one-off interpolations.

    Parameters
    ----------
    spectrum : ndarray of shape (n_old_wavelengths,)
        Input spectrum values
    old_wavelengths : ndarray of shape (n_old_wavelengths,)
        Original wavelength grid
    new_wavelengths : ndarray of shape (n_new_wavelengths,)
        Target wavelength grid
    kind : str, default='linear'
        Interpolation method: 'linear', 'cubic', 'nearest'

    Returns
    -------
    interpolated : ndarray of shape (n_new_wavelengths,)
        Interpolated spectrum

    Examples
    --------
    >>> from soilspec.utils.spectral import interpolate_spectrum
    >>> import numpy as np
    >>>
    >>> # Original spectrum: 600-4000 cm⁻¹, 2 cm⁻¹ resolution
    >>> old_wn = np.arange(600, 4001, 2)
    >>> spectrum = np.random.randn(len(old_wn))
    >>>
    >>> # Resample to 4 cm⁻¹ resolution
    >>> new_wn = np.arange(600, 4001, 4)
    >>> resampled = interpolate_spectrum(spectrum, old_wn, new_wn)

    Notes
    -----
    **For batch processing, use SpectralResample instead:**

    >>> from soilspec.preprocessing import SpectralResample
    >>> resampler = SpectralResample(wavenumbers_new=new_wn)
    >>> X_resampled = resampler.fit_transform(X)  # Works on (n_samples, n_wavelengths)

    See Also
    --------
    soilspec.preprocessing.SpectralResample : Sklearn-compatible resampling
    """
    from scipy.interpolate import interp1d

    # Validate inputs
    if len(spectrum) != len(old_wavelengths):
        raise ValueError(
            f"Spectrum length ({len(spectrum)}) must match old wavelength grid "
            f"({len(old_wavelengths)})"
        )

    # Check for extrapolation
    if new_wavelengths.min() < old_wavelengths.min() or new_wavelengths.max() > old_wavelengths.max():
        raise ValueError(
            f"New wavelengths [{new_wavelengths.min()}, {new_wavelengths.max()}] "
            f"outside old range [{old_wavelengths.min()}, {old_wavelengths.max()}]. "
            f"Extrapolation not supported."
        )

    # Interpolate
    interp_func = interp1d(old_wavelengths, spectrum, kind=kind)
    return interp_func(new_wavelengths)

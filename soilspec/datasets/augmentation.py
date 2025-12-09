"""
Spectral data augmentation for improving model robustness.

Augmentation techniques for soil spectra to:
* Increase effective training set size
* Improve model robustness to instrument variations
* Reduce overfitting

Scientific Background
---------------------
Spectral augmentation simulates realistic variations in spectral measurements
due to instrument noise, baseline shifts, and multiplicative effects.

**Important:** Only augment training data, never test data!

References
----------
.. [1] Padarian, J., et al. (2019). Using deep learning for digital soil
       mapping. Soil 5(1):79-89.
.. [2] Tsakiridis, N.L., et al. (2020). Simultaneous prediction of soil
       properties from VNIR-SWIR spectra using CNN. Geoderma 367:114208.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin


class SpectralAugmenter(BaseEstimator, TransformerMixin):
    """
    Apply multiple augmentation techniques to spectral data.

    Combines noise addition, baseline shifts, and multiplicative scaling
    to simulate realistic spectral variations.

    **Augmentation Types:**

    1. **Additive noise**: Gaussian noise → instrument detector noise
    2. **Baseline shift**: Linear or constant → sample positioning
    3. **Multiplicative scaling**: → particle size, packing density
    4. **Wavelength shift**: → instrument calibration drift

    Parameters
    ----------
    noise_level : float, default=0.01
        Standard deviation of Gaussian noise (as fraction of signal range)
    baseline_shift : float, default=0.0
        Maximum baseline shift (as fraction of signal range)
    scaling_range : tuple, default=(0.95, 1.05)
        Range for multiplicative scaling (min_scale, max_scale)
    wavelength_shift : int, default=0
        Maximum shift in wavelength indices (for simulating calibration drift)
    p : float, default=0.5
        Probability of applying each augmentation
    random_state : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> from soilspec.datasets import SpectralAugmenter
    >>> import numpy as np
    >>>
    >>> # Create augmenter
    >>> aug = SpectralAugmenter(
    >>>     noise_level=0.01,
    >>>     baseline_shift=0.02,
    >>>     scaling_range=(0.95, 1.05),
    >>>     p=0.5
    >>> )
    >>>
    >>> # Original spectrum
    >>> X_original = np.random.randn(100, 1801)
    >>>
    >>> # Augmented spectra (2x original size)
    >>> X_augmented = aug.fit_transform(X_original)
    >>> print(f"Original: {X_original.shape}, Augmented: {X_augmented.shape}")
    >>>
    >>> # Use in training
    >>> from sklearn.pipeline import Pipeline
    >>> from soilspec.models.traditional import MBLRegressor
    >>>
    >>> pipeline = Pipeline([
    >>>     ('augment', SpectralAugmenter(p=0.3)),  # 30% augmentation
    >>>     ('model', MBLRegressor())
    >>> ])
    >>> pipeline.fit(X_train, y_train)

    Notes
    -----
    **Best Practices:**

    * Start with conservative augmentation (noise_level=0.005-0.01)
    * Use cross-validation to tune augmentation strength
    * More augmentation needed for small datasets (n < 500)
    * Less augmentation for large datasets (n > 5000)
    * Always validate that augmented data looks realistic
    """

    def __init__(
        self,
        noise_level: float = 0.01,
        baseline_shift: float = 0.0,
        scaling_range: Tuple[float, float] = (0.95, 1.05),
        wavelength_shift: int = 0,
        p: float = 0.5,
        random_state: Optional[int] = None
    ):
        self.noise_level = noise_level
        self.baseline_shift = baseline_shift
        self.scaling_range = scaling_range
        self.wavelength_shift = wavelength_shift
        self.p = p
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit augmenter (no-op, but required for sklearn compatibility)."""
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_wavelengths)
            Input spectra

        Returns
        -------
        X_aug : ndarray of shape (n_samples, n_wavelengths)
            Augmented spectra
        """
        X_aug = X.copy()

        # Add noise
        if self.noise_level > 0 and self.rng_.rand() < self.p:
            X_aug = add_noise(X_aug, self.noise_level, rng=self.rng_)

        # Baseline shift
        if self.baseline_shift > 0 and self.rng_.rand() < self.p:
            X_aug = shift_baseline(X_aug, self.baseline_shift, rng=self.rng_)

        # Multiplicative scaling
        if self.scaling_range != (1.0, 1.0) and self.rng_.rand() < self.p:
            scale = self.rng_.uniform(*self.scaling_range, size=(X_aug.shape[0], 1))
            X_aug = X_aug * scale

        # Wavelength shift (circular shift)
        if self.wavelength_shift > 0 and self.rng_.rand() < self.p:
            shift = self.rng_.randint(-self.wavelength_shift, self.wavelength_shift + 1)
            X_aug = np.roll(X_aug, shift, axis=1)

        return X_aug

    def __repr__(self):
        return (
            f"SpectralAugmenter(noise_level={self.noise_level}, "
            f"baseline_shift={self.baseline_shift}, p={self.p})"
        )


def add_noise(
    X: np.ndarray,
    noise_level: float = 0.01,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Add Gaussian noise to spectra.

    Simulates instrument detector noise.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_wavelengths)
        Input spectra
    noise_level : float, default=0.01
        Noise standard deviation as fraction of signal range
    rng : RandomState, optional
        Random number generator

    Returns
    -------
    X_noisy : ndarray
        Spectra with added noise

    Examples
    --------
    >>> from soilspec.datasets.augmentation import add_noise
    >>> X_noisy = add_noise(X, noise_level=0.01)
    """
    if rng is None:
        rng = np.random.RandomState()

    # Estimate signal range
    signal_range = np.ptp(X, axis=1, keepdims=True)

    # Add noise proportional to signal
    noise = rng.randn(*X.shape) * (noise_level * signal_range)

    return X + noise


def shift_baseline(
    X: np.ndarray,
    max_shift: float = 0.02,
    mode: str = 'constant',
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Apply baseline shift to spectra.

    Simulates sample positioning and path length variations.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_wavelengths)
        Input spectra
    max_shift : float, default=0.02
        Maximum shift as fraction of signal range
    mode : str, default='constant'
        'constant' for constant shift, 'linear' for linear drift
    rng : RandomState, optional
        Random number generator

    Returns
    -------
    X_shifted : ndarray
        Spectra with baseline shift

    Examples
    --------
    >>> from soilspec.datasets.augmentation import shift_baseline
    >>>
    >>> # Constant baseline shift
    >>> X_shifted = shift_baseline(X, max_shift=0.02, mode='constant')
    >>>
    >>> # Linear baseline drift
    >>> X_shifted = shift_baseline(X, max_shift=0.02, mode='linear')
    """
    if rng is None:
        rng = np.random.RandomState()

    X_shifted = X.copy()
    signal_range = np.ptp(X, axis=1, keepdims=True)

    for i in range(X.shape[0]):
        shift_amount = rng.uniform(-max_shift, max_shift) * signal_range[i]

        if mode == 'constant':
            # Constant offset
            X_shifted[i] += shift_amount

        elif mode == 'linear':
            # Linear drift across wavelengths
            drift = np.linspace(0, shift_amount, X.shape[1])
            X_shifted[i] += drift

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return X_shifted


def scale_spectrum(
    X: np.ndarray,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Apply multiplicative scaling to spectra.

    Simulates particle size and packing density variations.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_wavelengths)
        Input spectra
    scale_range : tuple, default=(0.95, 1.05)
        Range for scaling factor (min_scale, max_scale)
    rng : RandomState, optional
        Random number generator

    Returns
    -------
    X_scaled : ndarray
        Scaled spectra

    Examples
    --------
    >>> from soilspec.datasets.augmentation import scale_spectrum
    >>> X_scaled = scale_spectrum(X, scale_range=(0.9, 1.1))
    """
    if rng is None:
        rng = np.random.RandomState()

    # Random scale for each sample
    scales = rng.uniform(*scale_range, size=(X.shape[0], 1))

    return X * scales


def wavelength_shift_augment(
    X: np.ndarray,
    max_shift: int = 2,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Apply wavelength shift augmentation.

    Simulates small calibration drifts in instrument.

    **Warning:** This is a circular shift - wavelengths wrap around!
    Use sparingly and with small shifts (1-3 indices).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_wavelengths)
        Input spectra
    max_shift : int, default=2
        Maximum shift in number of wavelength indices
    rng : RandomState, optional
        Random number generator

    Returns
    -------
    X_shifted : ndarray
        Wavelength-shifted spectra

    Examples
    --------
    >>> from soilspec.datasets.augmentation import wavelength_shift_augment
    >>> X_shifted = wavelength_shift_augment(X, max_shift=2)
    """
    if rng is None:
        rng = np.random.RandomState()

    X_shifted = X.copy()

    for i in range(X.shape[0]):
        shift = rng.randint(-max_shift, max_shift + 1)
        X_shifted[i] = np.roll(X[i], shift)

    return X_shifted


def mixup_spectra(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mixup augmentation to spectral data.

    Mixup creates virtual training examples by linearly interpolating
    between pairs of spectra and their targets.

    .. math::

        \\tilde{X} = \\lambda X_i + (1 - \\lambda) X_j
        \\tilde{y} = \\lambda y_i + (1 - \\lambda) y_j

    where :math:`\\lambda \\sim \\text{Beta}(\\alpha, \\alpha)`.

    **Warning:** Mixup assumes targets are continuous and can be interpolated.
    Not suitable for classification unless using soft labels.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_wavelengths)
        Input spectra
    y : ndarray of shape (n_samples,)
        Target values
    alpha : float, default=0.2
        Beta distribution parameter (higher = more mixing)
    rng : RandomState, optional
        Random number generator

    Returns
    -------
    X_mixed : ndarray
        Mixed spectra
    y_mixed : ndarray
        Mixed targets

    Examples
    --------
    >>> from soilspec.datasets.augmentation import mixup_spectra
    >>>
    >>> # Generate mixup examples
    >>> X_mixed, y_mixed = mixup_spectra(X_train, y_train, alpha=0.2)
    >>>
    >>> # Combine with original
    >>> X_combined = np.vstack([X_train, X_mixed])
    >>> y_combined = np.concatenate([y_train, y_mixed])

    References
    ----------
    Zhang, H., et al. (2018). mixup: Beyond empirical risk minimization.
    ICLR 2018.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_samples = X.shape[0]

    # Sample mixing coefficients
    lam = rng.beta(alpha, alpha, size=n_samples)

    # Random permutation for pairs
    indices = rng.permutation(n_samples)

    # Mix spectra and targets
    X_mixed = lam[:, None] * X + (1 - lam[:, None]) * X[indices]
    y_mixed = lam * y + (1 - lam) * y[indices]

    return X_mixed, y_mixed

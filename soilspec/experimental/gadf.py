"""
EXPERIMENTAL: Gramian Angular Difference Field (GADF) transformation.

Converts 1D spectra to 2D images for use with 2D CNNs and transfer learning.

**WARNING:** This is an experimental feature. The scientific justification
for GADF + transfer learning from ImageNet to soil spectra is questionable.

**Recommended approach:** Use 1D CNNs (soilspec.models.deep_learning.SimpleCNN1D)
which directly process spectral data without artificial 2D transformation.

Scientific Background
---------------------
GADF (Gramian Angular Difference Field) encodes 1D time series or spectra
as 2D images by computing angular relationships between values.

**Original use case:** Time series classification (Wang & Oates 2015)

**Soil spectroscopy adaptation:** Albinet et al. (2023) used GADF to convert
soil spectra to images, enabling transfer learning from ImageNet-pretrained
ResNet/ViT models.

**Advantages:**
* Can leverage pre-trained computer vision models
* Novel approach for academic papers
* May capture relationships between wavelengths

**Disadvantages:**
* Questionable relevance: What do cat/dog images have to do with molecular vibrations?
* Adds complexity and failure points
* 1D→2D transformation creates artificial structure
* More hyperparameters to tune
* Harder to interpret
* No strong evidence of superiority over 1D CNNs

**When to consider:**
* Very large datasets (>100k samples) where transfer learning helps
* Academic exploration and comparison studies
* When you have tried 1D CNNs and traditional models first

References
----------
.. [1] Wang, Z. & Oates, T. (2015). Encoding time series as images for
       visual inspection and classification using tiled convolutional neural
       networks. AAAI Workshop on Learning Rich Representations.
.. [2] Albinet, F., et al. (2023). Prediction of exchangeable potassium in
       soil through mid-infrared spectroscopy and deep learning.
       https://github.com/franckalbinet/lssm
"""

import numpy as np
from typing import Optional, Literal, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import warnings


class GADFTransformer(BaseEstimator, TransformerMixin):
    """
    Transform 1D spectra to 2D images using Gramian Angular Difference Field.

    **WARNING:** Experimental feature. Not recommended as primary approach.

    **Algorithm:**

    1. Rescale spectrum to [-1, 1] or [0, 1]
    2. Compute polar coordinates (angle, radius)
    3. Compute Gramian matrix: G[i,j] = sin(θ_i - θ_j)
    4. Result is 2D symmetric matrix

    Mathematically:

    .. math::

        \\tilde{x}_i = \\frac{x_i - \\min(x)}{\\max(x) - \\min(x)} \\in [0, 1]

        \\phi_i = \\arccos(\\tilde{x}_i) \\quad \\text{or} \\quad \\arccos(2\\tilde{x}_i - 1)

        G_{ij} = \\sin(\\phi_i - \\phi_j)

    **Output:**

    * Input: (n_samples, n_wavelengths)
    * Output: (n_samples, n_wavelengths, n_wavelengths)

    For MIR with 1801 wavelengths, creates 1801×1801 images (large!).
    Typically resized to 224×224 for pre-trained CNNs.

    Parameters
    ----------
    rescale_range : {'[-1,1]', '[0,1]'}, default='[-1,1]'
        Range for rescaling spectra before angular encoding
    output_size : int, optional
        If provided, resize GADF matrix to (output_size, output_size)
        using bilinear interpolation. Common: 224 for ImageNet models.

    Attributes
    ----------
    fitted_ : bool
        Whether transformer has been fitted

    Examples
    --------
    >>> from soilspec.experimental import GADFTransformer
    >>> import numpy as np
    >>>
    >>> # Transform spectra to 2D images
    >>> X = np.random.randn(100, 1801)  # 100 spectra, 1801 wavelengths
    >>>
    >>> gadf = GADFTransformer(output_size=224)
    >>> X_2d = gadf.fit_transform(X)
    >>> print(X_2d.shape)  # (100, 224, 224)
    >>>
    >>> # Use with 2D CNN or pre-trained model
    >>> import torch
    >>> import timm
    >>>
    >>> # Create ResNet18 for 1-channel input, 1 output
    >>> model = timm.create_model('resnet18', pretrained=True,
    >>>                           in_chans=1, num_classes=1)
    >>>
    >>> # Forward pass (add channel dimension)
    >>> X_torch = torch.FloatTensor(X_2d).unsqueeze(1)  # (100, 1, 224, 224)
    >>> y_pred = model(X_torch)

    Notes
    -----
    **Critical considerations:**

    1. **Computational cost:** For 1801 wavelengths, creates 1801²=3.2M values
       per spectrum. With 100k samples, that's 320 billion values!

    2. **Memory:** 1801×1801 float32 = 13 MB per spectrum.
       1000 spectra = 13 GB memory.

    3. **Typical usage:** Must resize to ~224×224 for pre-trained models,
       losing much of the spectral resolution.

    4. **Performance:** No strong evidence that GADF + ImageNet transfer
       outperforms 1D CNNs trained from scratch on soil spectra.

    **Recommendation:** Try 1D CNNs first. Only use GADF if you have:
    * Very large datasets (>50k samples)
    * Computational resources for 2D processing
    * Specific need for transfer learning experiments

    References
    ----------
    See [1]_ for original GADF method and [2]_ for soil spectroscopy application.
    """

    def __init__(
        self,
        rescale_range: Literal['[-1,1]', '[0,1]'] = '[-1,1]',
        output_size: Optional[int] = None
    ):
        self.rescale_range = rescale_range
        self.output_size = output_size

        warnings.warn(
            "GADFTransformer is experimental. The scientific justification for "
            "GADF + ImageNet transfer learning for soil spectra is questionable. "
            "Consider using 1D CNNs (soilspec.models.deep_learning.SimpleCNN1D) "
            "which directly process spectral data without artificial 2D transformation.",
            UserWarning,
            stacklevel=2
        )

    def fit(self, X, y=None):
        """
        Fit transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_wavelengths)
            Spectra
        y : ignored

        Returns
        -------
        self
        """
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform spectra to GADF matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_wavelengths)
            Input spectra

        Returns
        -------
        X_gadf : ndarray of shape (n_samples, H, W)
            GADF matrices. If output_size is None: H=W=n_wavelengths
            If output_size specified: H=W=output_size
        """
        n_samples, n_wavelengths = X.shape

        # Rescale to [0, 1]
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        X_rescaled = (X - X_min) / (X_max - X_min + 1e-8)

        # Map to [-1, 1] if requested
        if self.rescale_range == '[-1,1]':
            X_rescaled = 2 * X_rescaled - 1

        # Compute angles
        # Clip to avoid numerical issues with arccos
        X_clipped = np.clip(X_rescaled, -1 + 1e-8, 1 - 1e-8)
        angles = np.arccos(X_clipped)  # (n_samples, n_wavelengths)

        # Compute GADF matrices
        # G[i,j] = sin(angle[i] - angle[j])
        X_gadf = np.zeros((n_samples, n_wavelengths, n_wavelengths))

        for i in range(n_samples):
            # Broadcast angles to compute all pairwise differences
            angle_diff = angles[i:i+1, :, None] - angles[i:i+1, None, :]
            X_gadf[i] = np.sin(angle_diff[0])

        # Resize if requested
        if self.output_size is not None:
            X_gadf = self._resize_images(X_gadf, self.output_size)

        return X_gadf

    def _resize_images(
        self,
        images: np.ndarray,
        output_size: int
    ) -> np.ndarray:
        """
        Resize GADF matrices using bilinear interpolation.

        Parameters
        ----------
        images : ndarray of shape (n_samples, H, W)
            Input images
        output_size : int
            Target size (output will be output_size × output_size)

        Returns
        -------
        resized : ndarray of shape (n_samples, output_size, output_size)
            Resized images
        """
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError(
                "scipy required for image resizing. Install with: pip install scipy"
            )

        n_samples, H, W = images.shape
        zoom_factors = (1, output_size / H, output_size / W)

        resized = zoom(images, zoom_factors, order=1)  # Bilinear

        return resized

    def __repr__(self):
        return (
            f"GADFTransformer(rescale_range='{self.rescale_range}', "
            f"output_size={self.output_size})"
        )


def spectrum_to_gadf(
    spectrum: np.ndarray,
    rescale_range: str = '[-1,1]'
) -> np.ndarray:
    """
    Convert single spectrum to GADF matrix.

    Convenience function for visualizing GADF transformation.

    Parameters
    ----------
    spectrum : ndarray of shape (n_wavelengths,)
        Single spectrum
    rescale_range : str, default='[-1,1]'
        Rescaling range

    Returns
    -------
    gadf : ndarray of shape (n_wavelengths, n_wavelengths)
        GADF matrix

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from soilspec.experimental import spectrum_to_gadf
    >>>
    >>> # Generate sample spectrum
    >>> wavenumbers = np.arange(600, 4001, 2)
    >>> spectrum = np.random.randn(len(wavenumbers))
    >>>
    >>> # Convert to GADF
    >>> gadf = spectrum_to_gadf(spectrum)
    >>>
    >>> # Visualize
    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    >>> ax1.plot(wavenumbers, spectrum)
    >>> ax1.set_title('Original Spectrum')
    >>> ax1.set_xlabel('Wavenumber (cm⁻¹)')
    >>>
    >>> ax2.imshow(gadf, cmap='viridis')
    >>> ax2.set_title('GADF Image')
    >>> plt.show()
    """
    transformer = GADFTransformer(rescale_range=rescale_range)
    gadf = transformer.fit_transform(spectrum.reshape(1, -1))
    return gadf[0]


def compare_gadf_1dcnn(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> dict:
    """
    Compare GADF + 2D CNN vs 1D CNN approaches.

    **Benchmark function** to empirically test whether GADF transformation
    provides benefits over direct 1D processing.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_wavelengths)
        Spectra
    y : ndarray of shape (n_samples,)
        Target values
    test_size : float, default=0.2
        Fraction for test set
    random_state : int, optional
        Random seed

    Returns
    -------
    results : dict
        Comparison results with keys:
        * 'gadf_r2', 'gadf_rmse': GADF + 2D CNN performance
        * '1dcnn_r2', '1dcnn_rmse': 1D CNN performance
        * 'training_time_gadf', 'training_time_1dcnn': Training times
        * 'winner': Which approach performed better

    Examples
    --------
    >>> from soilspec.experimental import compare_gadf_1dcnn
    >>> from soilspec.datasets import OSSLDataset
    >>>
    >>> # Load data
    >>> ossl = OSSLDataset()
    >>> X, y, ids = ossl.load_mir(target='soc')
    >>>
    >>> # Compare approaches
    >>> results = compare_gadf_1dcnn(X, y, random_state=42)
    >>> print(f"1D CNN R²: {results['1dcnn_r2']:.3f}")
    >>> print(f"GADF R²: {results['gadf_r2']:.3f}")
    >>> print(f"Winner: {results['winner']}")

    Notes
    -----
    This function requires PyTorch and significant computational resources.
    Recommended to run on GPU with >10k samples for meaningful comparison.
    """
    raise NotImplementedError(
        "Benchmark function not yet implemented. "
        "This would require full training of both approaches, which is "
        "computationally expensive. Use soilspec.training.DeepLearningTrainer "
        "to train models manually and compare."
    )

"""
Visualization utilities for soil spectroscopy.

Provides convenient plotting functions for spectra, predictions, and
preprocessing effects.

**Note:** For model evaluation plots, use:
* soilspec.training.GenericTrainer.plot_predictions()
* soilspec.training.GenericTrainer.plot_residuals()

These utilities are for quick exploratory visualization.

References
----------
.. [1] Reeves III, J.B. (2010). Near- versus mid-infrared diffuse reflectance
       spectroscopy for soil analysis emphasizing carbon and laboratory versus
       on-site analysis. Geoderma 158:3-14.
.. [2] Soriano-Disla, J.M., et al. (2014). The performance of visible, near-,
       and mid-infrared reflectance spectroscopy for prediction of soil
       physical, chemical, and biological properties. Applied Spectroscopy
       Reviews 49(2):139-186.
"""

import numpy as np
from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_spectrum(
    spectrum: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Absorbance",
    spectral_range: str = "mir",
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot a single spectrum with appropriate axis labels.

    Convenience function for quick spectrum visualization with sensible
    defaults for different spectral ranges (MIR, NIR, VISNIR).

    Parameters
    ----------
    spectrum : ndarray of shape (n_wavelengths,)
        Spectral values to plot
    wavelengths : ndarray of shape (n_wavelengths,), optional
        Wavelength/wavenumber values. If None, uses indices.
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label. If None, inferred from spectral_range.
    ylabel : str, default="Absorbance"
        Y-axis label
    spectral_range : {'mir', 'nir', 'visnir'}, default='mir'
        Spectral range for axis label inference
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to plt.plot()

    Returns
    -------
    fig : matplotlib Figure
        Figure object
    ax : matplotlib Axes
        Axes object

    Examples
    --------
    **Basic usage:**

    >>> from soilspec.utils.visualization import plot_spectrum
    >>> import numpy as np
    >>>
    >>> # Generate sample MIR spectrum
    >>> wavenumbers = np.arange(600, 4001, 2)
    >>> spectrum = np.random.randn(len(wavenumbers))
    >>>
    >>> # Plot
    >>> fig, ax = plot_spectrum(spectrum, wavenumbers, title="Soil Sample #42")
    >>> plt.show()

    **Compare original vs preprocessed:**

    >>> from soilspec.preprocessing import SNVTransformer
    >>> from soilspec.utils.visualization import plot_spectra_comparison
    >>>
    >>> snv = SNVTransformer()
    >>> spectrum_snv = snv.fit_transform(spectrum.reshape(1, -1))[0]
    >>>
    >>> plot_spectra_comparison(
    ...     [spectrum, spectrum_snv],
    ...     wavelengths,
    ...     labels=["Raw", "SNV"],
    ...     title="Preprocessing Effect"
    ... )

    **Multiple spectra on same plot:**

    >>> fig, ax = plt.subplots(figsize=(10, 4))
    >>> for i, spec in enumerate(X[:5]):  # Plot first 5 spectra
    ...     plot_spectrum(spec, wavenumbers, ax=ax, alpha=0.5, label=f"Sample {i+1}")
    >>> ax.legend()
    >>> plt.show()

    **NIR spectrum:**

    >>> # NIR uses wavelength (nm) instead of wavenumber (cm⁻¹)
    >>> wavelengths_nm = np.arange(1000, 2501)
    >>> spectrum_nir = np.random.randn(len(wavelengths_nm))
    >>>
    >>> plot_spectrum(
    ...     spectrum_nir,
    ...     wavelengths_nm,
    ...     spectral_range="nir",
    ...     title="NIR Spectrum"
    ... )

    Notes
    -----
    **Spectral conventions:**

    * **MIR**: Wavenumber (cm⁻¹), decreasing x-axis (4000 → 600)
    * **NIR/VISNIR**: Wavelength (nm), increasing x-axis (350 → 2500)
    * **Absorbance**: -log₁₀(R) where R is reflectance
    * **Reflectance**: Often used for NIR (0-1 or 0-100%)

    **For publication-quality plots:**

    >>> fig, ax = plot_spectrum(spectrum, wavenumbers)
    >>> ax.set_ylim(0, 2)
    >>> ax.grid(alpha=0.3)
    >>> fig.savefig("spectrum.pdf", dpi=300, bbox_inches='tight')

    See Also
    --------
    plot_spectra_comparison : Compare multiple spectra
    soilspec.training.GenericTrainer.plot_predictions : Model performance plots
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    # Use indices if wavelengths not provided
    if wavelengths is None:
        wavelengths = np.arange(len(spectrum))

    # Validate shapes
    if len(spectrum) != len(wavelengths):
        raise ValueError(
            f"Spectrum length ({len(spectrum)}) must match wavelengths ({len(wavelengths)})"
        )

    # Plot spectrum
    ax.plot(wavelengths, spectrum, **kwargs)

    # Set axis labels based on spectral range
    if xlabel is None:
        if spectral_range == "mir":
            xlabel = "Wavenumber (cm⁻¹)"
        elif spectral_range in ["nir", "visnir"]:
            xlabel = "Wavelength (nm)"
        else:
            xlabel = "Wavelength/Wavenumber"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Reverse x-axis for MIR (convention: high to low wavenumber)
    if spectral_range == "mir" and wavelengths[0] < wavelengths[-1]:
        ax.invert_xaxis()

    # Set title
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_spectra_comparison(
    spectra: List[np.ndarray],
    wavelengths: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "Spectrum Comparison",
    xlabel: Optional[str] = None,
    ylabel: str = "Absorbance",
    spectral_range: str = "mir",
    colors: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot multiple spectra for comparison.

    Useful for visualizing preprocessing effects, comparing samples,
    or showing ensemble predictions.

    Parameters
    ----------
    spectra : list of ndarray
        List of spectra to plot. Each should be shape (n_wavelengths,)
    wavelengths : ndarray, optional
        Wavelength/wavenumber values
    labels : list of str, optional
        Labels for each spectrum (for legend)
    title : str, default="Spectrum Comparison"
        Plot title
    xlabel : str, optional
        X-axis label. If None, inferred from spectral_range.
    ylabel : str, default="Absorbance"
        Y-axis label
    spectral_range : {'mir', 'nir', 'visnir'}, default='mir'
        Spectral range
    colors : list of str, optional
        Colors for each spectrum. If None, uses default cycle.
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments passed to plt.plot()

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

    Examples
    --------
    **Compare preprocessing methods:**

    >>> from soilspec.preprocessing import SNVTransformer, SavitzkyGolayTransformer
    >>> from soilspec.utils.visualization import plot_spectra_comparison
    >>>
    >>> # Apply different preprocessing
    >>> spectrum_raw = X[0]
    >>> spectrum_snv = SNVTransformer().fit_transform(X[0:1])[0]
    >>> spectrum_sg = SavitzkyGolayTransformer(deriv=1).fit_transform(X[0:1])[0]
    >>>
    >>> plot_spectra_comparison(
    ...     [spectrum_raw, spectrum_snv, spectrum_sg],
    ...     wavenumbers,
    ...     labels=["Raw", "SNV", "1st Derivative"],
    ...     title="Preprocessing Comparison"
    ... )

    **Visualize sample variability:**

    >>> # Plot 10 random samples
    >>> indices = np.random.choice(len(X), 10, replace=False)
    >>> plot_spectra_comparison(
    ...     [X[i] for i in indices],
    ...     wavenumbers,
    ...     title="Sample Variability",
    ...     alpha=0.5
    ... )

    **Before/after baseline correction:**

    >>> from soilspec.utils.spectral import baseline_als
    >>>
    >>> baseline = baseline_als(spectrum)
    >>> corrected = spectrum - baseline
    >>>
    >>> plot_spectra_comparison(
    ...     [spectrum, baseline, corrected],
    ...     wavenumbers,
    ...     labels=["Original", "Baseline", "Corrected"],
    ...     colors=["blue", "red", "green"]
    ... )

    See Also
    --------
    plot_spectrum : Plot single spectrum
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    # Use indices if wavelengths not provided
    if wavelengths is None:
        wavelengths = np.arange(len(spectra[0]))

    # Generate labels if not provided
    if labels is None:
        labels = [f"Spectrum {i+1}" for i in range(len(spectra))]

    # Use default colors if not provided
    if colors is None:
        colors = [None] * len(spectra)

    # Plot each spectrum
    for spectrum, label, color in zip(spectra, labels, colors):
        plot_kwargs = {**kwargs}
        if color is not None:
            plot_kwargs['color'] = color
        ax.plot(wavelengths, spectrum, label=label, **plot_kwargs)

    # Set axis labels based on spectral range
    if xlabel is None:
        if spectral_range == "mir":
            xlabel = "Wavenumber (cm⁻¹)"
        elif spectral_range in ["nir", "visnir"]:
            xlabel = "Wavelength (nm)"
        else:
            xlabel = "Wavelength/Wavenumber"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Reverse x-axis for MIR
    if spectral_range == "mir" and wavelengths[0] < wavelengths[-1]:
        ax.invert_xaxis()

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return fig, ax


def plot_mean_spectrum(
    X: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    show_std: bool = True,
    title: str = "Mean Spectrum",
    xlabel: Optional[str] = None,
    ylabel: str = "Absorbance",
    spectral_range: str = "mir",
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot mean spectrum with optional standard deviation envelope.

    Useful for visualizing average spectral characteristics of a dataset
    and assessing variability.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_wavelengths)
        Spectral matrix
    wavelengths : ndarray, optional
        Wavelength/wavenumber values
    show_std : bool, default=True
        Whether to show ±1 std dev envelope
    title : str
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str
        Y-axis label
    spectral_range : str
        Spectral range
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments for plt.plot()

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

    Examples
    --------
    >>> from soilspec.datasets import OSSLDataset
    >>> from soilspec.utils.visualization import plot_mean_spectrum
    >>>
    >>> ossl = OSSLDataset()
    >>> X, y, ids = ossl.load_mir(target='soc')
    >>>
    >>> # Plot mean MIR spectrum of all OSSL samples
    >>> wavenumbers = np.arange(600, 4001, 2)
    >>> plot_mean_spectrum(X, wavenumbers, title="OSSL Mean MIR Spectrum")

    **Compare subsets:**

    >>> # High vs low SOC
    >>> X_high = X[y > np.median(y)]
    >>> X_low = X[y <= np.median(y)]
    >>>
    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    >>> plot_mean_spectrum(X_high, wavenumbers, ax=ax1, title="High SOC")
    >>> plot_mean_spectrum(X_low, wavenumbers, ax=ax2, title="Low SOC")

    See Also
    --------
    plot_spectrum : Single spectrum
    plot_spectra_comparison : Multiple spectra
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    # Use indices if wavelengths not provided
    if wavelengths is None:
        wavelengths = np.arange(X.shape[1])

    # Compute mean and std
    mean_spectrum = X.mean(axis=0)
    std_spectrum = X.std(axis=0)

    # Plot mean
    ax.plot(wavelengths, mean_spectrum, label=f"Mean (n={len(X)})", **kwargs)

    # Plot std envelope
    if show_std:
        ax.fill_between(
            wavelengths,
            mean_spectrum - std_spectrum,
            mean_spectrum + std_spectrum,
            alpha=0.3,
            label="±1 std dev"
        )

    # Set axis labels
    if xlabel is None:
        if spectral_range == "mir":
            xlabel = "Wavenumber (cm⁻¹)"
        elif spectral_range in ["nir", "visnir"]:
            xlabel = "Wavelength (nm)"
        else:
            xlabel = "Wavelength/Wavenumber"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Reverse x-axis for MIR
    if spectral_range == "mir" and wavelengths[0] < wavelengths[-1]:
        ax.invert_xaxis()

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return fig, ax

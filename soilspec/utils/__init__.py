"""
Utility functions for spectral processing and analysis.

Provides convenience functions for spectral processing, baseline correction,
and quick visualization. For production workflows, prefer:

* Preprocessing: soilspec.preprocessing (SNV, SG, detrending, etc.)
* Validation: sklearn.model_selection (train_test_split, cross_val_score)
* Model plots: soilspec.training.GenericTrainer.plot_predictions()

Modules
-------
spectral
    baseline_als, interpolate_spectrum

visualization
    plot_spectrum, plot_spectra_comparison, plot_mean_spectrum

Examples
--------
**Baseline correction:**

>>> from soilspec.utils.spectral import baseline_als
>>> baseline = baseline_als(spectrum, lam=1e5, p=0.01)
>>> corrected = spectrum - baseline

**Quick visualization:**

>>> from soilspec.utils.visualization import plot_spectrum
>>> fig, ax = plot_spectrum(spectrum, wavenumbers, title="Soil Sample #42")

**Compare preprocessing methods:**

>>> from soilspec.utils.visualization import plot_spectra_comparison
>>> from soilspec.preprocessing import SNVTransformer
>>> spectrum_snv = SNVTransformer().fit_transform(spectrum.reshape(1, -1))[0]
>>> plot_spectra_comparison([spectrum, spectrum_snv], wavenumbers,
...                          labels=["Raw", "SNV"])
"""

from soilspec.utils.spectral import (
    baseline_als,
    interpolate_spectrum,
)
from soilspec.utils.visualization import (
    plot_spectrum,
    plot_spectra_comparison,
    plot_mean_spectrum,
)

__all__ = [
    # Spectral processing
    "baseline_als",
    "interpolate_spectrum",
    # Visualization
    "plot_spectrum",
    "plot_spectra_comparison",
    "plot_mean_spectrum",
]

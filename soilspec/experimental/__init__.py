"""
EXPERIMENTAL features for soil spectroscopy.

**WARNING:** Features in this module are experimental and not recommended
for production use. They are provided for academic exploration and comparison.

**Recommended approach:** Use proven methods from soilspec.models.traditional
(MBL, Cubist) or soilspec.models.deep_learning (1D CNNs).

Experimental Features
---------------------

GADFTransformer
    Convert 1D spectra to 2D images using Gramian Angular Difference Field.

    **Why experimental:**
    * Questionable scientific justification for ImageNet→spectra transfer
    * Adds complexity without clear performance benefit over 1D CNNs
    * Computationally expensive (1801² = 3.2M values per spectrum)
    * Must resize to 224×224 losing spectral resolution

    **When to consider:**
    * Very large datasets (>50k samples) for transfer learning experiments
    * Academic comparison studies
    * After trying 1D CNNs and traditional methods

**Design Philosophy:**

We include experimental features for researchers who want to explore novel
approaches, but we DO NOT recommend them as primary methods. The core package
focuses on evidence-based, proven techniques.

**Citation:**

If you use GADF for soil spectroscopy, please cite the original work:

* Wang & Oates (2015): GADF for time series
* Albinet et al. (2023): Application to soil spectroscopy (lssm package)

And note that you are using it experimentally, not as a recommended approach.

Example Usage
-------------
>>> import warnings
>>> from soilspec.experimental import GADFTransformer
>>>
>>> # Will show warning about experimental nature
>>> gadf = GADFTransformer(output_size=224)
>>>
>>> # Transform spectra to 2D images
>>> X_2d = gadf.fit_transform(X)
>>>
>>> # Use with pre-trained 2D CNN
>>> # (Not recommended - try 1D CNN first!)

Notes
-----
**Before using experimental features:**

1. Try traditional methods (MBL, Cubist)
2. Try 1D CNNs if you have >5k samples
3. Only then consider experimental approaches
4. Document why standard methods are insufficient
5. Compare performance rigorously

**Reporting results:**

If experimental features work well, please share your results:
* Open GitHub issue with benchmark comparisons
* Include dataset size, property, and performance metrics
* Help us determine if features should graduate to core package
"""

import warnings

# Show warning when importing experimental module
warnings.warn(
    "soilspec.experimental contains experimental features that are not "
    "recommended for production use. Try proven methods from "
    "soilspec.models.traditional or soilspec.models.deep_learning first.",
    UserWarning,
    stacklevel=2
)

from soilspec.experimental.gadf import (
    GADFTransformer,
    spectrum_to_gadf,
    compare_gadf_1dcnn
)

__all__ = [
    "GADFTransformer",
    "spectrum_to_gadf",
    "compare_gadf_1dcnn",
]

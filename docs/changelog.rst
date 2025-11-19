Changelog
=========

All notable changes to soilspec will be documented here.

Version 0.1.0 (In Development)
------------------------------

Initial release with core functionality.

Added
~~~~~

**Knowledge Module**:

* ``SpectralBandDatabase``: Access to 150+ literature-referenced spectral band assignments
* ``ChemicalConstraints``: Validate predictions against soil chemistry
* ``spectral_bands.csv``: Curated database from Soriano-Disla et al., Margenot et al., Tinti et al.

**Preprocessing**:

* ``SNVTransformer``: Standard Normal Variate scatter correction
* ``MSCTransformer``: Multiplicative Scatter Correction
* ``DetrendTransformer``: Polynomial baseline removal
* ``SavitzkyGolayDerivative``: Savitzky-Golay smoothed derivatives (wraps scipy)
* ``GapSegmentDerivative``: Alternative derivative method
* ``SavitzkyGolaySmoother``: Savitzky-Golay smoothing (wraps scipy)
* ``WaveletDenoiser``: Wavelet-based denoising (wraps pywavelets)
* ``MovingAverageSmoother``: Simple moving average
* ``SpectralResample``: Interpolate to common wavenumber grid
* ``TrimSpectrum``: Remove spectral regions

**Features**:

* ``PeakIntegrator``: Extract features by integrating spectral regions
* ``SpectralRatios``: Compute chemical ratios (aliphatic/aromatic, etc.)
* ``SpectralIndices``: Calculate spectral indices
* ``PhysicsInformedFeatures``: Combined feature extraction with domain knowledge

**Models**:

* ``MBLRegressor``: Memory-Based Learning for local modeling
* ``CubistRegressor``: Wrapper for pjaselin/Cubist (OSSL standard)
* ``OSSLCubistPredictor``: Complete OSSL workflow (PCA + Cubist)

**Documentation**:

* Complete API reference with scientific background
* Mathematical formulations for all methods
* 50+ literature references
* User guide and quickstart
* Installation instructions

Dependencies
~~~~~~~~~~~~

Core:

* numpy >= 1.24.0
* scipy >= 1.10.0
* pandas >= 2.0.0
* scikit-learn >= 1.3.0
* pywavelets >= 1.4.0
* cubist >= 0.3.0
* matplotlib >= 3.7.0
* seaborn >= 0.12.0

Optional:

* torch >= 2.0.0 (deep learning)
* lightning >= 2.0.0 (deep learning)
* mlflow >= 2.8.0 (experiment tracking)
* wandb >= 0.15.0 (experiment tracking)

Planned for v0.2.0
------------------

* 1D Convolutional Neural Networks
* Physics-guided attention mechanisms
* SHAP-based interpretation tools
* Additional spectral libraries (LUCAS, AfSIS)
* Automated preprocessing selection
* Model ensemble methods

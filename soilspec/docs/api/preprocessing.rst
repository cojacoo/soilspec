Preprocessing API
=================

The preprocessing module provides sklearn-compatible transformers for spectral preprocessing. All methods wrap proven scipy/pywavelets implementations rather than custom code.

**Design Philosophy**: "Just enable the wrapping" - use established signal processing libraries (scipy, pywavelets) instead of reimplementing algorithms.

Baseline Correction
-------------------

SNV (Standard Normal Variate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.SNVTransformer
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Standard Normal Variate (SNV) transformation :cite:p:`barnes1989` corrects for multiplicative scatter effects in diffuse reflectance spectroscopy. It is one of the most widely used preprocessing methods in soil spectroscopy.

**Physical Motivation:**

Diffuse reflectance spectra are affected by:

1. **Particle size** → increased scattering for smaller particles
2. **Sample packing** → variable path length
3. **Surface roughness** → baseline shifts and tilts

These physical effects cause **multiplicative scatter**, where the entire spectrum is scaled and offset without changing peak positions.

**Mathematical Formulation:**

For each spectrum :math:`\\mathbf{x} = [x_1, x_2, \\ldots, x_p]`:

.. math::

   \\text{SNV}(\\mathbf{x}) = \\frac{\\mathbf{x} - \\bar{x}}{s(\\mathbf{x})}

where:

* :math:`\\bar{x} = \\frac{1}{p}\\sum_{i=1}^{p} x_i` is the mean
* :math:`s(\\mathbf{x}) = \\sqrt{\\frac{1}{p-1}\\sum_{i=1}^{p}(x_i - \\bar{x})^2}` is the standard deviation

This centers each spectrum to zero mean and unit variance, removing additive and multiplicative scatter effects.

**When to Use SNV:**

* **Always recommended** as first preprocessing step for soil MIR/NIR spectra
* Particularly effective for pressed pellet or powder samples
* Corrects for particle size and packing density variation
* Does NOT remove baseline curvature (use Detrend for that)

**Typical Performance Impact:**

SNV typically improves model R² by 5-15% for soil spectroscopy applications :cite:p:`rinnan2009`.

**Example:**

.. code-block:: python

   from soilspec.preprocessing import SNVTransformer

   snv = SNVTransformer()
   spectra_corrected = snv.fit_transform(spectra_raw)

**References:**

See :cite:p:`barnes1989` for original formulation and :cite:p:`rinnan2009` for comprehensive review of scatter correction methods.

MSC (Multiplicative Scatter Correction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.MSCTransformer
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Multiplicative Scatter Correction (MSC) :cite:p:`geladi1985` corrects scatter effects by aligning each spectrum to a reference spectrum (typically the mean of the calibration set).

**Physical Model:**

MSC assumes observed spectrum is related to "ideal" reference by:

.. math::

   \\mathbf{x}_{\\text{observed}} = a + b \\cdot \\mathbf{x}_{\\text{ref}} + \\boldsymbol{\\epsilon}

where:

* :math:`a` = additive scatter (baseline offset)
* :math:`b` = multiplicative scatter (scaling)
* :math:`\\boldsymbol{\\epsilon}` = noise

**Algorithm:**

1. Compute reference spectrum (mean of all spectra):

   .. math::

      \\mathbf{x}_{\\text{ref}} = \\frac{1}{n}\\sum_{i=1}^{n} \\mathbf{x}_i

2. For each spectrum, perform linear regression:

   .. math::

      \\mathbf{x}_i = a_i + b_i \\mathbf{x}_{\\text{ref}} + \\boldsymbol{\\epsilon}_i

3. Correct spectrum by removing scatter effects:

   .. math::

      \\mathbf{x}_{i,\\text{corrected}} = \\frac{\\mathbf{x}_i - a_i}{b_i}

**SNV vs MSC:**

* **SNV**: Centers each spectrum individually (no reference needed)
* **MSC**: Aligns all spectra to reference spectrum
* **Performance**: Usually similar, MSC slightly better for heterogeneous samples
* **Recommendation**: Try both, use cross-validation to choose

**When to Use MSC:**

* Alternative to SNV for scatter correction
* Better when samples have different chemical composition (heterogeneous)
* Requires calibration set to compute reference

**Example:**

.. code-block:: python

   from soilspec.preprocessing import MSCTransformer

   msc = MSCTransformer()
   msc.fit(spectra_calibration)  # Compute reference
   spectra_corrected = msc.transform(spectra_new)

**References:**

See :cite:p:`geladi1985` for original method and :cite:p:`rinnan2009` for comparison with SNV.

Detrend
~~~~~~~

.. autoclass:: soilspec.preprocessing.DetrendTransformer
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Detrending removes baseline curvature (linear or polynomial trends) from spectra while preserving peak shapes. Often used in combination with SNV :cite:p:`barnes1989`.

**Physical Motivation:**

Baseline curvature in soil spectra arises from:

1. **Instrument drift** → slow wavelength-dependent detector response
2. **Sample fluorescence** → broad background emission
3. **Scattering slope** → wavelength-dependent Mie scattering

**Algorithm:**

Fit polynomial to each spectrum and subtract:

.. math::

   \\mathbf{x}_{\\text{detrended}} = \\mathbf{x} - \\text{polyfit}(\\lambda, \\mathbf{x}, \\text{degree})

where :math:`\\lambda` are wavenumbers and degree is typically 1 (linear) or 2 (quadratic).

**SNV-Detrend Combination:**

Often applied sequentially:

.. code-block:: python

   from sklearn.pipeline import Pipeline

   pipeline = Pipeline([
       ('snv', SNVTransformer()),      # Remove scatter
       ('detrend', DetrendTransformer(degree=2))  # Remove baseline
   ])

**When to Use Detrend:**

* When spectra show baseline curvature after SNV
* Particularly useful for DRIFT and ATR-FTIR
* Degree=1 for linear baseline, degree=2 for curved baseline

**References:**

See :cite:p:`barnes1989` for SNV-Detrend combination.

Derivatives
-----------

Savitzky-Golay Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.SavitzkyGolayDerivative
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Savitzky-Golay filtering :cite:p:`savitzky1964` computes smoothed derivatives using least-squares polynomial fitting in a moving window. It is the **standard method** for spectral derivatives.

**Mathematical Formulation:**

For each point :math:`x_i`, fit polynomial of degree :math:`p` to window of :math:`w` points:

.. math::

   y(t) = a_0 + a_1 t + a_2 t^2 + \\ldots + a_p t^p

The derivative at :math:`x_i` is given by polynomial coefficients:

.. math::

   \\frac{dy}{dt}\\bigg|_{t=0} = a_1 \\quad \\text{(1st derivative)}

   \\frac{d^2y}{dt^2}\\bigg|_{t=0} = 2a_2 \\quad \\text{(2nd derivative)}

**Why Derivatives for Spectroscopy:**

1. **Baseline Removal**: Derivatives eliminate additive baselines (1st) and linear baselines (2nd).

2. **Peak Resolution**: Overlapping peaks become more distinct.

3. **Sensitivity**: Enhanced sensitivity to small spectral features.

**Derivative Order:**

* **1st derivative**: Removes baseline, highlights peak positions (zero-crossings)
* **2nd derivative**: Removes linear baseline, highlights peak curvature (minima = peak centers)
* **Higher derivatives**: Rarely used (too noisy)

**Parameter Selection:**

* **window_length**: Must be odd, larger = smoother but less detail

  - Small spectra features: 5-11
  - Typical: 11-21
  - Very smooth: 21-31

* **polyorder**: Polynomial degree

  - Must be < window_length
  - Typical: 2-4
  - Higher = captures sharper features but more noise

**Performance:**

2nd derivative + PLS often improves R² by 10-20% over raw spectra for soil properties :cite:p:`stevens2013`.

**Implementation:**

Wraps :func:`scipy.signal.savgol_filter` - do not reimplement!

**Example:**

.. code-block:: python

   from soilspec.preprocessing import SavitzkyGolayDerivative

   # 2nd derivative, window=11, polynomial=2 (typical)
   sg2 = SavitzkyGolayDerivative(window_length=11, polyorder=2, deriv=2)
   spectra_d2 = sg2.fit_transform(spectra_raw)

**References:**

See :cite:p:`savitzky1964` for original algorithm and :cite:p:`rinnan2009` for application to spectroscopy.

Gap-Segment Derivatives
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.GapSegmentDerivative
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Gap-segment derivative is an alternative to Savitzky-Golay that computes derivatives using finite differences with adjustable gap size :cite:p:`rinnan2009`.

**Algorithm:**

For gap size :math:`g` and segment size :math:`s`:

.. math::

   \\frac{dx}{d\\lambda}\\bigg|_i \\approx \\frac{\\bar{x}_{i+g+s} - \\bar{x}_{i-g-s}}{2(g+s)\\Delta\\lambda}

where :math:`\\bar{x}_{i \\pm g \\pm s}` is the mean over segment :math:`[i \\pm g, i \\pm g \\pm s]`.

**Advantages:**

* Simpler than Savitzky-Golay
* More intuitive parameters (gap and segment size)
* Less sensitive to noise in some cases

**When to Use:**

* Alternative to Savitzky-Golay
* When simpler parameter tuning is desired
* Typical: gap=5, segment=5

**Example:**

.. code-block:: python

   from soilspec.preprocessing import GapSegmentDerivative

   gsd = GapSegmentDerivative(gap=5, segment=5, deriv=1)
   spectra_d1 = gsd.fit_transform(spectra_raw)

**References:**

See :cite:p:`rinnan2009` for comparison with Savitzky-Golay.

Smoothing
---------

Savitzky-Golay Smoothing
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.SavitzkyGolaySmoother
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Savitzky-Golay smoothing (0th derivative) reduces noise while preserving peak shapes better than simple moving average :cite:p:`savitzky1964`.

**When to Use:**

* Raw spectra with high noise
* Before applying other preprocessing
* Alternative to wavelet denoising

**Parameters:**

* **window_length**: Larger = more smoothing (typical: 5-15)
* **polyorder**: 2-4 (higher preserves peaks better)

**Example:**

.. code-block:: python

   from soilspec.preprocessing import SavitzkyGolaySmoother

   smoother = SavitzkyGolaySmoother(window_length=11, polyorder=3)
   spectra_smooth = smoother.fit_transform(spectra_noisy)

Wavelet Denoising
~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.WaveletDenoiser
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Wavelet denoising :cite:p:`donoho1995` uses wavelet decomposition to separate signal from noise. It is particularly effective for spectra with localized features.

**Algorithm:**

1. **Decompose**: Wavelet transform separates spectrum into approximation (low-frequency) and detail (high-frequency) coefficients:

   .. math::

      \\mathbf{x} \\xrightarrow{\\text{DWT}} [\\mathbf{a}_L, \\mathbf{d}_L, \\mathbf{d}_{L-1}, \\ldots, \\mathbf{d}_1]

2. **Threshold**: Apply soft or hard thresholding to detail coefficients to remove noise:

   .. math::

      \\mathbf{d}_i' = \\begin{cases}
      \\text{sign}(d)\\max(|d| - \\lambda, 0) & \\text{soft} \\\\
      d \\cdot \\mathbb{1}_{|d| > \\lambda} & \\text{hard}
      \\end{cases}

3. **Reconstruct**: Inverse wavelet transform:

   .. math::

      \\mathbf{x}_{\\text{denoised}} \\xleftarrow{\\text{IDWT}} [\\mathbf{a}_L, \\mathbf{d}_L', \\mathbf{d}_{L-1}', \\ldots, \\mathbf{d}_1']

**Wavelet Selection:**

* **db4, db8**: Daubechies wavelets (good general purpose)
* **sym4, sym8**: Symlets (more symmetric, better for peaks)
* **coif1, coif2**: Coiflets (smoother)

**Thresholding:**

* **Soft**: Smoother, but can over-smooth peaks
* **Hard**: Preserves peaks better, but can introduce artifacts
* **Typical choice**: Soft thresholding with :math:`\\lambda = \\sigma\\sqrt{2\\log(n)}`

**When to Use Wavelet Denoising:**

* High noise levels
* Localized spectral features (sharp peaks)
* Alternative to Savitzky-Golay smoothing
* Often better than moving average

**Implementation:**

Uses :mod:`pywt` (PyWavelets) for all wavelet operations - do not reimplement!

**Example:**

.. code-block:: python

   from soilspec.preprocessing import WaveletDenoiser

   denoiser = WaveletDenoiser(wavelet='db4', level=3, threshold_mode='soft')
   spectra_clean = denoiser.fit_transform(spectra_noisy)

**References:**

See :cite:p:`donoho1995` for wavelet thresholding theory and :cite:p:`shao2004` for application to spectroscopy.

Moving Average Smoother
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.MovingAverageSmoother
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Simple moving average smoothing using convolution. Less sophisticated than Savitzky-Golay but computationally faster.

**Algorithm:**

.. math::

   x_i' = \\frac{1}{w}\\sum_{j=i-\\lfloor w/2\\rfloor}^{i+\\lfloor w/2\\rfloor} x_j

**When to Use:**

* Very noisy spectra where sophisticated methods are not needed
* Fast smoothing for large datasets
* Generally prefer Savitzky-Golay or wavelet denoising

**Example:**

.. code-block:: python

   from soilspec.preprocessing import MovingAverageSmoother

   smoother = MovingAverageSmoother(window_size=5)
   spectra_smooth = smoother.fit_transform(spectra_noisy)

Resampling
----------

Spectral Resampling
~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.SpectralResample
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Spectral resampling interpolates spectra to a common wavenumber grid, essential for:

1. **Instrument standardization**: Different instruments have different sampling rates
2. **Library merging**: Combine spectra from multiple sources
3. **Transfer learning**: Apply models across instruments

**Interpolation Methods:**

* **linear**: Fast, preserves peak positions, recommended default
* **cubic**: Smoother, better for undersampled spectra
* **pchip**: Preserves monotonicity, good for absorption bands

**Implementation:**

Uses :func:`scipy.interpolate.interp1d` for robust interpolation.

**Example:**

.. code-block:: python

   from soilspec.preprocessing import SpectralResample

   # Resample to OSSL standard grid (600-4000 cm⁻¹, every 2 cm⁻¹)
   resampler = SpectralResample(
       new_wavenumbers=np.arange(600, 4001, 2),
       method='linear'
   )
   resampler.fit(spectra, wavenumbers=original_wavenumbers)
   spectra_resampled = resampler.transform(spectra)

Trim Spectrum
~~~~~~~~~~~~~

.. autoclass:: soilspec.preprocessing.TrimSpectrum
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Trimming removes spectral regions with:

* Low signal-to-noise ratio (detector limits)
* Atmospheric interference (CO₂, H₂O)
* No useful chemical information

**Common MIR Ranges:**

* **Full MIR**: 4000-400 cm⁻¹
* **Soil MIR**: 4000-600 cm⁻¹ (standard for OSSL)
* **Organic region**: 3000-2800 cm⁻¹ (aliphatic C-H)
* **Fingerprint**: 1800-600 cm⁻¹ (most diagnostic)

**Example:**

.. code-block:: python

   from soilspec.preprocessing import TrimSpectrum

   # Trim to OSSL standard range
   trimmer = TrimSpectrum(wavenumber_range=(600, 4000))
   trimmer.fit(spectra, wavenumbers=wavenumbers)
   spectra_trimmed = trimmer.transform(spectra)

References
----------

.. bibliography::
   :filter: docname in docnames

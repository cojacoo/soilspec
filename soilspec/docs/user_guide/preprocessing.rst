Preprocessing Guide
===================

This guide covers spectral preprocessing methods in detail. For API documentation, see :doc:`../api/preprocessing`.

.. note::
   All preprocessing methods in soilspec are sklearn-compatible transformers that wrap scipy/pywavelets implementations.

Overview
--------

Preprocessing transforms raw spectra to improve model performance by:

1. **Removing scatter effects** (SNV, MSC)
2. **Removing baseline trends** (Detrend, derivatives)
3. **Reducing noise** (Smoothing, wavelet denoising)
4. **Standardizing format** (Resampling, trimming)

Recommended Workflows
---------------------

For soil MIR spectroscopy, we recommend:

**Standard workflow (OSSL-style)**:

.. code-block:: python

   Pipeline([
       ('snv', SNVTransformer()),
       ('model', CubistRegressor())
   ])

**With derivatives**:

.. code-block:: python

   Pipeline([
       ('snv', SNVTransformer()),
       ('sg2', SavitzkyGolayDerivative(window_length=11, polyorder=2, deriv=2)),
       ('model', PLSRegression())
   ])

**With denoising**:

.. code-block:: python

   Pipeline([
       ('denoise', WaveletDenoiser(wavelet='db4', level=3)),
       ('snv', SNVTransformer()),
       ('detrend', DetrendTransformer(degree=2)),
       ('model', CubistRegressor())
   ])

See :doc:`../api/preprocessing` for detailed API documentation and scientific background.

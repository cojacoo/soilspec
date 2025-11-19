Quick Start Guide
=================

This guide shows you how to get started with soilspec for soil property prediction from MIR spectra.

Basic Workflow
--------------

The typical soilspec workflow:

1. **Load spectral data** (MIR/NIR spectra + soil properties)
2. **Preprocess spectra** (SNV, derivatives, smoothing)
3. **Extract features** (optional: physics-informed features)
4. **Train model** (MBL, Cubist, or deep learning)
5. **Predict** soil properties for new spectra
6. **Interpret** results using chemical context

Minimal Example
---------------

Predict soil organic carbon (SOC) from MIR spectra using Cubist:

.. code-block:: python

   import numpy as np
   from soilspec.preprocessing import SNVTransformer
   from soilspec.models.traditional import CubistRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import Pipeline

   # Load your data
   # spectra: (n_samples, n_wavelengths) array
   # soc: (n_samples,) array of SOC values
   # wavenumbers: (n_wavelengths,) array

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       spectra, soc, test_size=0.2, random_state=42
   )

   # Create pipeline
   pipeline = Pipeline([
       ('snv', SNVTransformer()),  # Scatter correction
       ('cubist', CubistRegressor(n_committees=20, neighbors=5))  # OSSL standard
   ])

   # Train
   pipeline.fit(X_train, y_train)

   # Predict
   y_pred = pipeline.predict(X_test)

   # Evaluate
   from sklearn.metrics import r2_score, mean_squared_error
   r2 = r2_score(y_test, y_pred)
   rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   print(f"R² = {r2:.3f}, RMSE = {rmse:.3f}")

Complete Example with MBL
--------------------------

Memory-Based Learning with physics-informed features:

.. code-block:: python

   from soilspec.preprocessing import SNVTransformer
   from soilspec.features import PhysicsInformedFeatures
   from soilspec.models.traditional import MBLRegressor
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   # Create comprehensive pipeline
   pipeline = Pipeline([
       ('snv', SNVTransformer()),
       ('features', PhysicsInformedFeatures(
           include_peaks=True,
           include_ratios=True,
           region_selection='key'
       )),
       ('scaler', StandardScaler()),
       ('mbl', MBLRegressor(
           k_neighbors=50,
           similarity_metric='cosine',
           weighting='gaussian',
           local_model='pls',
           n_components=10
       ))
   ])

   # Train (pass wavenumbers to feature extractor)
   pipeline.fit(
       X_train, y_train,
       features__wavenumbers=wavenumbers
   )

   # Predict with uncertainty
   mbl_model = pipeline.named_steps['mbl']

   # Transform test data through preprocessing
   X_test_features = pipeline[:-1].transform(X_test)

   # Predict with uncertainty
   y_pred, y_std = mbl_model.predict_with_uncertainty(X_test_features)

   # 95% confidence interval
   y_lower = y_pred - 2 * y_std
   y_upper = y_pred + 2 * y_std

   print(f"Prediction: {y_pred[0]:.2f} ± {2*y_std[0]:.2f}")

OSSL-Style Workflow
-------------------

Replicate Open Soil Spectral Library methodology:

.. code-block:: python

   from soilspec.preprocessing import SNVTransformer
   from soilspec.models.traditional import OSSLCubistPredictor
   from sklearn.pipeline import Pipeline

   # OSSL workflow: SNV → PCA (120 PCs) → Cubist
   pipeline = Pipeline([
       ('snv', SNVTransformer()),
       ('ossl_cubist', OSSLCubistPredictor(
           n_pcs=120,
           cubist_params={
               'n_committees': 20,
               'neighbors': 5
           }
       ))
   ])

   # Train
   pipeline.fit(X_train, y_train)

   # Predict
   y_pred = pipeline.predict(X_test)

   # View rules (interpretable)
   cubist_model = pipeline.named_steps['ossl_cubist'].cubist_
   print(cubist_model.get_rules())

Advanced Preprocessing
----------------------

Multiple preprocessing steps:

.. code-block:: python

   from soilspec.preprocessing import (
       SNVTransformer,
       DetrendTransformer,
       SavitzkyGolayDerivative,
       WaveletDenoiser
   )
   from sklearn.pipeline import Pipeline

   # Option 1: SNV + 2nd derivative (common for PLS)
   pipeline_deriv = Pipeline([
       ('snv', SNVTransformer()),
       ('sg2', SavitzkyGolayDerivative(
           window_length=11,
           polyorder=2,
           deriv=2
       )),
       ('cubist', CubistRegressor())
   ])

   # Option 2: Wavelet denoising + SNV-Detrend
   pipeline_wavelet = Pipeline([
       ('denoise', WaveletDenoiser(wavelet='db4', level=3)),
       ('snv', SNVTransformer()),
       ('detrend', DetrendTransformer(degree=2)),
       ('cubist', CubistRegressor())
   ])

   # Compare both
   pipeline_deriv.fit(X_train, y_train)
   pipeline_wavelet.fit(X_train, y_train)

   r2_deriv = pipeline_deriv.score(X_test, y_test)
   r2_wavelet = pipeline_wavelet.score(X_test, y_test)

   print(f"2nd derivative: R² = {r2_deriv:.3f}")
   print(f"Wavelet denoising: R² = {r2_wavelet:.3f}")

Multi-Property Prediction
--------------------------

Predict multiple soil properties with constraint checking:

.. code-block:: python

   from soilspec.knowledge import ChemicalConstraints
   from sklearn.base import clone

   # Train models for multiple properties
   base_pipeline = Pipeline([
       ('snv', SNVTransformer()),
       ('cubist', CubistRegressor(n_committees=20))
   ])

   models = {}
   properties = ['soc', 'clay', 'silt', 'sand', 'total_n', 'cec']

   for prop in properties:
       models[prop] = clone(base_pipeline)
       models[prop].fit(X_train, y_train_dict[prop])

   # Predict all properties
   predictions = {}
   for prop in properties:
       predictions[prop] = models[prop].predict(X_test)

   # Validate chemical consistency
   constraints = ChemicalConstraints()

   for i in range(len(X_test)):
       sample_pred = {prop: predictions[prop][i] for prop in properties}
       validation = constraints.validate_prediction(sample_pred)

       if not validation['valid']:
           print(f"Sample {i} violates constraints:")
           print(validation['violations'])

Using Spectral Band Database
-----------------------------

Explore domain knowledge:

.. code-block:: python

   from soilspec.knowledge import SpectralBandDatabase

   bands = SpectralBandDatabase()

   # Summary
   summary = bands.summarize()
   print(f"Total bands: {summary['total_bands']}")

   # Find organic matter bands
   org_bands = bands.get_bands(type='org')
   print(f"Organic bands: {len(org_bands)}")

   # Search for specific chemistry
   protein_bands = bands.get_bands(information='Amide')
   print("Protein-related bands:")
   for _, band in protein_bands.iterrows():
       print(f"  {band['band_center']} cm⁻¹: {band['information']}")

   # Get key regions for feature extraction
   regions = bands.get_key_regions()
   for name, info in regions.items():
       print(f"{name}: {info['range']} cm⁻¹ - {info['description']}")

Loading Data
------------

From Bruker OPUS files:

.. code-block:: python

   from brukeropusreader import read_file
   import numpy as np

   # Read single file
   opus_data = read_file('sample.0')
   spectrum = opus_data.get_range('AB')  # Absorbance
   wavenumbers = np.array([p[0] for p in spectrum])
   absorbance = np.array([p[1] for p in spectrum])

   # Read multiple files
   import glob

   spectra_list = []
   for file in glob.glob('*.0'):
       opus_data = read_file(file)
       spectrum = opus_data.get_range('AB')
       absorbance = np.array([p[1] for p in spectrum])
       spectra_list.append(absorbance)

   spectra = np.array(spectra_list)

From CSV:

.. code-block:: python

   import pandas as pd

   # Spectra in CSV (rows = samples, columns = wavenumbers)
   df_spectra = pd.read_csv('spectra.csv', index_col=0)
   spectra = df_spectra.values
   wavenumbers = df_spectra.columns.astype(float).values

   # Soil properties
   df_properties = pd.read_csv('soil_properties.csv', index_col=0)
   soc = df_properties['SOC'].values

Common Pitfalls
---------------

**1. Not passing wavenumbers to feature extractor:**

.. code-block:: python

   # WRONG - will fail
   features = PhysicsInformedFeatures()
   features.fit(X_train)  # No wavenumbers!

   # CORRECT
   features.fit(X_train, wavenumbers=wavenumbers)

   # In pipeline, use parameter routing
   pipeline.fit(X_train, y_train, features__wavenumbers=wavenumbers)

**2. Forgetting to fit transformers:**

.. code-block:: python

   # WRONG
   snv = SNVTransformer()
   X_transformed = snv.transform(X)  # Not fitted!

   # CORRECT
   snv = SNVTransformer()
   snv.fit(X)
   X_transformed = snv.transform(X)

   # Or use fit_transform
   X_transformed = snv.fit_transform(X)

**3. Applying test preprocessing with wrong reference:**

.. code-block:: python

   # WRONG - MSC fitted on test set
   msc_train = MSCTransformer().fit_transform(X_train)
   msc_test = MSCTransformer().fit_transform(X_test)  # Different reference!

   # CORRECT - fit on training, transform test
   msc = MSCTransformer()
   msc.fit(X_train)
   X_train_transformed = msc.transform(X_train)
   X_test_transformed = msc.transform(X_test)  # Same reference

**4. Inconsistent wavenumber grids:**

.. code-block:: python

   # Ensure consistent grid
   from soilspec.preprocessing import SpectralResample

   # Define standard grid
   standard_wn = np.arange(600, 4001, 2)

   # Resample all spectra
   resampler = SpectralResample(new_wavenumbers=standard_wn)
   resampler.fit(X, wavenumbers=original_wn)
   X_resampled = resampler.transform(X)

Next Steps
----------

Now that you've seen the basics:

* :doc:`preprocessing` - Detailed preprocessing methods
* :doc:`feature_engineering` - Physics-informed features
* :doc:`models` - Model selection and tuning
* :doc:`examples` - Complete worked examples

For API details, see :doc:`../api/preprocessing`, :doc:`../api/features`, :doc:`../api/models`.

Features API
============

The features module provides physics-informed feature extraction methods that leverage domain knowledge from spectral band assignments.

**Design Philosophy**: Instead of using all 1801 wavelengths, extract ~50-100 chemically meaningful features based on literature-referenced spectral regions.

Peak Integration
----------------

.. autoclass:: soilspec.features.PeakIntegrator
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Peak integration extracts chemically meaningful features by integrating absorbance over literature-defined spectral regions :cite:p:`soriano2014,margenot2017`.

**Physical Basis - Beer-Lambert Law:**

For a given wavenumber region :math:`[\\nu_1, \\nu_2]`, the integrated absorbance is proportional to the concentration of the absorbing functional group:

.. math::

   A_{\\text{integrated}} = \\int_{\\nu_1}^{\\nu_2} A(\\nu) d\\nu \\propto C \\cdot \\epsilon_{\\text{avg}} \\cdot b

where:

* :math:`C` = concentration of functional group (e.g., aliphatic C-H)
* :math:`\\epsilon_{\\text{avg}}` = average molar absorptivity over region
* :math:`b` = path length

**Why Peak Integration Works:**

1. **Dimensionality Reduction**: 1801 wavelengths → ~50 features (fewer parameters to fit)

2. **Chemical Interpretation**: Each feature corresponds to specific functional group:

   * 2920 cm⁻¹: Aliphatic C-H (organic matter quality)
   * 1630 cm⁻¹: Amide I + COO⁻ (protein/carboxylate)
   * 3620 cm⁻¹: Clay OH (clay mineralogy)

3. **Noise Reduction**: Integration averages over wavelengths, reducing high-frequency noise

4. **Domain Knowledge**: Uses 150+ literature-referenced band assignments from soil science

**Spectral Regions Database:**

The package includes ``spectral_bands.csv`` with assignments from:

* :cite:p:`soriano2014`: Comprehensive review of MIR/NIR for soil
* :cite:p:`margenot2017`: Detailed functional group assignments
* :cite:p:`tinti2015`: Vibrational spectroscopy review
* :cite:p:`nguyen2013`: DRIFT spectroscopy methods
* :cite:p:`reeves2012`: NIR vs MIR comparison

**Example Regions:**

+-----------------+------------------+--------------------------------+
| Region          | Wavenumber       | Chemical Assignment            |
+=================+==================+================================+
| Aliphatic C-H   | 2920 cm⁻¹        | Lipids, waxes, alkanes         |
+-----------------+------------------+--------------------------------+
| Aromatic C-H    | 3030 cm⁻¹        | Lignin, aromatic compounds     |
+-----------------+------------------+--------------------------------+
| Amide I         | 1650 cm⁻¹        | Protein C=O stretch            |
+-----------------+------------------+--------------------------------+
| Amide II        | 1550 cm⁻¹        | Protein N-H bend               |
+-----------------+------------------+--------------------------------+
| Carbohydrate    | 1030 cm⁻¹        | Polysaccharides C-O stretch    |
+-----------------+------------------+--------------------------------+
| Clay OH         | 3620 cm⁻¹        | Kaolinite structural OH        |
+-----------------+------------------+--------------------------------+
| Quartz          | 800 cm⁻¹         | Si-O symmetric stretch         |
+-----------------+------------------+--------------------------------+

**Region Selection Modes:**

* **'key'**: ~20 most diagnostic regions (balanced organic + mineral)
* **'organic'**: Organic functional groups only (~30 regions)
* **'mineral'**: Clay and mineral features only (~25 regions)
* **'all'**: All 150+ regions (usually too many)

**Integration Methods:**

* **'trapz'**: Trapezoidal rule (recommended, fast and accurate)
* **'simpson'**: Simpson's rule (more accurate for smooth peaks)
* **'sum'**: Simple summation (fastest, less accurate)

**Typical Workflow:**

.. code-block:: python

   from soilspec.features import PeakIntegrator
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from soilspec.models.traditional import MBLRegressor

   # Create feature extraction pipeline
   pipeline = Pipeline([
       ('peaks', PeakIntegrator(region_selection='key')),
       ('scaler', StandardScaler()),
       ('model', MBLRegressor(k_neighbors=50))
   ])

   # Fit with wavenumbers
   pipeline.fit(spectra, soc, peaks__wavenumbers=wavenumbers)

   # Predict
   soc_pred = pipeline.predict(spectra_new)

   # View feature names
   feature_names = pipeline.named_steps['peaks'].get_feature_names_out()
   # ['aliphatic_ch', 'amide_i', 'clay_oh', ...]

**Performance:**

Peak integration typically achieves 85-95% of full-spectrum performance with only 5-10% of features:

* Full spectrum (1801 features): R² = 0.90
* Peak integration (50 features): R² = 0.85-0.88

Benefits:

* **Faster training**: 10-50x faster model fitting
* **Less overfitting**: Fewer parameters to estimate
* **Interpretable**: Can explain which functional groups matter
* **Robust**: Less sensitive to instrument noise

**Implementation:**

Uses ``spectral_bands.csv`` database and :class:`soilspec.knowledge.SpectralBandDatabase` for region definitions.

**References:**

See :cite:p:`soriano2014` for comprehensive spectral assignments, :cite:p:`margenot2017` for functional group interpretation, and :cite:p:`tinti2015` for MIR band positions.

Spectral Ratios
---------------

.. autoclass:: soilspec.features.SpectralRatios
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Spectral ratios normalize for sample thickness and compute relative abundances of functional groups :cite:p:`margenot2017`.

**Why Ratios Matter:**

1. **Self-Normalization**: Ratios cancel out sample thickness variations:

   .. math::

      R = \\frac{A_{\\nu_1}}{A_{\\nu_2}} = \\frac{C_1 \\epsilon_1}{C_2 \\epsilon_2}

   Independent of path length :math:`b`.

2. **Chemical Interpretation**: Ratios indicate:

   * **Aliphatic/Aromatic**: Organic matter decomposition stage (higher = fresher)
   * **Organic/Mineral**: Relative organic matter content
   * **Carbohydrate/Amide**: Plant vs microbial residues

**Common Soil Spectroscopy Ratios:**

+-------------------------+---------------------------+------------------------------+
| Ratio                   | Peaks Used                | Interpretation               |
+=========================+===========================+==============================+
| Aliphatic/Aromatic      | 2920 cm⁻¹ / 1620 cm⁻¹     | OM freshness, decomposition  |
+-------------------------+---------------------------+------------------------------+
| Organic/Mineral         | 2920 cm⁻¹ / 1030 cm⁻¹     | Relative OM content          |
+-------------------------+---------------------------+------------------------------+
| Carbohydrate/Amide      | 1030 cm⁻¹ / 1550 cm⁻¹     | Plant vs microbial origin    |
+-------------------------+---------------------------+------------------------------+
| Clay/Quartz             | 3620 cm⁻¹ / 800 cm⁻¹      | Mineralogy                   |
+-------------------------+---------------------------+------------------------------+

**Example:**

.. code-block:: python

   from soilspec.features import SpectralRatios

   ratios = SpectralRatios(ratios=['aliphatic_aromatic', 'organic_mineral'])
   ratios.fit(spectra, wavenumbers=wavenumbers)
   ratio_features = ratios.transform(spectra)
   # Returns (n_samples, 2) array

**References:**

See :cite:p:`margenot2017` for interpretation of spectral ratios in soil science.

Spectral Indices
----------------

.. autoclass:: soilspec.features.SpectralIndices
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Spectral indices combine multiple wavelengths using empirical formulas optimized for specific soil properties :cite:p:`viscarra2016`.

**Index Types:**

1. **Simple Ratios**: :math:`I = A_{\\lambda_1} / A_{\\lambda_2}`

2. **Normalized Differences**: :math:`I = (A_{\\lambda_1} - A_{\\lambda_2}) / (A_{\\lambda_1} + A_{\\lambda_2})`

3. **Linear Combinations**: :math:`I = w_1 A_{\\lambda_1} + w_2 A_{\\lambda_2} + \\ldots`

**Common Indices:**

* **Organic Matter Index**: Based on C-H absorption at 2920 cm⁻¹
* **Clay Index**: Based on Al-OH and Si-O features
* **Carbonate Index**: Based on CO₃²⁻ absorption at 1420, 880 cm⁻¹

**Example:**

.. code-block:: python

   from soilspec.features import SpectralIndices

   indices = SpectralIndices(indices=['om_index', 'clay_index'])
   indices.fit(spectra, wavenumbers=wavenumbers)
   index_features = indices.transform(spectra)

**References:**

See :cite:p:`viscarra2016` for review of spectral indices for soil properties.

Physics-Informed Features
--------------------------

.. autoclass:: soilspec.features.PhysicsInformedFeatures
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

PhysicsInformedFeatures combines peak integration, ratios, and indices into a single transformer for comprehensive feature extraction.

**Feature Categories:**

1. **Peak Integrals** (~20-50 features): Absolute abundances of functional groups
2. **Spectral Ratios** (~10-20 features): Relative abundances and OM quality
3. **Spectral Indices** (~5-10 features): Empirical predictors for properties

Total: ~35-80 features (vs 1801 original wavelengths)

**When to Use:**

* **Small training sets** (n < 500): Reduces overfitting risk
* **Interpretability required**: Can explain predictions via features
* **Transfer learning**: More robust across instruments than full spectra
* **Fast inference**: 20-50x faster prediction with MBL/Cubist

**Typical Performance:**

On OSSL benchmark (10-fold CV):

* Full spectrum + Cubist: R² = 0.90, training time = 120s
* Physics features + Cubist: R² = 0.87, training time = 8s

**Trade-off**: ~3% R² loss for 15x speedup and full interpretability.

**Example Pipeline:**

.. code-block:: python

   from soilspec.features import PhysicsInformedFeatures
   from soilspec.models.traditional import CubistRegressor
   from sklearn.pipeline import Pipeline

   pipeline = Pipeline([
       ('features', PhysicsInformedFeatures(
           include_peaks=True,
           include_ratios=True,
           include_indices=True,
           region_selection='key'
       )),
       ('model', CubistRegressor(n_committees=20))
   ])

   # Fit
   pipeline.fit(spectra, clay, features__wavenumbers=wavenumbers)

   # Predict
   clay_pred = pipeline.predict(spectra_test)

   # Feature importance (from Cubist rules)
   rules = pipeline.named_steps['model'].get_rules()
   print(rules)  # Shows which chemical features are important

**Integration with Chemical Constraints:**

Can be combined with :class:`soilspec.knowledge.ChemicalConstraints` to enforce physical consistency:

.. code-block:: python

   from soilspec.knowledge import ChemicalConstraints

   # Predict multiple properties
   soc_pred = pipeline_soc.predict(spectra)
   clay_pred = pipeline_clay.predict(spectra)

   # Check CEC consistency
   constraints = ChemicalConstraints()
   cec_expected = constraints.cec_constraint(soc_pred, clay_pred)
   cec_pred = pipeline_cec.predict(spectra)

   # cec_pred should be close to cec_expected

**References:**

See :cite:p:`margenot2017` for physics-informed approach and :cite:p:`soriano2014` for spectral region assignments.

References
----------

.. bibliography::
   :filter: docname in docnames

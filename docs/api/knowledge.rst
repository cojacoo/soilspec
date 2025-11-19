Knowledge API
=============

The knowledge module provides access to domain knowledge for soil spectroscopy, including spectral band assignments, chemical constraints, and visualization tools.

Spectral Band Database
-----------------------

.. autoclass:: soilspec.knowledge.SpectralBandDatabase
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

The Spectral Band Database provides programmatic access to 150+ literature-referenced MIR/NIR peak assignments for soil spectroscopy :cite:p:`soriano2014,margenot2017,tinti2015`.

**Database Structure:**

``spectral_bands.csv`` contains:

* **band_start, band_end**: Wavenumber range (cm⁻¹)
* **band_center**: Peak center wavenumber
* **type**: Functional group type ('org', 'min', 'water')
* **information**: Chemical assignment (e.g., "Aliphatic C-H stretch")
* **description**: Detailed description
* **references**: Literature citations

**Example Entries:**

+-------------+-----------+--------+-----------------------------+----------------------------+
| Band Center | Range     | Type   | Information                 | Reference                  |
+=============+===========+========+=============================+============================+
| 2920        | 2935-2915 | org    | Aliphatic C-H asymmetric    | Margenot et al. 2017       |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 2850        | 2865-2845 | org    | Aliphatic C-H symmetric     | Margenot et al. 2017       |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 1720        | 1740-1700 | org    | Carboxylic acid C=O         | Tinti et al. 2015          |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 1650        | 1660-1640 | org    | Amide I (protein C=O)       | Soriano-Disla et al. 2014  |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 1550        | 1560-1540 | org    | Amide II (protein N-H)      | Soriano-Disla et al. 2014  |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 3620        | 3630-3610 | min    | Kaolinite inner OH          | Nguyen et al. 1991         |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 3695        | 3705-3685 | min    | Kaolinite outer OH          | Nguyen et al. 1991         |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 1030        | 1050-1000 | org    | Polysaccharide C-O          | Tinti et al. 2015          |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 1420        | 1440-1400 | min    | Carbonate asymmetric CO₃    | Reeves 2010                |
+-------------+-----------+--------+-----------------------------+----------------------------+
| 800         | 820-780   | min    | Quartz Si-O symmetric       | Nguyen et al. 1991         |
+-------------+-----------+--------+-----------------------------+----------------------------+

**Usage Examples:**

**1. Query by functional group type:**

.. code-block:: python

   from soilspec.knowledge import SpectralBandDatabase

   bands = SpectralBandDatabase()

   # Get all organic bands
   org_bands = bands.get_bands(type='org')
   print(f"Found {len(org_bands)} organic functional groups")

   # Get mineral bands
   min_bands = bands.get_bands(type='min')

**2. Search by chemical information:**

.. code-block:: python

   # Find all clay-related bands
   clay_bands = bands.get_bands(information='clay')

   # Find protein-related bands
   protein_bands = bands.get_bands(information='Amide')

**3. Get region type for specific wavenumber:**

.. code-block:: python

   # What is 2920 cm⁻¹?
   info = bands.get_region_type(2920)
   print(info[0]['information'])  # "Aliphatic C-H stretch"

   # What is 3620 cm⁻¹?
   info = bands.get_region_type(3620)
   print(info[0]['information'])  # "Kaolinite structural OH"

**4. Create spectral masks:**

.. code-block:: python

   import numpy as np

   wavenumbers = np.arange(600, 4001, 2)  # OSSL grid

   # Create mask for organic regions
   org_mask = bands.create_mask(wavenumbers, type='org')

   # Apply mask to spectra
   spectra_organic_only = spectra[:, org_mask]
   wavenumbers_organic = wavenumbers[org_mask]

**5. Get predefined key regions:**

.. code-block:: python

   regions = bands.get_key_regions()

   print(regions['aliphatic_ch'])
   # {'range': (2935, 2915),
   #  'center': 2920,
   #  'description': 'Aliphatic C-H stretch (lipids, waxes)'}

**6. Database summary:**

.. code-block:: python

   summary = bands.summarize()
   print(f"Total bands: {summary['total_bands']}")
   print(f"Organic: {summary['by_type']['org']}")
   print(f"Mineral: {summary['by_type']['min']}")
   print(f"Wavenumber range: {summary['wavenumber_range']}")

**Integration with Feature Extraction:**

The database is automatically used by :class:`soilspec.features.PeakIntegrator`:

.. code-block:: python

   from soilspec.features import PeakIntegrator

   # Automatically loads spectral_bands.csv
   integrator = PeakIntegrator(region_selection='key')
   integrator.fit(spectra, wavenumbers=wavenumbers)

   # View which regions are used
   print(integrator.regions_)

**References:**

* :cite:p:`soriano2014`: Comprehensive MIR/NIR review for soil properties
* :cite:p:`margenot2017`: Detailed functional group assignments and interpretation
* :cite:p:`tinti2015`: Vibrational spectroscopy fundamentals
* :cite:p:`nguyen2013`: DRIFT spectroscopy methods and peak positions
* :cite:p:`reeves2012`: NIR vs MIR comparison for soil analysis

Chemical Constraints
---------------------

.. autoclass:: soilspec.knowledge.ChemicalConstraints
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Chemical constraints encode empirical relationships between soil properties that can be used to validate and improve predictions.

**Implemented Constraints:**

**1. Cation Exchange Capacity (CEC):**

CEC is approximately related to clay and organic carbon :cite:p:`soriano2014`:

.. math::

   \\text{CEC} \\approx 0.5 \\times \\text{Clay} + 2.0 \\times \\text{SOC} + \\epsilon

where:

* CEC in cmol(+)/kg
* Clay in %
* SOC in %
* :math:`\\epsilon` = site-specific adjustment for clay mineralogy

This relationship holds because:

* Clay minerals contribute ~0.1-0.9 cmol(+)/kg per % clay (avg 0.5)
* Organic matter contributes ~1.5-2.5 cmol(+)/kg per % SOC (avg 2.0)

**2. Texture Sum Constraint:**

For complete particle size analysis:

.. math::

   \\text{Sand} + \\text{Silt} + \\text{Clay} = 100\\%

**3. Carbon:Nitrogen Ratio:**

Typical soil C:N ratios range from 8-15:

.. math::

   8 \\leq \\frac{\\text{Total C}}{\\text{Total N}} \\leq 15 \\quad \\text{(for mineral soils)}

Higher ratios (>15) indicate fresh plant residues, lower ratios (<8) indicate microbial biomass or inorganic N.

**Usage Examples:**

**1. Validate CEC predictions:**

.. code-block:: python

   from soilspec.knowledge import ChemicalConstraints

   constraints = ChemicalConstraints()

   # Predict properties independently
   soc_pred = model_soc.predict(spectra)     # 2.5%
   clay_pred = model_clay.predict(spectra)   # 35%
   cec_pred = model_cec.predict(spectra)     # 15 cmol/kg

   # Check CEC consistency
   cec_expected = constraints.cec_constraint(soc_pred, clay_pred)
   # Expected: 0.5*35 + 2.0*2.5 = 17.5 + 5.0 = 22.5 cmol/kg

   # Measured: 15 cmol/kg - inconsistent!
   # Likely: Smectitic clays (higher CEC) or prediction error

**2. Multi-task consistency:**

.. code-block:: python

   # Predict all properties
   predictions = {
       'soc': model_soc.predict(spectra),
       'clay': model_clay.predict(spectra),
       'silt': model_silt.predict(spectra),
       'sand': model_sand.predict(spectra),
       'total_n': model_n.predict(spectra),
       'cec': model_cec.predict(spectra)
   }

   # Validate all constraints
   validation = constraints.validate_prediction(predictions[0])

   if not validation['valid']:
       print("Constraint violations:")
       for violation in validation['violations']:
           print(f"  - {violation}")

**3. Constrained optimization:**

Use constraints to improve predictions via post-hoc adjustment:

.. code-block:: python

   # Ensure texture sums to 100%
   sand, silt, clay = predictions['sand'], predictions['silt'], predictions['clay']
   total = sand + silt + clay

   # Normalize to sum to 100
   sand_corrected = 100 * sand / total
   silt_corrected = 100 * silt / total
   clay_corrected = 100 * clay / total

**Future Extensions:**

Potential additional constraints:

* **pH-CEC relationship**: Higher CEC generally correlates with higher pH buffering
* **Organic C vs Total C**: Inorganic C (carbonates) contribution
* **Spectral constraints**: Certain peak ratios should correlate with properties

**References:**

See :cite:p:`soriano2014` for empirical relationships between soil properties.

Visualization
-------------

.. automodule:: soilspec.knowledge.visualization
   :members:
   :undoc-members:

**Scientific Background**

Visualization tools for interpreting spectral data with annotated chemical assignments :cite:p:`margenot2017`.

**Available Visualizations:**

**1. Annotated Spectra:**

Plot spectra with chemical region annotations from spectral_bands.csv:

.. code-block:: python

   from soilspec.knowledge.visualization import plot_annotated_spectrum

   plot_annotated_spectrum(
       wavenumbers,
       spectrum,
       highlight_regions=['aliphatic_ch', 'amide_i', 'clay_oh']
   )

**2. Regional Comparison:**

Compare spectra colored by functional group regions:

.. code-block:: python

   from soilspec.knowledge.visualization import plot_regional_comparison

   plot_regional_comparison(wavenumbers, spectra, region_type='org')

**3. Feature Importance with Chemistry:**

Overlay feature importance on spectral regions:

.. code-block:: python

   from soilspec.knowledge.visualization import plot_feature_importance

   # Get feature importance from model
   importance = cubist.feature_importances_

   # Plot with chemical context
   plot_feature_importance(
       feature_names=integrator.get_feature_names_out(),
       importance=importance,
       spectral_db=bands
   )

These visualizations help interpret model predictions in chemical context, bridging machine learning and soil science.

**References:**

See :cite:p:`margenot2017` for best practices in spectral interpretation.

References
----------

.. bibliography::
   :filter: docname in docnames

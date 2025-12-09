Interpretation API
==================

The interpretation module provides tools for understanding model predictions and linking them to soil chemistry.

.. note::
   This module is under active development. Currently, interpretation is provided through:

   * :meth:`soilspec.models.traditional.CubistRegressor.get_rules` - View interpretable decision rules
   * :meth:`soilspec.models.traditional.MBLRegressor.predict_with_uncertainty` - Uncertainty estimates
   * :class:`soilspec.features.PhysicsInformedFeatures` - Chemically meaningful features
   * :class:`soilspec.knowledge.ChemicalConstraints` - Validate predictions against chemistry

Planned Features
----------------

Future releases will include:

**SHAP Values for Spectral Models**
   Explain predictions using Shapley values adapted for spectral data.

**Chemical Attribution**
   Link predictions to specific functional groups and chemical features.

**Uncertainty Quantification**
   Bayesian approaches and ensemble methods for prediction intervals.

**Model Diagnostics**
   Tools to identify when models are extrapolating or predictions are unreliable.

Current Interpretation Methods
-------------------------------

Cubist Rules
~~~~~~~~~~~~

View interpretable decision rules from Cubist models:

.. code-block:: python

   from soilspec.models.traditional import CubistRegressor

   cubist = CubistRegressor(n_committees=5, neighbors=5)
   cubist.fit(spectra, soc)

   # View rules
   rules = cubist.get_rules()
   print(rules)

   # Example output:
   # Rule 1: [50 cases, mean 2.3, range 1.1 to 4.2, est err 0.3]
   #   if
   #     PC_12 > 0.023
   #     PC_34 <= -0.014
   #   then
   #     outcome = 2.1 + 0.8*PC_12 - 0.3*PC_34 + ...

MBL Uncertainty
~~~~~~~~~~~~~~~

Get prediction uncertainty from local model variance:

.. code-block:: python

   from soilspec.models.traditional import MBLRegressor

   mbl = MBLRegressor(k_neighbors=50)
   mbl.fit(spectra_train, clay_train)

   # Predict with uncertainty
   clay_pred, clay_std = mbl.predict_with_uncertainty(spectra_test)

   # clay_std = local standard deviation
   # 95% confidence interval ≈ [clay_pred - 2*clay_std, clay_pred + 2*clay_std]

Physics-Informed Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use chemically interpretable features:

.. code-block:: python

   from soilspec.features import PhysicsInformedFeatures
   from soilspec.models.traditional import CubistRegressor

   # Extract chemical features
   features = PhysicsInformedFeatures()
   X_features = features.fit_transform(spectra, wavenumbers=wavenumbers)

   # Feature names show chemistry
   names = features.get_feature_names_out()
   # ['aliphatic_ch', 'aromatic_ch', 'amide_i', 'clay_oh', ...]

   # Train interpretable model
   cubist = CubistRegressor()
   cubist.fit(X_features, soc)

   # Rules now reference chemistry!
   rules = cubist.get_rules()
   # "if aliphatic_ch > 0.15 and clay_oh > 0.08 then SOC = ..."

Chemical Constraints
~~~~~~~~~~~~~~~~~~~~

Validate predictions against soil chemistry:

.. code-block:: python

   from soilspec.knowledge import ChemicalConstraints

   constraints = ChemicalConstraints()

   # Check if predictions make chemical sense
   validation = constraints.validate_prediction({
       'soc': 2.5,
       'clay': 35,
       'silt': 40,
       'sand': 25,
       'total_n': 0.2,
       'cec': 22
   })

   if not validation['valid']:
       print("Warning: Predictions violate chemical constraints!")
       print(validation['violations'])

References
----------

For interpretation methods in soil spectroscopy, see:

* :cite:p:`margenot2017`: Interpreting MIR spectra for soil properties
* :cite:p:`quinlan1992`: Interpretable rule-based models
* :cite:p:`ramirez2013`: Understanding local model predictions

.. bibliography::
   :filter: docname in docnames

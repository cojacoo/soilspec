.. soilspec documentation master file

soilspec: Evidence-Based Machine Learning for Soil Spectroscopy
================================================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/yourusername/soilspec/blob/main/LICENSE
   :alt: License

**soilspec** is a Python package for soil spectroscopy analysis using evidence-based machine learning methods. It combines physics-informed feature engineering with state-of-the-art statistical and machine learning models.

Key Features
------------

**Physics-Informed Feature Engineering**
   Extract chemically meaningful features using spectral band assignments from soil science literature.

**Strong Statistical Models**
   * **MBL (Memory-Based Learning)**: Local modeling with superior transfer learning capabilities
   * **Cubist**: OSSL standard model using rule-based regression

**Proven Preprocessing**
   Sklearn-compatible wrappers for scipy/pywavelets signal processing methods.

**Domain Knowledge Integration**
   Leverage 150+ literature-referenced spectral band assignments for interpretable predictions.

Scientific Background
---------------------

Soil spectroscopy uses infrared absorption to predict soil properties (carbon, nitrogen, clay content, etc.) from spectral signatures. This package implements methods proven in peer-reviewed literature:

* **Memory-Based Learning (MBL)**: Local calibrations outperform global models for transfer learning :cite:p:`ramirez2013`. Used extensively for large spectral libraries.

* **Cubist Rule-Based Regression**: Standard model for Open Soil Spectral Library (OSSL) :cite:p:`sanderman2020`. Combines interpretable rules with local linear regression.

* **Physics-Informed Features**: Spectral band assignments from Soriano-Disla et al. :cite:p:`soriano2014`, Margenot et al. :cite:p:`margenot2017`, and Tinti et al. :cite:p:`tinti2015` enable chemically meaningful feature extraction.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install soilspec

   # With deep learning support
   pip install soilspec[deep-learning]

   # Development installation
   pip install -e ".[dev]"

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from soilspec.models.traditional import MBLRegressor, CubistRegressor
   from soilspec.preprocessing import SNVTransformer
   from soilspec.features import PhysicsInformedFeatures
   from sklearn.pipeline import Pipeline

   # Create preprocessing + MBL pipeline
   pipeline = Pipeline([
       ('snv', SNVTransformer()),
       ('features', PhysicsInformedFeatures(include_peaks=True)),
       ('model', MBLRegressor(k_neighbors=50))
   ])

   # Fit and predict
   pipeline.fit(spectra_train, soc_train, features__wavenumbers=wavenumbers)
   soc_pred = pipeline.predict(spectra_test)

   # OSSL-style Cubist model
   from soilspec.models.traditional import OSSLCubistPredictor

   cubist = OSSLCubistPredictor(
       n_pcs=120,
       cubist_params={'n_committees': 20, 'neighbors': 5}
   )
   cubist.fit(spectra_train, clay_train)
   clay_pred = cubist.predict(spectra_test)

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/preprocessing
   user_guide/feature_engineering
   user_guide/models
   user_guide/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/preprocessing
   api/features
   api/knowledge
   api/models
   api/interpretation

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. bibliography::
   :all:

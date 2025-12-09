Models API
==========

The models module provides evidence-based statistical and machine learning models for soil spectroscopy. All models are sklearn-compatible and follow fit/predict patterns.

Traditional Models
------------------

Strong, proven models for soil spectroscopy based on peer-reviewed literature.

Memory-Based Learning (MBL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.models.traditional.MBLRegressor
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Memory-Based Learning (MBL), also known as instance-based learning or local modeling, is a non-parametric approach that makes predictions using local calibrations for each new sample :cite:p:`ramirez2013`.

**Key Principles:**

1. **Local Modeling**: Instead of fitting one global model, MBL fits a unique model for each prediction using k-nearest neighbors from the calibration set.

2. **Similarity-Based Selection**: Neighbors are selected based on spectral similarity (cosine, Euclidean, or Mahalanobis distance).

3. **Weighted Regression**: Closer neighbors receive higher weights (Gaussian or inverse distance weighting).

4. **Transfer Learning**: Superior performance when applying models across different instruments, soil types, or geographic regions :cite:p:`ramirez2013b`.

**Mathematical Formulation:**

For a new spectrum :math:`\\mathbf{x}_{\\text{new}}`, MBL:

1. Finds k-nearest neighbors from calibration set :math:`\\mathcal{X}_{\\text{cal}}`:

   .. math::

      \\mathcal{N}_k(\\mathbf{x}_{\\text{new}}) = \\{\\mathbf{x}_i \\in \\mathcal{X}_{\\text{cal}} : \\text{sim}(\\mathbf{x}_{\\text{new}}, \\mathbf{x}_i) \\text{ is among top } k\\}

2. Computes weights based on similarity:

   .. math::

      w_i = \\exp\\left(-\\frac{d_i^2}{2\\sigma^2}\\right) \\quad \\text{(Gaussian weighting)}

   where :math:`d_i` is the distance to neighbor :math:`i`.

3. Fits local model (PLS or Ridge) on weighted neighbors:

   .. math::

      \\hat{y}_{\\text{new}} = f_{\\text{local}}(\\mathbf{x}_{\\text{new}} | \\mathcal{N}_k, \\mathbf{w})

**When to Use MBL:**

* **Transfer learning**: Applying spectral library to new instruments or regions
* **Heterogeneous datasets**: When global models fail due to population diversity
* **Uncertainty estimation**: Local variance provides prediction confidence
* **Large calibration sets**: Works best with >1000 calibration samples

**Performance:**

MBL typically outperforms global PLS by 5-15% in R² when transferring models across instruments or regions :cite:p:`ramirez2013`. For the OSSL library (100k+ spectra), MBL achieves:

* SOC: R² = 0.87-0.92
* Clay: R² = 0.82-0.88
* Total N: R² = 0.85-0.90

**Implementation Notes:**

Our implementation uses sklearn's :class:`sklearn.neighbors.NearestNeighbors` for efficient neighbor search and supports:

* Multiple similarity metrics (cosine, euclidean, mahalanobis)
* Gaussian and inverse-distance weighting
* Local PLS or Ridge regression
* Uncertainty quantification via local variance

**References:**

See :cite:p:`ramirez2013` for original MBL formulation and :cite:p:`ramirez2013b` for optimal calibration set selection strategies.

Cubist Rule-Based Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.models.traditional.CubistRegressor
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

Cubist is a rule-based regression algorithm developed by Ross Quinlan :cite:p:`quinlan1992` that combines decision tree rules with local linear regression models. It is the **standard model** used by the Open Soil Spectral Library (OSSL) for soil property prediction :cite:p:`sanderman2020`.

**Key Principles:**

1. **Rule Generation**: Creates conditional rules that partition the input space into regions.

2. **Local Linear Models**: Each rule has an associated multivariate linear regression model.

3. **Smoothing**: Predictions are smoothed using neighboring instances (instance-based adjustment).

4. **Committees**: Multiple models (like bagging) improve stability and accuracy.

**Algorithm Overview:**

Cubist works in three stages:

1. **Rule Induction**: Build a regression tree, then convert tree paths into rules:

   .. math::

      \\text{Rule}_i: \\text{if } (X_1 > a_1) \\land (X_2 \\leq a_2) \\land \\ldots \\text{ then } \\hat{y} = \\beta_0 + \\beta_1 X_1 + \\ldots

2. **Pruning**: Simplify rules by removing conditions that don't improve performance.

3. **Instance-Based Adjustment**: For prediction :math:`\\mathbf{x}_{\\text{new}}`:

   .. math::

      \\hat{y}_{\\text{final}} = \\hat{y}_{\\text{rule}} + \\frac{1}{k}\\sum_{i=1}^{k} (y_i - \\hat{y}_{\\text{rule},i})

   where :math:`k` neighbors provide local correction to rule-based prediction.

**Why Cubist for Soil Spectroscopy:**

1. **Interpretability**: Rules show which spectral features matter for predictions.

2. **Non-linearity**: Handles complex relationships via piecewise linear models.

3. **OSSL Standard**: Proven on 100,000+ soil spectra across multiple continents :cite:p:`hengl2021`.

4. **Missing Data**: Native handling of missing spectral values.

5. **Extrapolation Control**: Can limit predictions outside training range.

**Performance:**

On OSSL benchmark datasets, Cubist with 20 committees and 5 neighbors achieves:

* SOC (MIR): R² = 0.88-0.93, RMSE = 0.3-0.5%
* Clay (MIR): R² = 0.85-0.90, RMSE = 5-8%
* pH (MIR): R² = 0.75-0.82, RMSE = 0.4-0.6

Typically performs 10-20% better than global PLS and comparable to or better than Random Forest :cite:p:`sanderman2020`.

**Implementation:**

This wrapper uses the `pjaselin/Cubist <https://github.com/pjaselin/Cubist>`_ package, which provides sklearn-compatible interface to Quinlan's original C code.

**Parameters:**

* ``n_committees``: Number of models (1-100). OSSL uses 20.
* ``neighbors``: Instance-based adjustment (0-9). OSSL uses 5.
* ``n_rules``: Maximum rules per committee (None = automatic).
* ``unbiased``: Use unbiased rule selection.
* ``extrapolation``: Limit predictions outside training range (0-100%).

**Example Usage:**

.. code-block:: python

   from soilspec.models.traditional import CubistRegressor

   # OSSL-style Cubist
   cubist = CubistRegressor(n_committees=20, neighbors=5)
   cubist.fit(spectra_train, soc_train)
   soc_pred = cubist.predict(spectra_test)

   # View interpretable rules
   print(cubist.get_rules())

**References:**

See :cite:p:`quinlan1992` for algorithm details, :cite:p:`kuhn2013` for practical guidance, and :cite:p:`sanderman2020` for OSSL implementation.

OSSL Cubist Predictor
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: soilspec.models.traditional.OSSLCubistPredictor
   :members:
   :undoc-members:
   :show-inheritance:

**Scientific Background**

The OSSL Cubist Predictor implements the exact workflow used by the Open Soil Spectral Library for operational soil property prediction :cite:p:`sanderman2020`.

**OSSL Workflow:**

1. **Spectral Compression**: Reduce MIR spectra (typically 1801 wavenumbers) to 120 principal components.

   .. math::

      \\mathbf{X}_{\\text{PCs}} = \\mathbf{X}_{\\text{spectra}} \\mathbf{W}_{\\text{PCA}}

   This achieves ~99% of spectral variance while reducing dimensionality.

2. **Cubist Training**: Train Cubist on PC-compressed spectra with 20 committees and 5 neighbors.

3. **Prediction**: New spectra → PCA transform → Cubist prediction.

**Why PCA Compression:**

1. **Dimensionality Reduction**: 1801 → 120 features reduces overfitting risk.

2. **Noise Reduction**: Lower-variance PCs contain mostly noise.

3. **Computational Efficiency**: Faster training and prediction.

4. **Multicollinearity**: PCs are orthogonal, avoiding correlation issues.

**Performance:**

OSSL reports consistent performance across global datasets:

* SOC: R² = 0.88-0.92 (validation on independent sites)
* Clay: R² = 0.85-0.89
* Sand: R² = 0.83-0.87
* CEC: R² = 0.78-0.85

**Implementation:**

Uses sklearn's :class:`sklearn.decomposition.PCA` for compression and :class:`CubistRegressor` for modeling.

**Example:**

.. code-block:: python

   from soilspec.models.traditional import OSSLCubistPredictor

   # Replicate OSSL workflow
   model = OSSLCubistPredictor(
       n_pcs=120,
       cubist_params={'n_committees': 20, 'neighbors': 5}
   )
   model.fit(mir_spectra, clay_content)
   clay_pred = model.predict(new_spectra)

**References:**

See :cite:p:`sanderman2020` for OSSL methodology and :cite:p:`hengl2021` for continental-scale validation.

Deep Learning Models
--------------------

Modern neural network architectures for soil spectroscopy (optional deep-learning dependencies).

.. note::
   Deep learning models require ``pip install soilspec[deep-learning]``

1D Convolutional Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coming soon. See :cite:p:`tsakiridis2020` and :cite:p:`padarian2019` for applications of CNNs to soil spectroscopy.

References
----------

.. bibliography::
   :filter: docname in docnames

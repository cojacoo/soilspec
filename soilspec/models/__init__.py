"""
Models for soil spectroscopy prediction.

Provides both traditional statistical models (MBL, Cubist) and deep learning
models (1D CNNs) for soil property prediction from spectra.

**Model Selection Guide:**

+-----------------+------------------+------------------------+-------------------+
| Model           | Dataset Size     | Performance            | Interpretability  |
+=================+==================+========================+===================+
| Cubist          | 500-100k         | Excellent (OSSL std)   | High (rules)      |
+-----------------+------------------+------------------------+-------------------+
| MBL             | 1k-100k          | Excellent (transfer)   | Medium (local)    |
+-----------------+------------------+------------------------+-------------------+
| 1D CNN          | >5k              | Good-Excellent         | Low (black box)   |
+-----------------+------------------+------------------------+-------------------+
| PLS             | 100-5k           | Baseline only          | Medium (weights)  |
+-----------------+------------------+------------------------+-------------------+

**Recommendations:**

* **n < 1000**: Use Cubist (interpretable, robust)
* **Transfer learning**: Use MBL (local calibrations)
* **n > 5000, multi-task**: Consider 1D CNN
* **OSSL workflow**: Use Cubist (OSSL standard)

Modules
-------
traditional
    MBLRegressor, CubistRegressor, OSSLCubistPredictor

deep_learning
    SimpleCNN1D, ResNet1D, MultiScaleCNN1D (requires PyTorch)

Examples
--------
**Traditional model:**

>>> from soilspec.models.traditional import CubistRegressor
>>> from soilspec.preprocessing import SNVTransformer
>>> from sklearn.pipeline import Pipeline
>>>
>>> pipeline = Pipeline([
>>>     ('snv', SNVTransformer()),
>>>     ('cubist', CubistRegressor(n_committees=20, neighbors=5))
>>> ])
>>> pipeline.fit(X_train, y_train)

**Deep learning model:**

>>> from soilspec.models.deep_learning import SimpleCNN1D
>>> from soilspec.training import DeepLearningTrainer
>>>
>>> model = SimpleCNN1D(input_size=1801)
>>> trainer = DeepLearningTrainer(model=model, max_epochs=100)
>>> trainer.fit(X_train, y_train, X_val, y_val)
"""

# Traditional models always available
from soilspec.models import traditional

# Deep learning models (optional PyTorch dependency)
from soilspec.models import deep_learning

__all__ = [
    "traditional",
    "deep_learning",
]

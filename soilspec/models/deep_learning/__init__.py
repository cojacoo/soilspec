"""
Deep learning models for soil spectroscopy.

Implements 1D CNNs and other neural network architectures for spectral regression.

**Design Philosophy:**

We use **direct 1D processing** of spectra rather than converting to 2D images
(GADF + ResNet approach from Albinet et al.). 1D CNNs are more natural for
spectral data because:

* Spectra are inherently 1D (absorbance vs wavelength)
* 1D convolutions directly learn spectral features (peaks, bands)
* Fewer parameters → less overfitting with moderate datasets
* Faster training and inference
* More interpretable (filters correspond to chemical features)

**When to Use Deep Learning:**

* **Large datasets** (>5000 samples): CNNs can match/exceed traditional methods
* **Multi-task learning**: Predict multiple properties simultaneously
* **Transfer learning**: Pre-train on large library, fine-tune on local data

**When NOT to Use:**

* **Small datasets** (<1000 samples): Use MBL or Cubist instead
* **Interpretability critical**: Cubist provides rules, CNNs are black boxes
* **Limited compute**: Traditional models train faster

Models
------
**1D CNNs:**
    - SimpleCNN1D: Baseline 3-4 layer CNN for spectral regression
    - ResNet1D: Deep residual network with skip connections
    - MultiScaleCNN1D: Multi-scale feature extraction (narrow + broad peaks)

**Experimental (soilspec.experimental):**
    - GADF transforms for 2D CNN approaches (if needed)

Example Usage
-------------
**Basic 1D CNN:**

>>> from soilspec.models.deep_learning import SimpleCNN1D, create_cnn1d
>>> from soilspec.training import DeepLearningTrainer
>>> from soilspec.datasets import OSSLDataset
>>>
>>> # Load data
>>> ossl = OSSLDataset()
>>> X, y, ids = ossl.load_mir(target='soc')
>>> splits = ossl.split_dataset(X, y, ids, test_size=0.2, val_size=0.1)
>>>
>>> # Create model
>>> model = SimpleCNN1D(input_size=X.shape[1], n_outputs=1)
>>>
>>> # Train
>>> trainer = DeepLearningTrainer(model=model, max_epochs=100, batch_size=64)
>>> trainer.fit(
>>>     splits['X_train'], splits['y_train'],
>>>     splits['X_val'], splits['y_val']
>>> )
>>>
>>> # Evaluate
>>> results = trainer.evaluate(splits['X_test'], splits['y_test'])
>>> print(f"Test R²: {results['r2']:.3f}, RPD: {results['rpd']:.2f}")

**Multi-task learning:**

>>> # Predict SOC, clay, and pH simultaneously
>>> X, y_multi, ids = ossl.load_mir(target=['soc', 'clay', 'ph'])
>>>
>>> model = SimpleCNN1D(input_size=X.shape[1], n_outputs=3)
>>> # y_multi is (n_samples, 3)

**Model comparison:**

>>> from soilspec.models.deep_learning import create_cnn1d
>>>
>>> # Try different architectures
>>> models = {
>>>     'simple': create_cnn1d('simple', input_size=1801),
>>>     'resnet': create_cnn1d('resnet', input_size=1801, n_blocks=3),
>>>     'multiscale': create_cnn1d('multiscale', input_size=1801)
>>> }

Notes
-----
Requires ``pip install soilspec[deep-learning]`` for PyTorch and Lightning.

For most users with <5000 samples, traditional methods (MBL, Cubist) will
perform better and train faster. Deep learning shines with large datasets
and multi-task scenarios.

References
----------
.. [1] Padarian et al. (2019). Using deep learning for digital soil mapping.
       Soil 5(1):79-89.
.. [2] Tsakiridis et al. (2020). Simultaneous prediction of soil properties
       from VNIR-SWIR spectra using 1-D CNN. Geoderma 367:114208.
.. [3] Albinet et al. (2023). Prediction of exchangeable potassium using
       MIR spectroscopy and deep learning (GADF + transfer learning approach).
       Note: We use direct 1D CNNs instead of GADF for simplicity and
       interpretability.
"""

try:
    from soilspec.models.deep_learning.cnn1d import (
        SimpleCNN1D,
        ResNet1D,
        MultiScaleCNN1D,
        create_cnn1d
    )
    __all__ = [
        "SimpleCNN1D",
        "ResNet1D",
        "MultiScaleCNN1D",
        "create_cnn1d",
    ]
except ImportError:
    # PyTorch not available
    def _torch_not_available(*args, **kwargs):
        raise ImportError(
            "PyTorch required for deep learning models. "
            "Install with: pip install soilspec[deep-learning]"
        )

    SimpleCNN1D = _torch_not_available
    ResNet1D = _torch_not_available
    MultiScaleCNN1D = _torch_not_available
    create_cnn1d = _torch_not_available

    __all__ = [
        "SimpleCNN1D",
        "ResNet1D",
        "MultiScaleCNN1D",
        "create_cnn1d",
    ]

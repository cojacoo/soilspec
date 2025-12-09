"""
Training infrastructure for soil spectroscopy models.

Provides trainers for traditional models (MBL, Cubist, PLS) and deep learning
models (1D CNNs), along with callbacks and evaluation metrics following
soil spectroscopy best practices.

Components
----------
**Trainers:**
    - GenericTrainer: For sklearn-compatible models (MBL, Cubist, PLS)
    - DeepLearningTrainer: For PyTorch neural networks (1D CNNs, ResNets)
    - PINNTrainer: DEPRECATED alias for DeepLearningTrainer

**Callbacks (for deep learning):**
    - EarlyStopping: Stop training when validation metric stops improving
    - ModelCheckpoint: Save best models during training
    - LRScheduler: Learning rate scheduling

**Metrics (spectroscopy-specific):**
    - RMSE: Root mean squared error
    - R2Score: Coefficient of determination
    - RPD: Ratio of performance to deviation (Williams 1987)
    - RPIQ: Ratio of performance to inter-quartile range (robust to outliers)
    - MAE: Mean absolute error
    - Bias: Systematic prediction bias

Example Usage
-------------
**Traditional model training:**

>>> from soilspec.training import GenericTrainer
>>> from soilspec.models.traditional import MBLRegressor
>>> from sklearn.pipeline import Pipeline
>>> from soilspec.preprocessing import SNVTransformer
>>>
>>> pipeline = Pipeline([
>>>     ('snv', SNVTransformer()),
>>>     ('mbl', MBLRegressor(k_neighbors=50))
>>> ])
>>>
>>> trainer = GenericTrainer(model=pipeline, cv=10)
>>> trainer.fit(X_train, y_train)
>>> results = trainer.evaluate(X_test, y_test)
>>> print(f"R²: {results['r2']:.3f}, RPD: {results['rpd']:.2f}")

**Deep learning training:**

>>> from soilspec.training import DeepLearningTrainer
>>> from soilspec.training.callbacks import EarlyStopping
>>> # from soilspec.models.deep_learning import CNN1D  # Future implementation
>>>
>>> trainer = DeepLearningTrainer(
>>>     model=my_cnn,
>>>     max_epochs=100,
>>>     callbacks=[EarlyStopping(monitor='val_rmse', patience=15)]
>>> )
>>> trainer.fit(X_train, y_train, X_val, y_val)
"""

# Trainers
from soilspec.training.trainer import GenericTrainer
from soilspec.training.deep_learning_trainer import DeepLearningTrainer

# Backwards compatibility (with deprecation warning)
from soilspec.training.pinn_trainer import PINNTrainer

# Callbacks
from soilspec.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LRScheduler,
    CallbackList
)

# Metrics
from soilspec.training.metrics import (
    RMSE,
    R2Score,
    RPD,
    RPIQ,
    MAE,
    Bias,
    evaluate_model
)

__all__ = [
    # Trainers
    "GenericTrainer",
    "DeepLearningTrainer",
    "PINNTrainer",  # Deprecated
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRScheduler",
    "CallbackList",
    # Metrics
    "RMSE",
    "R2Score",
    "RPD",
    "RPIQ",
    "MAE",
    "Bias",
    "evaluate_model",
]

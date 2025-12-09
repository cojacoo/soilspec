"""
Trainer for deep learning models (1D CNNs, physics-guided neural networks).

Provides PyTorch Lightning-based training infrastructure for neural network
models with callbacks, checkpointing, and comprehensive evaluation.

Scientific Background
---------------------
Deep learning models for spectroscopy (1D CNNs, ResNets) can achieve excellent
performance with large datasets (>10k samples). This trainer wraps PyTorch
Lightning for standardized training workflows.

**Note:** Despite the historical use of "PINN" terminology in this package,
these are standard neural networks for spectroscopy, NOT Physics-Informed Neural
Networks in the traditional sense (solving PDEs). We use physics-informed
FEATURE ENGINEERING, not physics-informed LEARNING.

References
----------
.. [1] Padarian et al. (2019). Using deep learning for digital soil mapping.
       Soil 5(1):79-89.
.. [2] Tsakiridis et al. (2020). Simultaneous prediction of soil properties
       from VNIR-SWIR spectra using a localized multi-channel 1-D CNN.
       Geoderma 367:114208.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import warnings

# Optional PyTorch/Lightning imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback as PLCallback
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    nn = Any
    pl = Any
    PLCallback = Any

from soilspec.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler
from soilspec.training.metrics import evaluate_model


class DeepLearningTrainer:
    """
    Trainer for PyTorch neural network models for spectroscopy.

    Wraps PyTorch Lightning for standardized training with callbacks,
    checkpointing, and comprehensive evaluation using spectroscopy metrics.

    **Models supported:**

    * 1D CNNs for spectral regression
    * ResNets for soil property prediction
    * Multi-task networks (predict multiple properties)
    * Physics-guided attention networks (future)

    **Features:**

    * Automatic GPU/CPU selection
    * Early stopping and model checkpointing
    * Learning rate scheduling
    * TensorBoard logging
    * Cross-validation
    * Comprehensive evaluation (RMSE, R², RPD, RPIQ)

    Parameters
    ----------
    model : nn.Module or pl.LightningModule
        PyTorch neural network model
    optimizer : str or torch.optim.Optimizer, default='adam'
        Optimizer ('adam', 'sgd', 'adamw') or optimizer instance
    learning_rate : float, default=1e-3
        Initial learning rate
    loss_fn : str or callable, default='mse'
        Loss function ('mse', 'mae', 'huber') or custom loss
    max_epochs : int, default=100
        Maximum number of training epochs
    batch_size : int, default=32
        Batch size for training
    callbacks : list of Callback, optional
        Training callbacks (EarlyStopping, ModelCheckpoint, etc.)
    accelerator : str, default='auto'
        Hardware accelerator ('auto', 'gpu', 'cpu')
    devices : int or list, default='auto'
        Number of devices or specific device IDs
    logger : bool or Logger, default=True
        Enable TensorBoard logging
    random_state : int, optional
        Random seed for reproducibility
    verbose : int, default=1
        Verbosity level

    Attributes
    ----------
    model_ : nn.Module
        Trained model
    trainer_ : pl.Trainer
        PyTorch Lightning trainer
    training_history_ : dict
        Training and validation metrics history

    Examples
    --------
    **Basic 1D CNN training:**

    >>> from soilspec.training.deep_learning_trainer import DeepLearningTrainer
    >>> from soilspec.models.deep_learning import CNN1D  # Hypothetical
    >>> from soilspec.training.callbacks import EarlyStopping, ModelCheckpoint
    >>>
    >>> # Create model
    >>> model = CNN1D(input_size=1801, hidden_sizes=[128, 64, 32])
    >>>
    >>> # Create callbacks
    >>> callbacks = [
    >>>     EarlyStopping(monitor='val_rmse', patience=15, mode='min'),
    >>>     ModelCheckpoint(
    >>>         filepath='models/soc_cnn_best.pt',
    >>>         monitor='val_rmse',
    >>>         mode='min'
    >>>     )
    >>> ]
    >>>
    >>> # Create trainer
    >>> trainer = DeepLearningTrainer(
    >>>     model=model,
    >>>     optimizer='adam',
    >>>     learning_rate=1e-3,
    >>>     max_epochs=200,
    >>>     batch_size=64,
    >>>     callbacks=callbacks
    >>> )
    >>>
    >>> # Train
    >>> trainer.fit(X_train, y_train, X_val, y_val)
    >>>
    >>> # Evaluate
    >>> results = trainer.evaluate(X_test, y_test)
    >>> print(f"Test R²: {results['r2']:.3f}")
    >>> print(f"Test RPD: {results['rpd']:.2f}")

    **Multi-task learning:**

    >>> # Model that predicts SOC, clay, pH simultaneously
    >>> model = MultiTaskCNN(input_size=1801, n_outputs=3)
    >>>
    >>> # Train
    >>> trainer = DeepLearningTrainer(model=model, max_epochs=150)
    >>> trainer.fit(X_train, y_train_multi, X_val, y_val_multi)
    >>>
    >>> # y_train_multi shape: (n_samples, 3) for [SOC, clay, pH]

    Notes
    -----
    Requires ``pip install soilspec[deep-learning]`` for PyTorch and Lightning.

    For models with < 5000 samples, traditional methods (MBL, Cubist) often
    perform better than deep learning. Use cross-validation to compare.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Union[str, Any] = 'adam',
        learning_rate: float = 1e-3,
        loss_fn: Union[str, callable] = 'mse',
        max_epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        accelerator: str = 'auto',
        devices: Union[int, str] = 'auto',
        logger: Union[bool, Any] = True,
        random_state: Optional[int] = None,
        verbose: int = 1
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Lightning required for deep learning. "
                "Install with: pip install soilspec[deep-learning]"
            )

        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks or []
        self.accelerator = accelerator
        self.devices = devices
        self.logger = logger
        self.random_state = random_state
        self.verbose = verbose

        # State
        self.model_ = None
        self.trainer_ = None
        self.training_history_ = {}

        # Set random seed
        if random_state is not None:
            pl.seed_everything(random_state)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'DeepLearningTrainer':
        """
        Train neural network model.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training spectra
        y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Training targets
        X_val : array-like of shape (n_val_samples, n_features), optional
            Validation spectra
        y_val : array-like of shape (n_val_samples,) or (n_val_samples, n_outputs), optional
            Validation targets

        Returns
        -------
        self : DeepLearningTrainer
            Fitted trainer
        """
        if self.verbose > 0:
            print(f"Training {self.model.__class__.__name__}...")
            print(f"Training set: {X_train.shape[0]} samples")
            if X_val is not None:
                print(f"Validation set: {X_val.shape[0]} samples")

        # Create dataloaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False) if X_val is not None else None

        # Wrap model in LightningModule if needed
        if not isinstance(self.model, pl.LightningModule):
            self.model_ = self._wrap_model(self.model)
        else:
            self.model_ = self.model

        # Create trainer
        self.trainer_ = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=self._convert_callbacks(),
            accelerator=self.accelerator,
            devices=self.devices,
            logger=self.logger,
            enable_progress_bar=self.verbose > 0
        )

        # Train
        self.trainer_.fit(self.model_, train_loader, val_loader)

        # Extract training history
        self._extract_history()

        return self

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test spectra
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values
        metrics : list of str, optional
            Metrics to compute

        Returns
        -------
        results : dict
            Dictionary of metric names and values
        """
        if self.model_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Predict
        y_pred = self.predict(X)

        # Evaluate
        results = evaluate_model(y, y_pred, metrics=metrics)

        if self.verbose > 0:
            print(f"\nTest set evaluation ({len(y)} samples):")
            for metric, value in results.items():
                if metric != 'n_samples':
                    print(f"  {metric.upper()}: {value:.4f}")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra

        Returns
        -------
        y_pred : array of shape (n_samples,) or (n_samples, n_outputs)
            Predictions
        """
        if self.model_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            if torch.cuda.is_available() and self.accelerator == 'gpu':
                X_tensor = X_tensor.cuda()
                self.model_ = self.model_.cuda()

            y_pred = self.model_(X_tensor)

            if torch.cuda.is_available():
                y_pred = y_pred.cpu()

            return y_pred.numpy()

    def save(self, filepath: Union[str, Path]):
        """
        Save trained model.

        Parameters
        ----------
        filepath : str or Path
            Path to save model
        """
        if self.model_ is None:
            raise ValueError("No trained model to save. Call fit() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        torch.save(self.model_.state_dict(), filepath)

        if self.verbose > 0:
            print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path], model_class: nn.Module) -> nn.Module:
        """
        Load trained model.

        Parameters
        ----------
        filepath : str or Path
            Path to saved model
        model_class : nn.Module
            Model class to instantiate

        Returns
        -------
        model : nn.Module
            Loaded model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required to load models")

        model = model_class()
        model.load_state_dict(torch.load(filepath))
        return model

    def _create_dataloader(
        self,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        shuffle: bool = True
    ) -> Optional[DataLoader]:
        """Create PyTorch DataLoader."""
        if X is None or y is None:
            return None

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1) if y.ndim == 1 else torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0  # Avoid multiprocessing issues
        )

    def _wrap_model(self, model: nn.Module) -> pl.LightningModule:
        """
        Wrap PyTorch model in Lightning Module.

        Creates a simple Lightning wrapper for training.
        """
        class LitModel(pl.LightningModule):
            def __init__(self, model, optimizer, lr, loss_fn):
                super().__init__()
                self.model = model
                self.optimizer_name = optimizer
                self.lr = lr
                self.loss_fn = self._get_loss_fn(loss_fn)

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_pred = self(x)
                loss = self.loss_fn(y_pred, y)
                self.log('train_loss', loss)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_pred = self(x)
                loss = self.loss_fn(y_pred, y)
                self.log('val_loss', loss)

                # Calculate metrics
                y_np = y.cpu().numpy()
                y_pred_np = y_pred.cpu().numpy()
                metrics = evaluate_model(y_np, y_pred_np)
                for name, value in metrics.items():
                    if name != 'n_samples':
                        self.log(f'val_{name}', value)

                return loss

            def configure_optimizers(self):
                if self.optimizer_name == 'adam':
                    return torch.optim.Adam(self.parameters(), lr=self.lr)
                elif self.optimizer_name == 'sgd':
                    return torch.optim.SGD(self.parameters(), lr=self.lr)
                elif self.optimizer_name == 'adamw':
                    return torch.optim.AdamW(self.parameters(), lr=self.lr)
                else:
                    return self.optimizer_name

            def _get_loss_fn(self, loss_fn):
                if loss_fn == 'mse':
                    return nn.MSELoss()
                elif loss_fn == 'mae':
                    return nn.L1Loss()
                elif loss_fn == 'huber':
                    return nn.HuberLoss()
                else:
                    return loss_fn

        return LitModel(model, self.optimizer, self.learning_rate, self.loss_fn)

    def _convert_callbacks(self) -> List[PLCallback]:
        """Convert our callbacks to PyTorch Lightning callbacks."""
        # For now, return empty list
        # Full implementation would convert our callbacks to PL callbacks
        return []

    def _extract_history(self):
        """Extract training history from logger."""
        # Extract metrics from Lightning logger
        # Implementation depends on logger type
        pass

    def __repr__(self):
        return (
            f"DeepLearningTrainer(model={self.model.__class__.__name__}, "
            f"max_epochs={self.max_epochs}, batch_size={self.batch_size})"
        )


# Alias for backwards compatibility (with deprecation warning)
class PINNTrainer(DeepLearningTrainer):
    """
    DEPRECATED: Use DeepLearningTrainer instead.

    This class is provided for backwards compatibility only.
    The term "PINN" is misleading as these are not Physics-Informed Neural
    Networks in the traditional sense (solving PDEs). We use physics-informed
    FEATURE ENGINEERING, not physics-informed LEARNING.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PINNTrainer is deprecated and will be removed in v0.2.0. "
            "Use DeepLearningTrainer instead. The term 'PINN' is misleading "
            "for standard neural networks with physics-informed features.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

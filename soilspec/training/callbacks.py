"""
Training callbacks for deep learning models.

Callbacks allow custom actions during model training, such as early stopping,
model checkpointing, and learning rate scheduling.

Scientific Background
---------------------
Callbacks are standard in deep learning training to:
* Prevent overfitting (early stopping)
* Save best models (checkpointing)
* Improve convergence (learning rate scheduling)

Note: These callbacks are primarily for deep learning models (PyTorch/Lightning).
Traditional sklearn-compatible models (MBL, Cubist) don't typically use callbacks.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any
import warnings


class Callback:
    """
    Base class for callbacks.

    All custom callbacks should inherit from this class and override
    the relevant methods.
    """

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.

    Scientific Background
    ---------------------
    Early stopping prevents overfitting by monitoring validation performance
    and stopping when it stops improving. This is a form of regularization.

    **Principle:**

    Training continues while validation metric improves. Once validation metric
    stops improving for `patience` epochs, training stops and best model is restored.

    **Advantages:**

    * Prevents overfitting without manual epoch tuning
    * Automatic regularization
    * Saves training time

    **Best Practices:**

    * Monitor validation loss or metric (not training)
    * Patience should be ~10-20% of max epochs
    * Restore best weights when stopping
    * Use with ModelCheckpoint to save best model

    Parameters
    ----------
    monitor : str, default='val_loss'
        Metric to monitor (e.g., 'val_loss', 'val_r2', 'val_rmse')
    patience : int, default=10
        Number of epochs with no improvement before stopping
    min_delta : float, default=0.0
        Minimum change to qualify as improvement
    mode : str, default='min'
        'min' if metric should decrease (loss), 'max' if increase (R²)
    restore_best_weights : bool, default=True
        Whether to restore model weights from epoch with best metric
    verbose : int, default=1
        Verbosity mode (0=silent, 1=progress messages)

    Attributes
    ----------
    best_value : float
        Best observed value of monitored metric
    best_epoch : int
        Epoch with best metric value
    wait : int
        Number of epochs since last improvement
    stopped_epoch : int
        Epoch at which training was stopped

    Examples
    --------
    >>> from soilspec.training.callbacks import EarlyStopping
    >>>
    >>> # For PyTorch Lightning trainer
    >>> early_stop = EarlyStopping(
    >>>     monitor='val_rmse',
    >>>     patience=15,
    >>>     mode='min',
    >>>     min_delta=0.001
    >>> )
    >>>
    >>> # Add to trainer
    >>> trainer = pl.Trainer(callbacks=[early_stop], max_epochs=200)
    >>>
    >>> # Training will stop if val_rmse doesn't improve for 15 epochs
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: int = 1
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # Internal state
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        # Comparison operator
        if mode == 'min':
            self.monitor_op = np.less
            self.delta = -min_delta
        elif mode == 'max':
            self.monitor_op = np.greater
            self.delta = min_delta
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Reset state at start of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if should stop after each epoch."""
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                f"Early stopping conditioned on metric `{self.monitor}` "
                f"which is not available. Available metrics: {list(logs.keys())}",
                RuntimeWarning
            )
            return

        # Check if improvement
        if self.monitor_op(current - self.delta, self.best_value):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights and hasattr(self, '_model'):
                # Store model weights (implementation specific to framework)
                self.best_weights = self._get_model_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose > 0:
                    print(f"\nEpoch {epoch}: early stopping")
                    print(f"Restoring model weights from epoch {self.best_epoch}")
                if self.restore_best_weights and self.best_weights is not None:
                    self._set_model_weights(self.best_weights)
                # Signal to stop training
                return True  # Framework-specific stopping

    def _get_model_weights(self):
        """Get model weights (framework-specific)."""
        # To be implemented based on framework (PyTorch, etc.)
        pass

    def _set_model_weights(self, weights):
        """Set model weights (framework-specific)."""
        # To be implemented based on framework
        pass

    def __repr__(self):
        return (
            f"EarlyStopping(monitor='{self.monitor}', patience={self.patience}, "
            f"mode='{self.mode}')"
        )


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Scientific Background
    ---------------------
    Model checkpointing saves the best model (or models from each epoch)
    during training. This allows recovery from crashes and ensures best
    model is preserved even if training continues past optimal point.

    **Strategies:**

    * **Best only**: Save only when metric improves (disk efficient)
    * **All epochs**: Save every epoch (useful for analysis)
    * **Period**: Save every N epochs

    **Best Practices:**

    * Monitor validation metric, not training
    * Save with informative filenames (include metric value)
    * Keep only top K models to save disk space

    Parameters
    ----------
    filepath : str or Path
        Path to save model files. Can include formatting:
        * {epoch}: current epoch number
        * {metric}: current metric value
        Example: 'models/soc_epoch{epoch:02d}_r2{val_r2:.3f}.pt'
    monitor : str, default='val_loss'
        Metric to monitor for determining best model
    mode : str, default='min'
        'min' if metric should decrease, 'max' if increase
    save_best_only : bool, default=True
        Only save when metric improves
    save_weights_only : bool, default=False
        Save only weights (True) or full model (False)
    period : int, default=1
        Save every N epochs
    verbose : int, default=1
        Verbosity mode

    Examples
    --------
    >>> from soilspec.training.callbacks import ModelCheckpoint
    >>>
    >>> # Save best model only
    >>> checkpoint = ModelCheckpoint(
    >>>     filepath='models/best_soc_model.pt',
    >>>     monitor='val_rmse',
    >>>     mode='min',
    >>>     save_best_only=True
    >>> )
    >>>
    >>> # Save every epoch with metric in filename
    >>> checkpoint_all = ModelCheckpoint(
    >>>     filepath='models/soc_epoch{epoch:02d}_r2{val_r2:.3f}.pt',
    >>>     save_best_only=False,
    >>>     period=5  # Save every 5 epochs
    >>> )
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        period: int = 1,
        verbose: int = 1
    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.verbose = verbose

        # State
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.epochs_since_last_save = 0

        # Comparison operator
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save model checkpoint if conditions met."""
        if logs is None:
            return

        self.epochs_since_last_save += 1

        # Check if should save this epoch
        if self.epochs_since_last_save < self.period:
            return

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                f"ModelCheckpoint conditioned on metric `{self.monitor}` "
                f"which is not available. Available metrics: {list(logs.keys())}",
                RuntimeWarning
            )
            return

        # Format filepath with epoch and metrics
        filepath = str(self.filepath).format(epoch=epoch, **logs)

        # Check if should save
        if self.save_best_only:
            if self.monitor_op(current, self.best_value):
                self.best_value = current
                self._save_model(filepath, epoch, logs)
                self.epochs_since_last_save = 0
        else:
            self._save_model(filepath, epoch, logs)
            self.epochs_since_last_save = 0

    def _save_model(self, filepath: str, epoch: int, logs: Dict):
        """Save model (framework-specific)."""
        if self.verbose > 0:
            print(f"\nEpoch {epoch}: saving model to {filepath}")
            if self.monitor in logs:
                print(f"{self.monitor}: {logs[self.monitor]:.4f}")

        # To be implemented based on framework (PyTorch, etc.)
        # Example for PyTorch:
        # if hasattr(self, '_model'):
        #     if self.save_weights_only:
        #         torch.save(self._model.state_dict(), filepath)
        #     else:
        #         torch.save(self._model, filepath)

    def __repr__(self):
        return (
            f"ModelCheckpoint(filepath='{self.filepath}', monitor='{self.monitor}', "
            f"mode='{self.mode}', save_best_only={self.save_best_only})"
        )


class LRScheduler(Callback):
    """
    Adjust learning rate during training.

    Scientific Background
    ---------------------
    Learning rate scheduling improves convergence by adjusting the learning
    rate during training. Common strategies:

    **ReduceLROnPlateau:**

    Reduce LR when validation metric stops improving. Allows model to fine-tune
    when stuck in plateau.

    .. math::

        \\text{lr}_{\\text{new}} = \\text{lr}_{\\text{old}} \\times \\text{factor}

    **StepLR:**

    Reduce LR by factor every N epochs. Simple but effective.

    **CosineAnnealing:**

    LR follows cosine curve, gradually decreasing:

    .. math::

        \\text{lr}_t = \\text{lr}_{\\text{min}} + \\frac{1}{2}(\\text{lr}_{\\text{max}} - \\text{lr}_{\\text{min}})(1 + \\cos(\\frac{t\\pi}{T}))

    **ExponentialLR:**

    Exponential decay:

    .. math::

        \\text{lr}_t = \\text{lr}_0 \\times \\gamma^t

    **Best Practices:**

    * Start with high LR (1e-3 to 1e-2)
    * Reduce by factor of 2-10 when plateaus
    * Use ReduceLROnPlateau for automatic scheduling
    * Monitor validation loss, not training loss

    Parameters
    ----------
    scheduler_type : str, default='reduce_on_plateau'
        Type of scheduler: 'reduce_on_plateau', 'step', 'cosine', 'exponential'
    monitor : str, default='val_loss'
        Metric to monitor (for reduce_on_plateau)
    mode : str, default='min'
        'min' if metric should decrease, 'max' if increase
    factor : float, default=0.1
        Factor to reduce LR by
    patience : int, default=10
        Epochs to wait before reducing LR (reduce_on_plateau)
    min_lr : float, default=1e-6
        Minimum learning rate
    verbose : int, default=1
        Verbosity mode

    Examples
    --------
    >>> from soilspec.training.callbacks import LRScheduler
    >>>
    >>> # Reduce LR when validation loss plateaus
    >>> lr_scheduler = LRScheduler(
    >>>     scheduler_type='reduce_on_plateau',
    >>>     monitor='val_loss',
    >>>     factor=0.5,  # Halve LR
    >>>     patience=10,  # After 10 epochs without improvement
    >>>     min_lr=1e-6
    >>> )
    >>>
    >>> # Cosine annealing
    >>> lr_cosine = LRScheduler(
    >>>     scheduler_type='cosine',
    >>>     min_lr=1e-6
    >>> )
    """

    def __init__(
        self,
        scheduler_type: str = 'reduce_on_plateau',
        monitor: str = 'val_loss',
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        verbose: int = 1
    ):
        super().__init__()
        self.scheduler_type = scheduler_type
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        # State for reduce_on_plateau
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.wait = 0

        # Comparison operator
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Adjust learning rate if needed."""
        if logs is None:
            return

        if self.scheduler_type == 'reduce_on_plateau':
            self._reduce_on_plateau(epoch, logs)
        elif self.scheduler_type == 'step':
            self._step_lr(epoch)
        elif self.scheduler_type == 'cosine':
            self._cosine_lr(epoch)
        elif self.scheduler_type == 'exponential':
            self._exponential_lr(epoch)

    def _reduce_on_plateau(self, epoch: int, logs: Dict):
        """Reduce LR when metric plateaus."""
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                f"LRScheduler conditioned on metric `{self.monitor}` "
                f"which is not available.",
                RuntimeWarning
            )
            return

        # Check if improvement
        if self.monitor_op(current, self.best_value):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce LR
                self._reduce_lr(epoch)
                self.wait = 0

    def _reduce_lr(self, epoch: int):
        """Reduce learning rate by factor."""
        if hasattr(self, '_optimizer'):
            old_lr = self._get_lr()
            new_lr = max(old_lr * self.factor, self.min_lr)
            self._set_lr(new_lr)

            if self.verbose > 0:
                print(f"\nEpoch {epoch}: reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")

    def _step_lr(self, epoch: int):
        """Step LR decay (framework-specific)."""
        pass

    def _cosine_lr(self, epoch: int):
        """Cosine annealing (framework-specific)."""
        pass

    def _exponential_lr(self, epoch: int):
        """Exponential decay (framework-specific)."""
        pass

    def _get_lr(self) -> float:
        """Get current learning rate (framework-specific)."""
        # To be implemented based on optimizer
        return 1e-3

    def _set_lr(self, lr: float):
        """Set learning rate (framework-specific)."""
        # To be implemented based on optimizer
        pass

    def __repr__(self):
        return (
            f"LRScheduler(type='{self.scheduler_type}', monitor='{self.monitor}', "
            f"factor={self.factor})"
        )


class CallbackList:
    """
    Container for managing multiple callbacks.

    Examples
    --------
    >>> from soilspec.training.callbacks import CallbackList, EarlyStopping, ModelCheckpoint
    >>>
    >>> callbacks = CallbackList([
    >>>     EarlyStopping(monitor='val_rmse', patience=15),
    >>>     ModelCheckpoint(filepath='models/best.pt', monitor='val_rmse'),
    >>>     LRScheduler(scheduler_type='reduce_on_plateau')
    >>> ])
    >>>
    >>> # In training loop:
    >>> callbacks.on_epoch_end(epoch, logs={'val_rmse': 0.35, 'val_r2': 0.89})
    """

    def __init__(self, callbacks: Optional[list] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        """Add callback to list."""
        self.callbacks.append(callback)

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_end for all callbacks."""
        stop_training = False
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, logs)
            if result is True:  # Callback signals to stop
                stop_training = True
        return stop_training

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def __len__(self):
        return len(self.callbacks)

    def __repr__(self):
        return f"CallbackList({[repr(cb) for cb in self.callbacks]})"

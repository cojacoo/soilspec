"""
Training infrastructure for spectral models.

Provides generic trainers, PINN-specific training loops,
callbacks, and evaluation metrics.
"""

from soilspec.training.trainer import GenericTrainer
from soilspec.training.pinn_trainer import PINNTrainer
from soilspec.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler
from soilspec.training.metrics import RMSE, R2Score, RPD, RPIQ

__all__ = [
    "GenericTrainer",
    "PINNTrainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRScheduler",
    "RMSE",
    "R2Score",
    "RPD",
    "RPIQ",
]

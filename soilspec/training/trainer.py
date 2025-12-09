"""
Generic trainer for traditional (sklearn-compatible) spectroscopy models.

Provides unified interface for training and evaluating MBL, Cubist, PLS,
and other sklearn-compatible models with cross-validation, hyperparameter
tuning, and comprehensive evaluation.

Scientific Background
---------------------
Traditional spectroscopy models (MBL, Cubist, PLS) are typically trained
using cross-validation with grid or random search for hyperparameters.
This trainer wraps sklearn functionality with spectroscopy-specific
evaluation metrics (RPD, RPIQ) and best practices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from sklearn.model_selection import (
    cross_validate, GridSearchCV, RandomizedSearchCV,
    KFold, StratifiedKFold
)
from sklearn.base import clone, BaseEstimator
from sklearn.pipeline import Pipeline
import joblib
import json

from soilspec.training.metrics import evaluate_model, RMSE, R2Score, RPD, RPIQ


class GenericTrainer:
    """
    Trainer for sklearn-compatible spectroscopy models.

    Provides unified interface for training MBL, Cubist, PLS, and other
    traditional models with:

    * Cross-validation
    * Hyperparameter tuning (grid/random search)
    * Model evaluation with spectroscopy metrics (RPD, RPIQ)
    * Model persistence (save/load)
    * Results logging

    **Workflow:**

    1. Create trainer with model/pipeline
    2. Optionally tune hyperparameters
    3. Train with cross-validation
    4. Evaluate on test set
    5. Save best model

    Parameters
    ----------
    model : estimator or Pipeline
        Sklearn-compatible model or pipeline
    param_grid : dict, optional
        Parameter grid for hyperparameter tuning
    search_type : str, default='grid'
        'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    cv : int or cross-validator, default=5
        Number of CV folds or cross-validator instance
    scoring : str or callable, default='neg_root_mean_squared_error'
        Scoring metric for hyperparameter selection
    random_state : int, optional
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = use all cores)
    verbose : int, default=1
        Verbosity level

    Attributes
    ----------
    best_model_ : estimator
        Best fitted model after training
    cv_results_ : dict
        Cross-validation results
    test_results_ : dict
        Test set evaluation results
    training_history_ : list
        History of training metrics

    Examples
    --------
    **Basic training:**

    >>> from soilspec.training.trainer import GenericTrainer
    >>> from soilspec.models.traditional import MBLRegressor
    >>> from soilspec.preprocessing import SNVTransformer
    >>> from sklearn.pipeline import Pipeline
    >>>
    >>> # Create pipeline
    >>> pipeline = Pipeline([
    >>>     ('snv', SNVTransformer()),
    >>>     ('mbl', MBLRegressor(k_neighbors=50))
    >>> ])
    >>>
    >>> # Create trainer
    >>> trainer = GenericTrainer(
    >>>     model=pipeline,
    >>>     cv=10,
    >>>     random_state=42
    >>> )
    >>>
    >>> # Train
    >>> trainer.fit(X_train, y_train)
    >>>
    >>> # Evaluate
    >>> results = trainer.evaluate(X_test, y_test)
    >>> print(f"Test R²: {results['r2']:.3f}")
    >>> print(f"Test RPD: {results['rpd']:.2f}")

    **Hyperparameter tuning:**

    >>> # Define parameter grid
    >>> param_grid = {
    >>>     'mbl__k_neighbors': [30, 50, 100],
    >>>     'mbl__similarity_metric': ['cosine', 'euclidean'],
    >>>     'mbl__n_components': [5, 10, 15]
    >>> }
    >>>
    >>> # Train with hyperparameter tuning
    >>> trainer = GenericTrainer(
    >>>     model=pipeline,
    >>>     param_grid=param_grid,
    >>>     search_type='grid',
    >>>     cv=5
    >>> )
    >>>
    >>> trainer.fit(X_train, y_train)
    >>> print(f"Best parameters: {trainer.best_params_}")
    >>> print(f"Best CV score: {trainer.best_score_:.3f}")

    **OSSL-style workflow:**

    >>> from soilspec.models.traditional import OSSLCubistPredictor
    >>>
    >>> param_grid = {
    >>>     'n_pcs': [100, 120, 150],
    >>>     'cubist_params': [
    >>>         {'n_committees': 20, 'neighbors': 5},
    >>>         {'n_committees': 30, 'neighbors': 9}
    >>>     ]
    >>> }
    >>>
    >>> trainer = GenericTrainer(
    >>>     model=OSSLCubistPredictor(),
    >>>     param_grid=param_grid
    >>> )
    >>>
    >>> trainer.fit(X_train, y_train)
    >>> trainer.save('models/ossl_soc_cubist.pkl')
    """

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: Optional[Dict] = None,
        search_type: str = 'grid',
        cv: Union[int, Any] = 5,
        scoring: Union[str, callable] = 'neg_root_mean_squared_error',
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        self.model = model
        self.param_grid = param_grid
        self.search_type = search_type
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # State
        self.best_model_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.test_results_ = None
        self.training_history_ = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **fit_params
    ) -> 'GenericTrainer':
        """
        Train model with optional hyperparameter tuning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training spectra
        y : array-like of shape (n_samples,)
            Target values
        **fit_params : dict
            Additional parameters passed to model.fit()
            (e.g., features__wavenumbers for pipelines)

        Returns
        -------
        self : GenericTrainer
            Fitted trainer
        """
        if self.verbose > 0:
            print(f"Training {self.model.__class__.__name__}...")
            print(f"Training set: {X.shape[0]} samples, {X.shape[1]} features")

        # Hyperparameter tuning if param_grid provided
        if self.param_grid is not None:
            if self.verbose > 0:
                print(f"\nPerforming {self.search_type} search...")

            if self.search_type == 'grid':
                search = GridSearchCV(
                    self.model,
                    param_grid=self.param_grid,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    return_train_score=True
                )
            elif self.search_type == 'random':
                search = RandomizedSearchCV(
                    self.model,
                    param_distributions=self.param_grid,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    return_train_score=True
                )
            else:
                raise ValueError(f"Unknown search_type: {self.search_type}")

            search.fit(X, y, **fit_params)

            self.best_model_ = search.best_estimator_
            self.best_params_ = search.best_params_
            self.best_score_ = search.best_score_
            self.cv_results_ = search.cv_results_

            if self.verbose > 0:
                print(f"\nBest parameters: {self.best_params_}")
                print(f"Best CV score: {self.best_score_:.4f}")

        else:
            # No hyperparameter tuning - just cross-validate
            if self.verbose > 0:
                print(f"\nPerforming {self.cv}-fold cross-validation...")

            cv_scores = cross_validate(
                self.model,
                X, y,
                cv=self.cv,
                scoring={
                    'rmse': 'neg_root_mean_squared_error',
                    'r2': 'r2',
                    'mae': 'neg_mean_absolute_error'
                },
                n_jobs=self.n_jobs,
                return_train_score=True,
                fit_params=fit_params
            )

            self.cv_results_ = cv_scores

            # Train on full training set
            self.best_model_ = clone(self.model)
            self.best_model_.fit(X, y, **fit_params)

            if self.verbose > 0:
                rmse_mean = -cv_scores['test_rmse'].mean()
                rmse_std = cv_scores['test_rmse'].std()
                r2_mean = cv_scores['test_r2'].mean()
                r2_std = cv_scores['test_r2'].std()

                print(f"\nCross-validation results ({self.cv} folds):")
                print(f"  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
                print(f"  R²: {r2_mean:.4f} ± {r2_std:.4f}")

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
        y : array-like of shape (n_samples,)
            True values
        metrics : list of str, optional
            Metrics to compute. Default: ['rmse', 'r2', 'rpd', 'rpiq', 'bias', 'mae']

        Returns
        -------
        results : dict
            Dictionary of metric names and values
        """
        if self.best_model_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        if metrics is None:
            metrics = ['rmse', 'r2', 'rpd', 'rpiq', 'bias', 'mae']

        # Predict
        y_pred = self.best_model_.predict(X)

        # Evaluate
        results = evaluate_model(y, y_pred, metrics=metrics)

        # Store results
        self.test_results_ = results

        if self.verbose > 0:
            print(f"\nTest set evaluation ({len(y)} samples):")
            for metric, value in results.items():
                if metric != 'n_samples':
                    print(f"  {metric.upper()}: {value:.4f}")

            # Interpret RPD
            rpd = results.get('rpd', 0)
            if rpd > 2.5:
                print("  → Excellent predictions (quantitative use)")
            elif rpd > 2.0:
                print("  → Good predictions (quantitative screening)")
            elif rpd > 1.4:
                print("  → Fair predictions (qualitative screening)")
            else:
                print("  → Poor predictions (not recommended)")

        return results

    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra
        return_uncertainty : bool, default=False
            If True and model supports it, return uncertainty estimates

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predictions
        y_std : array of shape (n_samples,), optional
            Prediction uncertainties (if return_uncertainty=True and supported)
        """
        if self.best_model_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        if return_uncertainty and hasattr(self.best_model_, 'predict_with_uncertainty'):
            # MBL supports uncertainty
            return self.best_model_.predict_with_uncertainty(X)
        else:
            return self.best_model_.predict(X)

    def save(self, filepath: Union[str, Path]):
        """
        Save trained model to disk.

        Parameters
        ----------
        filepath : str or Path
            Path to save model

        Examples
        --------
        >>> trainer.fit(X_train, y_train)
        >>> trainer.save('models/soc_mbl_model.pkl')
        """
        if self.best_model_ is None:
            raise ValueError("No trained model to save. Call fit() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.best_model_, filepath)

        # Save metadata
        metadata = {
            'model_class': self.best_model_.__class__.__name__,
            'best_params': self.best_params_,
            'cv_score': self.best_score_,
            'test_results': self.test_results_
        }

        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbose > 0:
            print(f"\nModel saved to {filepath}")
            print(f"Metadata saved to {metadata_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> BaseEstimator:
        """
        Load trained model from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to saved model

        Returns
        -------
        model : estimator
            Loaded model

        Examples
        --------
        >>> model = GenericTrainer.load('models/soc_mbl_model.pkl')
        >>> y_pred = model.predict(X_new)
        """
        filepath = Path(filepath)
        model = joblib.load(filepath)
        return model

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances if model supports it.

        Returns
        -------
        importance : array of shape (n_features,), optional
            Feature importances, or None if not supported

        Examples
        --------
        >>> trainer.fit(X_train, y_train)
        >>> importance = trainer.get_feature_importance()
        >>> if importance is not None:
        >>>     # Plot or analyze importance
        >>>     top_features = np.argsort(importance)[-10:]
        """
        if self.best_model_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Check for feature_importances_ attribute
        if hasattr(self.best_model_, 'feature_importances_'):
            return self.best_model_.feature_importances_

        # Check for coef_ attribute (linear models)
        if hasattr(self.best_model_, 'coef_'):
            return np.abs(self.best_model_.coef_)

        # Check in pipeline
        if isinstance(self.best_model_, Pipeline):
            final_estimator = self.best_model_.steps[-1][1]
            if hasattr(final_estimator, 'feature_importances_'):
                return final_estimator.feature_importances_
            if hasattr(final_estimator, 'coef_'):
                return np.abs(final_estimator.coef_)

        return None

    def plot_predictions(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Plot predicted vs true values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra
        y_true : array-like of shape (n_samples,)
            True values
        save_path : str or Path, optional
            Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting")
            return

        y_pred = self.predict(X)
        results = evaluate_model(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)

        # 1:1 line
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

        # Labels and metrics
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Predicted vs True Values')

        # Add metrics text
        metrics_text = f"R² = {results['r2']:.3f}\n"
        metrics_text += f"RMSE = {results['rmse']:.3f}\n"
        metrics_text += f"RPD = {results['rpd']:.2f}\n"
        metrics_text += f"RPIQ = {results['rpiq']:.2f}"

        ax.text(0.05, 0.95, metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose > 0:
                print(f"Plot saved to {save_path}")

        plt.show()

    def __repr__(self):
        return (
            f"GenericTrainer(model={self.model.__class__.__name__}, "
            f"cv={self.cv}, n_jobs={self.n_jobs})"
        )

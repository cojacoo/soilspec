"""
Cubist rule-based regression wrapper.

Wraps pjaselin/Cubist package for OSSL-compatible soil property prediction.
Cubist is the standard model used by Open Soil Spectral Library (OSSL).

Strong model - rule-based + local linear regression outperforms PLS/RF.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional
import warnings

# Try to import cubist
try:
    from cubist import Cubist
    CUBIST_AVAILABLE = True
except ImportError:
    CUBIST_AVAILABLE = False
    warnings.warn(
        "cubist package not available. Install with: pip install cubist\n"
        "Cubist is the standard model used by OSSL for soil spectroscopy."
    )


class CubistRegressor(BaseEstimator, RegressorMixin):
    """
    Cubist rule-based regression for soil spectroscopy.

    Wrapper around pjaselin/Cubist package, which is sklearn-compatible.
    This is the STANDARD MODEL used by Open Soil Spectral Library (OSSL)
    for soil property prediction from MIR spectra.

    Cubist generates conditional rules with corresponding linear regressors:
    - Interpretable (can see which rules fire)
    - Handles non-linear relationships via piecewise linear models
    - Often outperforms global linear models (PLS, Ridge)
    - Native support for missing values

    Args:
        n_rules: Maximum number of rules (None = automatic)
        n_committees: Number of committee models (ensemble size)
        neighbors: Number of neighbors for instance-based adjustment (0-9)
        unbiased: Use unbiased rules
        extrapolation: Percentage for extrapolation (0-100)
        sample: Percentage of data to sample for each committee
        seed: Random seed

    Example:
        >>> from soilspec.models.traditional import CubistRegressor
        >>> # OSSL-style Cubist model
        >>> cubist = CubistRegressor(n_committees=20, neighbors=5)
        >>> cubist.fit(spectra, soil_properties)
        >>> predictions = cubist.predict(new_spectra)
        >>> # View rules
        >>> print(cubist.get_rules())

    Reference:
        Quinlan, J.R. (1992). Learning with Continuous Classes.
        Proceedings of the 5th Australian Joint Conference on AI.

        OSSL: https://github.com/soilspectroscopy/ossl-models
    """

    def __init__(
        self,
        n_rules: Optional[int] = None,
        n_committees: int = 1,
        neighbors: int = 0,
        unbiased: bool = False,
        extrapolation: float = 5.0,
        sample: float = 100.0,
        seed: Optional[int] = None
    ):
        """
        Initialize Cubist regressor.

        Args:
            n_rules: Maximum number of rules (None = automatic)
            n_committees: Number of committee models (like bagging)
            neighbors: Instance-based correction (0-9)
            unbiased: Use unbiased rules
            extrapolation: Extrapolation percentage
            sample: Sampling percentage for committees
            seed: Random seed
        """
        if not CUBIST_AVAILABLE:
            raise ImportError(
                "cubist package is required for CubistRegressor. "
                "Install with: pip install cubist"
            )

        self.n_rules = n_rules
        self.n_committees = n_committees
        self.neighbors = neighbors
        self.unbiased = unbiased
        self.extrapolation = extrapolation
        self.sample = sample
        self.seed = seed

        self.model_ = None

    def fit(self, X, y):
        """
        Fit Cubist model.

        Args:
            X: Training spectra (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Handle 2D y
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        # Create Cubist model
        self.model_ = Cubist(
            n_rules=self.n_rules,
            n_committees=self.n_committees,
            neighbors=self.neighbors,
            unbiased=self.unbiased,
            extrapolation=self.extrapolation,
            sample=self.sample,
            seed=self.seed
        )

        # Fit model
        self.model_.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict using Cubist model.

        Args:
            X: Prediction spectra (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X)
        return self.model_.predict(X)

    def score(self, X, y):
        """
        Return R² score.

        Args:
            X: Test spectra
            y: True values

        Returns:
            R² score
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before scoring")

        return self.model_.score(X, y)

    def get_rules(self):
        """
        Get text representation of rules.

        Returns:
            String with rule descriptions
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")

        # Try to get rules from model
        if hasattr(self.model_, 'rules_'):
            return self.model_.rules_
        elif hasattr(self.model_, 'get_rules'):
            return self.model_.get_rules()
        else:
            return "Rules not available in this version of cubist package"


class OSSLCubistPredictor(BaseEstimator, RegressorMixin):
    """
    OSSL-compatible Cubist predictor.

    Follows OSSL workflow:
    1. Compress spectra to first 120 PCs
    2. Train Cubist on PCs
    3. Predict soil properties

    This matches the OSSL pre-trained model approach.

    Args:
        n_pcs: Number of principal components (OSSL uses 120)
        cubist_params: Parameters for CubistRegressor

    Example:
        >>> from soilspec.models.traditional import OSSLCubistPredictor
        >>> # OSSL-style model
        >>> model = OSSLCubistPredictor(n_pcs=120, cubist_params={
        ...     'n_committees': 20,
        ...     'neighbors': 5
        ... })
        >>> model.fit(spectra, soil_properties)
        >>> predictions = model.predict(new_spectra)

    Reference:
        https://github.com/soilspectroscopy/ossl-models
    """

    def __init__(
        self,
        n_pcs: int = 120,
        cubist_params: Optional[dict] = None
    ):
        """
        Initialize OSSL Cubist predictor.

        Args:
            n_pcs: Number of PCs for compression
            cubist_params: Parameters for Cubist
        """
        self.n_pcs = n_pcs
        self.cubist_params = cubist_params or {}

        self.pca_ = None
        self.cubist_ = None

    def fit(self, X, y):
        """
        Fit PCA + Cubist model.

        Args:
            X: Training spectra (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            self
        """
        from sklearn.decomposition import PCA

        X = np.asarray(X)
        y = np.asarray(y)

        # Fit PCA
        self.pca_ = PCA(n_components=self.n_pcs)
        X_pcs = self.pca_.fit_transform(X)

        # Fit Cubist on PCs
        self.cubist_ = CubistRegressor(**self.cubist_params)
        self.cubist_.fit(X_pcs, y)

        return self

    def predict(self, X):
        """
        Predict via PCA compression + Cubist.

        Args:
            X: Prediction spectra (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if self.pca_ is None or self.cubist_ is None:
            raise ValueError("Model must be fitted first")

        X = np.asarray(X)

        # Transform to PCs
        X_pcs = self.pca_.transform(X)

        # Predict with Cubist
        return self.cubist_.predict(X_pcs)

    def score(self, X, y):
        """Return R² score."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

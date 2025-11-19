"""
Memory-Based Learning (MBL) for spectroscopy.

Based on saxSSL application and resemble R package methodology.
Local modeling approach where predictions use only k-nearest neighbors.

Strong model - outperforms PLS in many scenarios, especially for:
- Instrument transfer
- Non-stationary spectral characteristics
- Limited training data per local region
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import pairwise_distances
from typing import Optional, Literal
import warnings


class MBLRegressor(BaseEstimator, RegressorMixin):
    """
    Memory-Based Learning regressor for spectroscopy.

    Makes predictions using k-nearest neighbors in spectral space,
    fitting local models (PLS, Ridge, or WLS) on neighbors only.

    This is a STRONG model that often outperforms global PLS/Ridge,
    especially for:
    - Non-linear spectral-property relationships
    - Transferring between instruments/labs
    - Handling spectral drift

    Args:
        k_neighbors: Number of nearest neighbors to use (default: 50)
        similarity_metric: Distance metric ('cosine', 'euclidean', 'mahalanobis')
        weighting: Neighbor weighting ('uniform', 'distance', 'gaussian')
        local_model: Local regression model ('pls', 'ridge', 'wls')
        n_components: Number of components for PLS local model

    Example:
        >>> from soilspec.models.traditional import MBLRegressor
        >>> # MBL with local PLS models
        >>> mbl = MBLRegressor(k_neighbors=50, local_model='pls', n_components=5)
        >>> mbl.fit(X_calibration, y_calibration)
        >>> predictions = mbl.predict(X_new)
        >>> # Get uncertainty from local variance
        >>> predictions, uncertainties = mbl.predict_with_uncertainty(X_new)

    Reference:
        Ramirez-Lopez et al. (2013). "The spectrum-based learner: A new local
        approach for modeling soil vis-NIR spectra of complex datasets." Geoderma.
    """

    def __init__(
        self,
        k_neighbors: int = 50,
        similarity_metric: Literal['cosine', 'euclidean', 'mahalanobis'] = 'cosine',
        weighting: Literal['uniform', 'distance', 'gaussian'] = 'gaussian',
        local_model: Literal['pls', 'ridge', 'wls'] = 'pls',
        n_components: int = 10
    ):
        """
        Initialize MBL regressor.

        Args:
            k_neighbors: Number of nearest neighbors
            similarity_metric: Distance/similarity metric
            weighting: How to weight neighbors
            local_model: Local regression model type
            n_components: PLS components for local model
        """
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.weighting = weighting
        self.local_model = local_model
        self.n_components = n_components

        # Storage for calibration set (memory)
        self.X_cal_ = None
        self.y_cal_ = None
        self.nn_ = None

    def fit(self, X, y):
        """
        Store calibration (memory) set.

        No global model is trained - all computation happens during prediction.

        Args:
            X: Calibration spectra (n_samples, n_features)
            y: Calibration property values (n_samples,) or (n_samples, n_targets)

        Returns:
            self
        """
        self.X_cal_ = np.asarray(X)
        self.y_cal_ = np.asarray(y)

        # Handle 1D target
        if self.y_cal_.ndim == 1:
            self.y_cal_ = self.y_cal_.reshape(-1, 1)

        # Fit neighbor search structure
        if self.similarity_metric == 'cosine':
            # Cosine similarity = 1 - cosine distance
            self.nn_ = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, len(self.X_cal_)),
                metric='cosine',
                n_jobs=-1
            )
        elif self.similarity_metric == 'euclidean':
            self.nn_ = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, len(self.X_cal_)),
                metric='euclidean',
                n_jobs=-1
            )
        elif self.similarity_metric == 'mahalanobis':
            # Mahalanobis requires precomputing covariance
            self.cov_ = np.cov(self.X_cal_.T)
            self.nn_ = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, len(self.X_cal_)),
                metric='mahalanobis',
                metric_params={'V': self.cov_},
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown similarity_metric: {self.similarity_metric}")

        self.nn_.fit(self.X_cal_)

        return self

    def predict(self, X):
        """
        Predict using memory-based learning.

        For each prediction sample:
        1. Find k nearest neighbors in calibration set
        2. Fit local model on neighbors
        3. Predict using local model

        Args:
            X: Prediction spectra (n_samples, n_features)

        Returns:
            Predictions (n_samples,) or (n_samples, n_targets)
        """
        if self.X_cal_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X)
        n_samples = X.shape[0]
        n_targets = self.y_cal_.shape[1]

        predictions = np.zeros((n_samples, n_targets))

        for i in range(n_samples):
            spectrum = X[i:i+1, :]

            # Find k nearest neighbors
            distances, indices = self.nn_.kneighbors(spectrum)
            neighbor_indices = indices[0]
            neighbor_distances = distances[0]

            # Get neighbor data
            X_neighbors = self.X_cal_[neighbor_indices]
            y_neighbors = self.y_cal_[neighbor_indices]

            # Compute weights
            weights = self._compute_weights(neighbor_distances)

            # Fit local model and predict
            pred = self._fit_and_predict_local(
                X_neighbors, y_neighbors, weights, spectrum
            )

            predictions[i, :] = pred

        # Return 1D if single target
        if n_targets == 1:
            return predictions.ravel()
        else:
            return predictions

    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty quantification from local variance.

        Args:
            X: Prediction spectra (n_samples, n_features)

        Returns:
            predictions: Predicted values (n_samples,) or (n_samples, n_targets)
            uncertainties: Standard deviation of neighbors (n_samples,) or (n_samples, n_targets)
        """
        if self.X_cal_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X)
        n_samples = X.shape[0]
        n_targets = self.y_cal_.shape[1]

        predictions = np.zeros((n_samples, n_targets))
        uncertainties = np.zeros((n_samples, n_targets))

        for i in range(n_samples):
            spectrum = X[i:i+1, :]

            # Find k nearest neighbors
            distances, indices = self.nn_.kneighbors(spectrum)
            neighbor_indices = indices[0]
            neighbor_distances = distances[0]

            # Get neighbor data
            X_neighbors = self.X_cal_[neighbor_indices]
            y_neighbors = self.y_cal_[neighbor_indices]

            # Compute weights
            weights = self._compute_weights(neighbor_distances)

            # Prediction from local model
            pred = self._fit_and_predict_local(
                X_neighbors, y_neighbors, weights, spectrum
            )
            predictions[i, :] = pred

            # Uncertainty from weighted variance of neighbors
            for j in range(n_targets):
                # Weighted standard deviation
                weighted_mean = np.average(y_neighbors[:, j], weights=weights)
                weighted_var = np.average(
                    (y_neighbors[:, j] - weighted_mean) ** 2,
                    weights=weights
                )
                uncertainties[i, j] = np.sqrt(weighted_var)

        # Return 1D if single target
        if n_targets == 1:
            return predictions.ravel(), uncertainties.ravel()
        else:
            return predictions, uncertainties

    def _compute_weights(self, distances):
        """
        Compute neighbor weights based on distances.

        Args:
            distances: Distances to neighbors (k,)

        Returns:
            weights: Weights for neighbors (k,)
        """
        if self.weighting == 'uniform':
            # Equal weights
            return np.ones(len(distances)) / len(distances)

        elif self.weighting == 'distance':
            # Inverse distance weighting
            # Convert distances to similarities
            # For cosine: distance = 1 - similarity, so similarity = 1 - distance
            if self.similarity_metric == 'cosine':
                similarities = 1.0 - distances
            else:
                # For Euclidean/Mahalanobis: similarity = 1 / (1 + distance)
                similarities = 1.0 / (1.0 + distances)

            # Normalize
            weights = similarities / np.sum(similarities)
            return weights

        elif self.weighting == 'gaussian':
            # Gaussian kernel weighting
            sigma = np.std(distances) if np.std(distances) > 0 else 1.0
            weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
            weights = weights / np.sum(weights)
            return weights

        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

    def _fit_and_predict_local(self, X_neighbors, y_neighbors, weights, X_pred):
        """
        Fit local model on neighbors and predict.

        Args:
            X_neighbors: Neighbor spectra (k, n_features)
            y_neighbors: Neighbor values (k, n_targets)
            weights: Neighbor weights (k,)
            X_pred: Prediction spectrum (1, n_features)

        Returns:
            prediction: (n_targets,)
        """
        n_targets = y_neighbors.shape[1]

        if self.local_model == 'pls':
            # Local PLS
            # Note: sklearn PLS doesn't support sample_weight
            # Use unweighted PLS (weights used for variance estimation)
            n_comp = min(self.n_components, X_neighbors.shape[0] - 1, X_neighbors.shape[1])

            try:
                model = PLSRegression(n_components=n_comp)
                model.fit(X_neighbors, y_neighbors)
                pred = model.predict(X_pred).ravel()
            except:
                # Fallback to weighted mean if PLS fails
                pred = np.average(y_neighbors, axis=0, weights=weights)

        elif self.local_model == 'ridge':
            # Local Ridge regression
            try:
                model = Ridge(alpha=1.0)
                model.fit(X_neighbors, y_neighbors, sample_weight=weights)
                pred = model.predict(X_pred).ravel()
            except:
                # Fallback
                pred = np.average(y_neighbors, axis=0, weights=weights)

        elif self.local_model == 'wls':
            # Weighted least squares
            try:
                # Apply weights by scaling
                sqrt_weights = np.sqrt(weights)
                X_weighted = X_neighbors * sqrt_weights[:, np.newaxis]
                y_weighted = y_neighbors * sqrt_weights[:, np.newaxis]

                model = LinearRegression()
                model.fit(X_weighted, y_weighted)

                # Predict on unweighted spectrum
                pred = model.predict(X_pred).ravel()
            except:
                # Fallback
                pred = np.average(y_neighbors, axis=0, weights=weights)

        else:
            raise ValueError(f"Unknown local_model: {self.local_model}")

        return pred

    def score(self, X, y):
        """
        Return R² score.

        Args:
            X: Test spectra (n_samples, n_features)
            y: True values (n_samples,) or (n_samples, n_targets)

        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

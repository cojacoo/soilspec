"""
Combined feature extraction transformers.

Provides convenience classes that combine multiple feature extraction steps
into single sklearn-compatible transformers.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import Optional

from .peak_integration import PeakIntegrator, PeakHeightExtractor
from .ratios import SpectralRatios, SpectralIndices


class PhysicsInformedFeatures(BaseEstimator, TransformerMixin):
    """
    Extract all physics-informed features in one transformer.

    Combines peak integration, ratios, and indices into a single feature set.
    This is a convenience class for quickly extracting all domain-knowledge features.

    Example:
        >>> from soilspec_pinn.features import PhysicsInformedFeatures
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.cross_decomposition import PLSRegression
        >>> from sklearn.pipeline import Pipeline
        >>>
        >>> # Create enhanced PLS model
        >>> model = Pipeline([
        ...     ('features', PhysicsInformedFeatures()),
        ...     ('scaler', StandardScaler()),
        ...     ('pls', PLSRegression(n_components=10))
        ... ])
        >>>
        >>> model.fit(spectra, y_values, features__wavenumbers=wavenumbers)
        >>> predictions = model.predict(new_spectra)
    """

    def __init__(
        self,
        include_peaks: bool = True,
        include_ratios: bool = True,
        include_indices: bool = True,
        region_selection: str = 'key',
        spectral_bands_csv: Optional[str] = None
    ):
        """
        Initialize combined feature extractor.

        Args:
            include_peaks: Whether to include peak integration features
            include_ratios: Whether to include ratio features
            include_indices: Whether to include spectral indices
            region_selection: Region selection for peak integrator
                            ('key', 'all', 'organic', 'mineral')
            spectral_bands_csv: Path to spectral_bands.csv
        """
        self.include_peaks = include_peaks
        self.include_ratios = include_ratios
        self.include_indices = include_indices
        self.region_selection = region_selection
        self.spectral_bands_csv = spectral_bands_csv

        self.peak_integrator_ = None
        self.ratio_calculator_ = None
        self.index_calculator_ = None
        self.wavenumbers_ = None
        self.feature_names_ = None

    def fit(self, X, y=None, wavenumbers=None):
        """
        Fit feature extractors.

        Args:
            X: Spectra (n_samples, n_wavelengths)
            y: Target values (ignored)
            wavenumbers: Array of wavenumber values (required)

        Returns:
            self
        """
        if wavenumbers is None:
            raise ValueError("wavenumbers must be provided during fit")

        self.wavenumbers_ = np.array(wavenumbers)
        self.feature_names_ = []

        # Fit peak integrator
        if self.include_peaks:
            self.peak_integrator_ = PeakIntegrator(
                spectral_bands_csv=self.spectral_bands_csv,
                region_selection=self.region_selection
            )
            self.peak_integrator_.fit(X, wavenumbers=wavenumbers)
            self.feature_names_.extend(self.peak_integrator_.get_feature_names_out())

        # Fit ratio calculator
        if self.include_ratios and self.include_peaks:
            # Ratios require peak features
            peak_features = self.peak_integrator_.transform(X)
            peak_names = self.peak_integrator_.get_feature_names_out()

            self.ratio_calculator_ = SpectralRatios()
            self.ratio_calculator_.fit(peak_features, feature_names=peak_names)

            # Get ratio names after transform (since they depend on available features)
            ratio_features = self.ratio_calculator_.transform(peak_features)
            if ratio_features.shape[1] > 0:
                self.feature_names_.extend(self.ratio_calculator_.get_feature_names_out())

        # Fit index calculator
        if self.include_indices:
            self.index_calculator_ = SpectralIndices()
            self.index_calculator_.fit(X, wavenumbers=wavenumbers)

            # Get index names after transform
            index_features = self.index_calculator_.transform(X)
            if index_features.shape[1] > 0:
                self.feature_names_.extend(self.index_calculator_.get_feature_names_out())

        return self

    def transform(self, X):
        """
        Transform spectra to physics-informed features.

        Args:
            X: Spectra (n_samples, n_wavelengths)

        Returns:
            Feature matrix (n_samples, n_features)
        """
        feature_list = []

        # Peak integration features
        if self.include_peaks and self.peak_integrator_ is not None:
            peak_features = self.peak_integrator_.transform(X)
            feature_list.append(peak_features)

            # Ratio features (need peak features)
            if self.include_ratios and self.ratio_calculator_ is not None:
                ratio_features = self.ratio_calculator_.transform(peak_features)
                if ratio_features.shape[1] > 0:
                    feature_list.append(ratio_features)

        # Index features
        if self.include_indices and self.index_calculator_ is not None:
            index_features = self.index_calculator_.transform(X)
            if index_features.shape[1] > 0:
                feature_list.append(index_features)

        if len(feature_list) == 0:
            raise ValueError("No features could be extracted")

        # Concatenate all features
        combined_features = np.hstack(feature_list)

        return combined_features

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if self.feature_names_ is None:
            raise ValueError("Transformer must be fitted first")

        return np.array(self.feature_names_)

    def get_feature_importance_mapping(self):
        """
        Get mapping from features to spectral regions.

        Useful for interpreting model coefficients.

        Returns:
            Dictionary mapping feature names to wavenumber regions
        """
        mapping = {}

        if self.peak_integrator_ is not None:
            info = self.peak_integrator_.get_feature_info()
            for _, row in info.iterrows():
                mapping[row['feature_name']] = {
                    'type': 'peak_integration',
                    'wavenumber_range': (row['wavenumber_min'], row['wavenumber_max']),
                    'description': row['description'],
                    'chemical_type': row['type']
                }

        if self.ratio_calculator_ is not None:
            ratio_info = self.ratio_calculator_.get_ratio_info()
            for _, row in ratio_info.iterrows():
                mapping[row['ratio_name']] = {
                    'type': 'ratio',
                    'description': row['description'],
                    'reference': row['reference']
                }

        if self.index_calculator_ is not None:
            index_defs = self.index_calculator_._get_index_definitions()
            for name in self.index_calculator_.feature_names_:
                if name in index_defs:
                    mapping[name] = {
                        'type': 'index',
                        'description': index_defs[name]['description'],
                        'formula': index_defs[name]['formula']
                    }

        return mapping


class CompactFeatures(BaseEstimator, TransformerMixin):
    """
    Extract compact set of most important features.

    Uses only the most informative spectral regions for maximum efficiency
    with minimal feature set (~10-20 features).

    Example:
        >>> from soilspec_pinn.features import CompactFeatures
        >>> compact = CompactFeatures()
        >>> compact.fit(spectra, wavenumbers=wavenumbers)
        >>> features = compact.transform(spectra)
        >>> print(f"Extracted {features.shape[1]} features")
    """

    def __init__(self, spectral_bands_csv: Optional[str] = None):
        """
        Initialize compact feature extractor.

        Args:
            spectral_bands_csv: Path to spectral_bands.csv
        """
        self.spectral_bands_csv = spectral_bands_csv
        self.extractor_ = None
        self.wavenumbers_ = None

    def fit(self, X, y=None, wavenumbers=None):
        """
        Fit compact feature extractor.

        Args:
            X: Spectra (n_samples, n_wavelengths)
            y: Target values (ignored)
            wavenumbers: Array of wavenumber values (required)

        Returns:
            self
        """
        if wavenumbers is None:
            raise ValueError("wavenumbers must be provided during fit")

        self.wavenumbers_ = np.array(wavenumbers)

        # Use key regions only + top ratios + key indices
        self.extractor_ = PhysicsInformedFeatures(
            include_peaks=True,
            include_ratios=True,
            include_indices=True,
            region_selection='key',  # Only key regions
            spectral_bands_csv=self.spectral_bands_csv
        )

        self.extractor_.fit(X, wavenumbers=wavenumbers)

        return self

    def transform(self, X):
        """Transform to compact features."""
        return self.extractor_.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.extractor_.get_feature_names_out()


class ExtensiveFeatures(BaseEstimator, TransformerMixin):
    """
    Extract extensive set of all available features.

    Uses all spectral regions for maximum information (~50-100 features).
    Best for large datasets where overfitting is less of a concern.

    Example:
        >>> from soilspec_pinn.features import ExtensiveFeatures
        >>> extensive = ExtensiveFeatures()
        >>> extensive.fit(spectra, wavenumbers=wavenumbers)
        >>> features = extensive.transform(spectra)
        >>> print(f"Extracted {features.shape[1]} features")
    """

    def __init__(self, spectral_bands_csv: Optional[str] = None):
        """
        Initialize extensive feature extractor.

        Args:
            spectral_bands_csv: Path to spectral_bands.csv
        """
        self.spectral_bands_csv = spectral_bands_csv
        self.extractor_ = None
        self.wavenumbers_ = None

    def fit(self, X, y=None, wavenumbers=None):
        """
        Fit extensive feature extractor.

        Args:
            X: Spectra (n_samples, n_wavelengths)
            y: Target values (ignored)
            wavenumbers: Array of wavenumber values (required)

        Returns:
            self
        """
        if wavenumbers is None:
            raise ValueError("wavenumbers must be provided during fit")

        self.wavenumbers_ = np.array(wavenumbers)

        # Use all regions + all ratios + all indices
        self.extractor_ = PhysicsInformedFeatures(
            include_peaks=True,
            include_ratios=True,
            include_indices=True,
            region_selection='all',  # All regions
            spectral_bands_csv=self.spectral_bands_csv
        )

        self.extractor_.fit(X, wavenumbers=wavenumbers)

        return self

    def transform(self, X):
        """Transform to extensive features."""
        return self.extractor_.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.extractor_.get_feature_names_out()

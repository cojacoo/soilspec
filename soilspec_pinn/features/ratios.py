"""
Spectral ratio features based on domain knowledge.

Computes chemically meaningful ratios between spectral regions that are
known to be informative for soil properties.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List


class SpectralRatios(BaseEstimator, TransformerMixin):
    """
    Compute chemically meaningful spectral ratios.

    Ratios between specific spectral regions are often more robust than
    absolute intensities and can indicate soil organic matter quality,
    mineraology, and other properties.

    Example:
        >>> from soilspec_pinn.features import PeakIntegrator, SpectralRatios
        >>> from sklearn.pipeline import Pipeline
        >>> # Chain with PeakIntegrator
        >>> pipeline = Pipeline([
        ...     ('peaks', PeakIntegrator()),
        ...     ('ratios', SpectralRatios())
        ... ])
        >>> pipeline.fit(spectra, wavenumbers=wavenumbers)
        >>> ratio_features = pipeline.transform(spectra)
    """

    def __init__(self, ratios: Optional[List[str]] = None):
        """
        Initialize spectral ratio calculator.

        Args:
            ratios: List of ratio types to compute. If None, uses all default ratios.
                   Options: 'aliphatic_aromatic', 'organic_mineral',
                           'carbohydrate_amide', 'clay_carbonate', etc.
        """
        self.ratios = ratios
        self.feature_names_ = None
        self.input_feature_names_ = None

    def fit(self, X, y=None, feature_names=None):
        """
        Fit by storing feature names.

        Args:
            X: Features from PeakIntegrator (n_samples, n_features)
            y: Target values (ignored)
            feature_names: Names of input features (from PeakIntegrator)

        Returns:
            self
        """
        if feature_names is not None:
            self.input_feature_names_ = np.array(feature_names)
        elif hasattr(X, 'columns'):
            # If X is a DataFrame
            self.input_feature_names_ = X.columns.values
        else:
            # Try to infer from X if it's from a pipeline
            self.input_feature_names_ = None

        return self

    def transform(self, X):
        """
        Compute spectral ratios.

        Args:
            X: Features (n_samples, n_features) from PeakIntegrator

        Returns:
            Ratio features (n_samples, n_ratios)
        """
        # Convert to DataFrame if needed for easier column access
        if not isinstance(X, pd.DataFrame):
            if self.input_feature_names_ is not None:
                X_df = pd.DataFrame(X, columns=self.input_feature_names_)
            else:
                # Can't compute named ratios without feature names
                raise ValueError(
                    "Feature names must be provided during fit or X must be a DataFrame"
                )
        else:
            X_df = X

        ratios_dict = {}
        ratio_definitions = self._get_ratio_definitions()

        # Determine which ratios to compute
        ratios_to_compute = self.ratios if self.ratios is not None else ratio_definitions.keys()

        for ratio_name in ratios_to_compute:
            if ratio_name not in ratio_definitions:
                continue

            ratio_def = ratio_definitions[ratio_name]

            # Compute ratio
            ratio_values = self._compute_ratio(
                X_df,
                ratio_def['numerator'],
                ratio_def['denominator']
            )

            if ratio_values is not None:
                ratios_dict[ratio_name] = ratio_values

        # Store feature names
        self.feature_names_ = list(ratios_dict.keys())

        # Convert to array
        if len(ratios_dict) == 0:
            return np.zeros((X.shape[0], 0))

        ratio_matrix = np.column_stack([ratios_dict[k] for k in self.feature_names_])

        return ratio_matrix

    def _get_ratio_definitions(self) -> dict:
        """
        Define spectral ratios based on soil science literature.

        Returns:
            Dictionary of ratio definitions
        """
        return {
            # Organic matter quality ratios
            'aliphatic_aromatic': {
                'numerator': ['aliphatic_ch', 'Aliphates'],
                'denominator': ['aromatic', 'Aromates'],
                'description': 'High = fresh OM, Low = humified OM',
                'reference': 'Haberhauer et al., 1998'
            },
            'carbohydrate_amide': {
                'numerator': ['carbohydrates', 'Carbohydrates'],
                'denominator': ['amide', 'Amide'],
                'description': 'Plant-derived vs microbial organic matter',
                'reference': 'Calderon et al., 2013'
            },
            'aliphatic_carboxylate': {
                'numerator': ['aliphatic_ch'],
                'denominator': ['Carboxylate'],
                'description': 'Organic matter decomposition stage',
                'reference': 'Parikh et al., 2014'
            },

            # Organic vs mineral ratios
            'organic_mineral': {
                'numerator': ['fingerprint'],  # Will sum all organic
                'denominator': ['clay_silicate'],  # Will sum all mineral
                'description': 'Proxy for organic matter content',
                'reference': 'General soil spectroscopy'
            },

            # Mineral ratios
            'clay_carbonate': {
                'numerator': ['clay_oh', 'clay_silicate', 'Clay minerals'],
                'denominator': ['carbonates', 'Carbonates'],
                'description': 'Mineralogy indicator',
                'reference': 'Tinti et al., 2015'
            },
            'clay_quartz': {
                'numerator': ['clay_silicate', 'Clay minerals'],
                'denominator': ['quartz', 'Quartz'],
                'description': 'Fine vs coarse mineral fraction',
                'reference': 'Soriano-Disla et al., 2014'
            },

            # Specific functional group ratios
            'ch2_ch3': {
                'numerator': ['2920'],  # CH2 asymmetric stretch
                'denominator': ['2850'],  # CH3 symmetric stretch
                'description': 'Chain length of aliphatic compounds',
                'reference': 'Socrates, 2001'
            },
            'amide_I_II': {
                'numerator': ['1630'],  # Amide I
                'denominator': ['1510'],  # Amide II
                'description': 'Protein secondary structure',
                'reference': 'Barth, 2007'
            },
        }

    def _compute_ratio(
        self,
        X_df: pd.DataFrame,
        numerator_names: List[str],
        denominator_names: List[str],
        epsilon: float = 1e-6
    ) -> Optional[np.ndarray]:
        """
        Compute a specific ratio.

        Args:
            X_df: Feature DataFrame
            numerator_names: List of feature names to sum for numerator
            denominator_names: List of feature names to sum for denominator
            epsilon: Small value to avoid division by zero

        Returns:
            Ratio values, or None if features not found
        """
        # Find numerator features
        numerator_cols = []
        for name in numerator_names:
            # Check for exact match or substring match
            matches = [col for col in X_df.columns if name in col or col in name]
            numerator_cols.extend(matches)

        # Find denominator features
        denominator_cols = []
        for name in denominator_names:
            matches = [col for col in X_df.columns if name in col or col in name]
            denominator_cols.extend(matches)

        # Remove duplicates
        numerator_cols = list(set(numerator_cols))
        denominator_cols = list(set(denominator_cols))

        if len(numerator_cols) == 0 or len(denominator_cols) == 0:
            # Can't compute ratio - features not found
            return None

        # Sum numerator and denominator
        numerator = X_df[numerator_cols].sum(axis=1).values
        denominator = X_df[denominator_cols].sum(axis=1).values

        # Compute ratio with epsilon to avoid division by zero
        ratio = numerator / (denominator + epsilon)

        return ratio

    def get_feature_names_out(self, input_features=None):
        """Get feature names for output."""
        if self.feature_names_ is None:
            raise ValueError("Transformer must be fitted and transformed first")

        return np.array(self.feature_names_)

    def get_ratio_info(self) -> pd.DataFrame:
        """
        Get detailed information about computed ratios.

        Returns:
            DataFrame with ratio names, descriptions, and references
        """
        if self.feature_names_ is None:
            raise ValueError("Transformer must be fitted and transformed first")

        ratio_defs = self._get_ratio_definitions()

        info = []
        for ratio_name in self.feature_names_:
            if ratio_name in ratio_defs:
                info.append({
                    'ratio_name': ratio_name,
                    'description': ratio_defs[ratio_name]['description'],
                    'reference': ratio_defs[ratio_name]['reference']
                })

        return pd.DataFrame(info)


class SpectralIndices(BaseEstimator, TransformerMixin):
    """
    Compute spectral indices similar to vegetation indices (NDVI, etc.).

    These are mathematical transformations of specific spectral regions
    that normalize for intensity variations.

    Example:
        >>> from soilspec_pinn.features import SpectralIndices
        >>> indices = SpectralIndices(indices=['OMNI', 'CMI'])
        >>> indices.fit(spectra, wavenumbers=wavenumbers)
        >>> index_features = indices.transform(spectra)
    """

    def __init__(self, indices: Optional[List[str]] = None):
        """
        Initialize spectral index calculator.

        Args:
            indices: List of indices to compute. If None, uses common indices.
                    Options: 'OMNI' (Organic Matter Normalized Index),
                            'CMI' (Clay Mineral Index),
                            'CARI' (Carbonate Index), etc.
        """
        self.indices = indices
        self.wavenumbers_ = None
        self.feature_names_ = None

    def fit(self, X, y=None, wavenumbers=None):
        """
        Fit by storing wavenumbers.

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
        return self

    def transform(self, X):
        """
        Compute spectral indices.

        Args:
            X: Spectra (n_samples, n_wavelengths)

        Returns:
            Index features (n_samples, n_indices)
        """
        indices_dict = {}
        index_definitions = self._get_index_definitions()

        # Determine which indices to compute
        indices_to_compute = self.indices if self.indices is not None else index_definitions.keys()

        for index_name in indices_to_compute:
            if index_name not in index_definitions:
                continue

            index_def = index_definitions[index_name]

            # Compute index
            index_values = self._compute_index(X, index_def)

            if index_values is not None:
                indices_dict[index_name] = index_values

        # Store feature names
        self.feature_names_ = list(indices_dict.keys())

        # Convert to array
        if len(indices_dict) == 0:
            return np.zeros((X.shape[0], 0))

        index_matrix = np.column_stack([indices_dict[k] for k in self.feature_names_])

        return index_matrix

    def _get_index_definitions(self) -> dict:
        """
        Define spectral indices.

        Returns:
            Dictionary of index definitions
        """
        return {
            'OMNI': {
                'type': 'normalized_difference',
                'band1': 2920,  # Aliphatic C-H (organic)
                'band2': 1080,  # Si-O (mineral)
                'description': 'Organic Matter Normalized Index',
                'formula': '(R2920 - R1080) / (R2920 + R1080)'
            },
            'CMI': {
                'type': 'normalized_difference',
                'band1': 3620,  # Clay OH
                'band2': 1880,  # Quartz
                'description': 'Clay Mineral Index',
                'formula': '(R3620 - R1880) / (R3620 + R1880)'
            },
            'CARI': {
                'type': 'simple_ratio',
                'band1': 1430,  # Carbonates
                'band2': 1080,  # Silicates
                'description': 'Carbonate Index',
                'formula': 'R1430 / R1080'
            },
            'AMI': {
                'type': 'normalized_difference',
                'band1': 1630,  # Amide I
                'band2': 2920,  # Aliphatic
                'description': 'Amide (Protein) Index',
                'formula': '(R1630 - R2920) / (R1630 + R2920)'
            }
        }

    def _compute_index(self, X: np.ndarray, index_def: dict) -> Optional[np.ndarray]:
        """
        Compute a specific index.

        Args:
            X: Spectra array
            index_def: Index definition dictionary

        Returns:
            Index values, or None if wavenumbers not found
        """
        # Find band positions
        idx1 = self._find_nearest_index(index_def['band1'])
        idx2 = self._find_nearest_index(index_def['band2'])

        if idx1 is None or idx2 is None:
            return None

        R1 = X[:, idx1]
        R2 = X[:, idx2]

        # Compute based on type
        if index_def['type'] == 'normalized_difference':
            # (R1 - R2) / (R1 + R2)
            index = (R1 - R2) / (R1 + R2 + 1e-6)

        elif index_def['type'] == 'simple_ratio':
            # R1 / R2
            index = R1 / (R2 + 1e-6)

        else:
            return None

        return index

    def _find_nearest_index(self, target_wn: float, tolerance: float = 10.0) -> Optional[int]:
        """
        Find index of nearest wavenumber.

        Args:
            target_wn: Target wavenumber
            tolerance: Maximum distance to search

        Returns:
            Index, or None if not found within tolerance
        """
        distances = np.abs(self.wavenumbers_ - target_wn)
        closest_idx = np.argmin(distances)

        if distances[closest_idx] <= tolerance:
            return closest_idx
        else:
            return None

    def get_feature_names_out(self, input_features=None):
        """Get feature names for output."""
        if self.feature_names_ is None:
            raise ValueError("Transformer must be fitted and transformed first")

        return np.array(self.feature_names_)

"""
Peak integration features based on spectral band assignments.

Extracts chemically meaningful features by integrating absorbance over
specific wavenumber regions defined in spectral_bands.csv.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Dict
from pathlib import Path

from ..knowledge import SpectralBandDatabase


class PeakIntegrator(BaseEstimator, TransformerMixin):
    """
    Extract features by integrating absorbance over spectral bands.

    Uses spectral_bands.csv to define chemically meaningful integration regions.
    Reduces dimensionality from ~1800 wavelengths to ~50-100 chemical features.

    Example:
        >>> from soilspec.features import PeakIntegrator
        >>> integrator = PeakIntegrator()
        >>> wavenumbers = np.arange(600, 4001, 2)
        >>> # Fit with wavenumbers
        >>> integrator.fit(spectra, wavenumbers=wavenumbers)
        >>> # Transform spectra to features
        >>> features = integrator.transform(spectra)
        >>> # Get feature names
        >>> names = integrator.get_feature_names_out()
    """

    def __init__(
        self,
        spectral_bands_csv: Optional[str] = None,
        integration_method: str = 'trapz',
        region_selection: str = 'key'
    ):
        """
        Initialize peak integrator.

        Args:
            spectral_bands_csv: Path to spectral_bands.csv. If None, uses default.
            integration_method: Integration method ('trapz', 'sum', 'max', 'mean')
            region_selection: Which regions to use:
                - 'key': Use predefined key regions (default, ~10 features)
                - 'all': Use all unique regions (~50-100 features)
                - 'organic': Only organic bands
                - 'mineral': Only mineral bands
        """
        self.spectral_bands_csv = spectral_bands_csv
        self.integration_method = integration_method
        self.region_selection = region_selection
        self.band_db = None
        self.regions_ = None
        self.wavenumbers_ = None
        self.feature_names_ = None

    def fit(self, X, y=None, wavenumbers=None):
        """
        Fit by defining integration regions.

        Args:
            X: Spectra (n_samples, n_wavelengths)
            y: Target values (ignored)
            wavenumbers: Array of wavenumber values (required)

        Returns:
            self
        """
        if wavenumbers is None:
            raise ValueError(
                "wavenumbers must be provided during fit. "
                "Pass as: integrator.fit(X, wavenumbers=wavenumber_array)"
            )

        self.wavenumbers_ = np.array(wavenumbers)

        # Load spectral band database
        self.band_db = SpectralBandDatabase(self.spectral_bands_csv)

        # Define integration regions based on selection method
        self.regions_ = self._define_regions()
        self.feature_names_ = [region['name'] for region in self.regions_]

        return self

    def _define_regions(self) -> List[Dict]:
        """
        Define integration regions based on region_selection.

        Returns:
            List of region dictionaries
        """
        regions = []

        if self.region_selection == 'key':
            # Use predefined key regions
            key_regions = self.band_db.get_key_regions()

            for region_name, region_info in key_regions.items():
                wn_min, wn_max = region_info['range']
                regions.append({
                    'name': region_name,
                    'wn_min': wn_min,
                    'wn_max': wn_max,
                    'type': region_info.get('type', 'unknown'),
                    'description': region_info['description']
                })

        elif self.region_selection == 'all':
            # Group bands by information field
            unique_info = self.band_db.bands['information'].unique()

            for info in unique_info:
                bands = self.band_db.get_bands(information=info)
                if len(bands) > 0:
                    # Use the broadest range for this information type
                    wn_min = bands['band_start'].min()
                    wn_max = bands['band_end'].max()
                    band_type = bands['type'].mode()[0] if len(bands['type']) > 0 else 'unknown'

                    regions.append({
                        'name': info,
                        'wn_min': wn_min,
                        'wn_max': wn_max,
                        'type': band_type,
                        'description': info
                    })

        elif self.region_selection in ['organic', 'mineral', 'water']:
            # Use all bands of specific type
            band_type_map = {
                'organic': 'org',
                'mineral': 'min',
                'water': 'water'
            }
            band_type = band_type_map[self.region_selection]
            type_bands = self.band_db.get_bands(type=band_type)

            # Group by information field
            for info in type_bands['information'].unique():
                info_bands = type_bands[type_bands['information'] == info]
                wn_min = info_bands['band_start'].min()
                wn_max = info_bands['band_end'].max()

                regions.append({
                    'name': info,
                    'wn_min': wn_min,
                    'wn_max': wn_max,
                    'type': band_type,
                    'description': info
                })

        else:
            raise ValueError(
                f"Unknown region_selection: {self.region_selection}. "
                "Choose from: 'key', 'all', 'organic', 'mineral', 'water'"
            )

        return regions

    def transform(self, X):
        """
        Transform spectra to integrated features.

        Args:
            X: Spectra (n_samples, n_wavelengths)

        Returns:
            Feature matrix (n_samples, n_regions)
        """
        if self.regions_ is None:
            raise ValueError("Transformer must be fitted before transform")

        n_samples = X.shape[0]
        n_regions = len(self.regions_)
        features = np.zeros((n_samples, n_regions))

        for i, region in enumerate(self.regions_):
            # Find wavenumbers in this region
            mask = (self.wavenumbers_ >= region['wn_min']) & \
                   (self.wavenumbers_ <= region['wn_max'])

            if not np.any(mask):
                # Region not in spectral range
                features[:, i] = 0
                continue

            # Extract region for all samples
            region_spectra = X[:, mask]
            region_wn = self.wavenumbers_[mask]

            # Apply integration method
            if self.integration_method == 'trapz':
                # Trapezoidal integration
                for sample_idx in range(n_samples):
                    features[sample_idx, i] = np.trapz(
                        region_spectra[sample_idx, :],
                        region_wn
                    )

            elif self.integration_method == 'sum':
                # Simple sum
                features[:, i] = np.sum(region_spectra, axis=1)

            elif self.integration_method == 'max':
                # Maximum absorbance in region
                features[:, i] = np.max(region_spectra, axis=1)

            elif self.integration_method == 'mean':
                # Mean absorbance in region
                features[:, i] = np.mean(region_spectra, axis=1)

            else:
                raise ValueError(f"Unknown integration_method: {self.integration_method}")

        return features

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for output.

        Args:
            input_features: Ignored (for sklearn compatibility)

        Returns:
            Array of feature names
        """
        if self.feature_names_ is None:
            raise ValueError("Transformer must be fitted first")

        return np.array(self.feature_names_)

    def get_feature_info(self) -> pd.DataFrame:
        """
        Get detailed information about extracted features.

        Returns:
            DataFrame with feature names, types, ranges, and descriptions
        """
        if self.regions_ is None:
            raise ValueError("Transformer must be fitted first")

        info = []
        for region in self.regions_:
            info.append({
                'feature_name': region['name'],
                'type': region['type'],
                'wavenumber_min': region['wn_min'],
                'wavenumber_max': region['wn_max'],
                'description': region['description']
            })

        return pd.DataFrame(info)


class PeakHeightExtractor(BaseEstimator, TransformerMixin):
    """
    Extract peak heights at specific wavenumbers.

    Useful for extracting intensities at known important peaks rather than
    integrating over regions.

    Example:
        >>> from soilspec.features import PeakHeightExtractor
        >>> # Extract heights at key wavenumbers
        >>> extractor = PeakHeightExtractor(
        ...     peak_wavenumbers=[2920, 1630, 1030, 3620]
        ... )
        >>> extractor.fit(spectra, wavenumbers=wavenumbers)
        >>> peak_heights = extractor.transform(spectra)
    """

    def __init__(
        self,
        peak_wavenumbers: Optional[List[float]] = None,
        tolerance: float = 5.0
    ):
        """
        Initialize peak height extractor.

        Args:
            peak_wavenumbers: List of wavenumbers to extract. If None, uses
                             common soil spectroscopy peaks.
            tolerance: Maximum distance (cm⁻¹) to search for peak
        """
        self.peak_wavenumbers = peak_wavenumbers
        self.tolerance = tolerance
        self.wavenumbers_ = None
        self.peak_indices_ = None
        self.feature_names_ = None

    def fit(self, X, y=None, wavenumbers=None):
        """
        Fit by finding peak positions.

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

        # Use default peaks if none provided
        if self.peak_wavenumbers is None:
            self.peak_wavenumbers = self._get_default_peaks()

        # Find indices of peak positions
        self.peak_indices_ = []
        self.feature_names_ = []

        band_db = SpectralBandDatabase()

        for peak_wn in self.peak_wavenumbers:
            # Find closest wavenumber
            distances = np.abs(self.wavenumbers_ - peak_wn)
            closest_idx = np.argmin(distances)

            if distances[closest_idx] <= self.tolerance:
                self.peak_indices_.append(closest_idx)

                # Get chemical information for feature name
                info = band_db.get_region_type(peak_wn)
                if info:
                    name = f"{peak_wn:.0f}_{info[0]['information']}"
                else:
                    name = f"{peak_wn:.0f}_unknown"

                self.feature_names_.append(name)

        return self

    def _get_default_peaks(self) -> List[float]:
        """Get default peak positions for soil spectroscopy."""
        return [
            # Organic matter
            2920,  # Aliphatic C-H
            2850,  # Aliphatic CH2
            1630,  # Amide I / Carboxylate
            1510,  # Aromatic C=C / TC proxy
            1450,  # Aliphatic CH
            1030,  # Carbohydrate C-O
            # Mineral
            3620,  # Clay OH
            1080,  # Clay Si-O
            1430,  # Carbonates
            915,   # Clay Al-OH
            # Other
            1640,  # Water
        ]

    def transform(self, X):
        """
        Extract peak heights.

        Args:
            X: Spectra (n_samples, n_wavelengths)

        Returns:
            Peak heights (n_samples, n_peaks)
        """
        if self.peak_indices_ is None:
            raise ValueError("Transformer must be fitted first")

        return X[:, self.peak_indices_]

    def get_feature_names_out(self, input_features=None):
        """Get feature names for output."""
        if self.feature_names_ is None:
            raise ValueError("Transformer must be fitted first")

        return np.array(self.feature_names_)

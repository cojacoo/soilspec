"""
Parse and query spectral band assignments.

This module provides an interface to the spectral_bands.csv database containing
150+ literature-referenced peak assignments for soil MIR spectroscopy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union


class SpectralBandDatabase:
    """
    Query spectral band assignments from spectral_bands.csv.

    Based on 150+ literature-referenced peak assignments from multiple studies
    including Margenot et al., Tinti et al., Soriano-Disla et al., and others.

    Example:
        >>> from soilspec_pinn.knowledge import SpectralBandDatabase
        >>> bands = SpectralBandDatabase()
        >>> organic_bands = bands.get_bands(type='org')
        >>> clay_bands = bands.get_bands(information='Clay minerals')
    """

    def __init__(self, csv_path: Optional[Union[str, Path]] = None):
        """
        Load spectral band database.

        Args:
            csv_path: Path to spectral_bands.csv. If None, uses default location.
        """
        if csv_path is None:
            # Default: look in knowledge directory
            csv_path = Path(__file__).parent / 'spectral_bands.csv'
        else:
            csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"spectral_bands.csv not found at {csv_path}. "
                "Please ensure the file is in the knowledge directory."
            )

        # Load and clean the CSV
        self.bands = pd.read_csv(csv_path, encoding='utf-8-sig')  # Handle BOM

        # Standardize column names
        self.bands.columns = [
            'band_start', 'band_end', 'type',
            'information', 'description', 'reference'
        ]

        # Convert wavenumbers to float
        self.bands['band_start'] = pd.to_numeric(self.bands['band_start'], errors='coerce')
        self.bands['band_end'] = pd.to_numeric(self.bands['band_end'], errors='coerce')

        # Remove any rows with missing wavenumbers
        self.bands = self.bands.dropna(subset=['band_start', 'band_end'])

        # Clean type column (standardize to lowercase)
        self.bands['type'] = self.bands['type'].str.lower().str.strip()

        # Sort by wavenumber
        self.bands = self.bands.sort_values('band_start').reset_index(drop=True)

    def get_bands(
        self,
        type: Optional[str] = None,
        information: Optional[str] = None,
        wavenumber_range: Optional[tuple] = None
    ) -> pd.DataFrame:
        """
        Query bands by type, information, or wavenumber range.

        Args:
            type: Filter by type ('org', 'min', 'water', 'rg')
            information: Filter by chemical information (case-insensitive substring match)
            wavenumber_range: Tuple (min_wn, max_wn) to filter bands

        Returns:
            DataFrame of matching band assignments

        Example:
            >>> bands = SpectralBandDatabase()
            >>> # Get all organic bands
            >>> org_bands = bands.get_bands(type='org')
            >>> # Get clay mineral bands
            >>> clay_bands = bands.get_bands(information='Clay')
            >>> # Get bands in fingerprint region
            >>> fingerprint = bands.get_bands(wavenumber_range=(600, 1500))
        """
        result = self.bands.copy()

        if type is not None:
            result = result[result['type'] == type.lower()]

        if information is not None:
            result = result[
                result['information'].str.contains(information, case=False, na=False) |
                result['description'].str.contains(information, case=False, na=False)
            ]

        if wavenumber_range is not None:
            min_wn, max_wn = wavenumber_range
            # Keep bands that overlap with the range
            result = result[
                (result['band_end'] >= min_wn) &
                (result['band_start'] <= max_wn)
            ]

        return result.reset_index(drop=True)

    def get_region_type(self, wavenumber: float) -> List[dict]:
        """
        Determine what a wavenumber represents.

        Args:
            wavenumber: Wavenumber in cm⁻¹

        Returns:
            List of dictionaries describing what this region represents,
            sorted by priority (organic > mineral > water)

        Example:
            >>> bands = SpectralBandDatabase()
            >>> bands.get_region_type(2920)
            [{'type': 'org', 'information': 'Aliphates',
              'description': 'Aliphatic C-H of aliphatic methyl and methylene groups',
              'reference': 'Haberhauer et al., 1998'}]
        """
        matches = self.bands[
            (self.bands['band_start'] <= wavenumber) &
            (self.bands['band_end'] >= wavenumber)
        ]

        if len(matches) == 0:
            return []

        # Sort by type priority: org > min > water > other
        type_priority = {'org': 0, 'min': 1, 'water': 2}
        matches['priority'] = matches['type'].map(lambda t: type_priority.get(t, 3))
        matches = matches.sort_values('priority')

        # Convert to list of dicts
        results = []
        for _, row in matches.iterrows():
            results.append({
                'type': row['type'],
                'information': row['information'],
                'description': row['description'],
                'reference': row['reference']
            })

        return results

    def create_mask(self, wavenumbers: np.ndarray, type: str = 'org') -> np.ndarray:
        """
        Create binary mask for spectral regions of given type.

        Args:
            wavenumbers: Array of wavenumber values
            type: Band type ('org', 'min', 'water')

        Returns:
            Binary mask (1 where wavenumber is in type regions, 0 elsewhere)

        Example:
            >>> bands = SpectralBandDatabase()
            >>> wavenumbers = np.arange(600, 4001, 2)
            >>> organic_mask = bands.create_mask(wavenumbers, type='org')
            >>> # Use mask to select organic regions
            >>> organic_absorbance = spectrum * organic_mask
        """
        mask = np.zeros_like(wavenumbers, dtype=float)
        type_bands = self.get_bands(type=type)

        for _, band in type_bands.iterrows():
            in_band = (wavenumbers >= band['band_start']) & \
                      (wavenumbers <= band['band_end'])
            mask[in_band] = 1.0

        return mask

    def get_key_regions(self) -> dict:
        """
        Get key spectral regions organized by chemical type.

        Returns:
            Dictionary mapping region names to wavenumber ranges

        Example:
            >>> bands = SpectralBandDatabase()
            >>> regions = bands.get_key_regions()
            >>> print(regions['aliphatic_ch'])
            {'range': (2800, 3000), 'type': 'org', 'description': '...'}
        """
        regions = {
            'fingerprint': {
                'range': (600, 1500),
                'description': 'Fingerprint region - overlapping organic and mineral'
            },
            'aliphatic_ch': {
                'range': (2800, 3000),
                'type': 'org',
                'description': 'Aliphatic C-H stretching (organic matter)'
            },
            'aromatic': {
                'range': (700, 900),
                'type': 'org',
                'description': 'Aromatic C-H out-of-plane bending'
            },
            'carbohydrates': {
                'range': (1030, 1170),
                'type': 'org',
                'description': 'Carbohydrate C-O stretching'
            },
            'amide': {
                'range': (1480, 1700),
                'type': 'org',
                'description': 'Amide I and II bands (proteins)'
            },
            'clay_oh': {
                'range': (3620, 3700),
                'type': 'min',
                'description': 'Clay mineral OH stretching'
            },
            'clay_silicate': {
                'range': (950, 1100),
                'type': 'min',
                'description': 'Clay Si-O stretching'
            },
            'carbonates': {
                'range': (1400, 1500),
                'type': 'min',
                'description': 'Carbonate vibrations'
            },
            'quartz': {
                'range': (1790, 2000),
                'type': 'min',
                'description': 'Quartz overtones'
            },
            'water': {
                'range': (1569, 1642),
                'type': 'water',
                'description': 'Water OH bending'
            }
        }

        return regions

    def get_references(self) -> pd.DataFrame:
        """
        Get unique references cited in the database.

        Returns:
            DataFrame with reference citations and counts

        Example:
            >>> bands = SpectralBandDatabase()
            >>> refs = bands.get_references()
            >>> print(refs.head())
        """
        ref_counts = self.bands['reference'].value_counts().reset_index()
        ref_counts.columns = ['reference', 'num_bands']
        return ref_counts

    def summarize(self) -> dict:
        """
        Get summary statistics of the database.

        Returns:
            Dictionary with summary information

        Example:
            >>> bands = SpectralBandDatabase()
            >>> summary = bands.summarize()
            >>> print(f"Total bands: {summary['total_bands']}")
            >>> print(f"Organic bands: {summary['by_type']['org']}")
        """
        summary = {
            'total_bands': len(self.bands),
            'by_type': self.bands['type'].value_counts().to_dict(),
            'wavenumber_range': (
                self.bands['band_start'].min(),
                self.bands['band_end'].max()
            ),
            'unique_references': self.bands['reference'].nunique()
        }

        return summary

    def __repr__(self) -> str:
        """String representation."""
        summary = self.summarize()
        return (
            f"SpectralBandDatabase({summary['total_bands']} bands, "
            f"{summary['unique_references']} references, "
            f"range: {summary['wavenumber_range'][0]:.0f}-"
            f"{summary['wavenumber_range'][1]:.0f} cm⁻¹)"
        )

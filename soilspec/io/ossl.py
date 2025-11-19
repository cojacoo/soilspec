"""
OSSL (Open Soil Spectral Library) format handlers.

Provides readers for OSSL dataset formats and interfaces
to OSSL pre-trained models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class OSSLReader:
    """
    Reader for OSSL format spectral data.

    The OSSL format typically uses CSV files with compressed spectra
    (first 120 principal components) or full spectral data.

    Example:
        >>> reader = OSSLReader()
        >>> spectra_df = reader.load_spectra("ossl_mir_data.csv")
        >>> print(f"Loaded {len(spectra_df)} spectra")
    """

    def __init__(self):
        """Initialize OSSL reader."""
        pass

    def load_spectra(
        self, filepath: Union[str, Path], spectral_range: str = "mir"
    ) -> pd.DataFrame:
        """
        Load OSSL format spectral data.

        Args:
            filepath: Path to OSSL CSV file
            spectral_range: Spectral range ('mir', 'nir', 'visnir')

        Returns:
            DataFrame with spectral data and metadata

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            df = pd.read_csv(filepath)

            # Validate OSSL format
            if not self._is_valid_ossl_format(df, spectral_range):
                raise ValueError(f"Invalid OSSL format in {filepath}")

            return df

        except Exception as e:
            raise ValueError(f"Error loading OSSL file {filepath}: {str(e)}")

    def load_model(self, model_name: str) -> "OSSLModel":
        """
        Load a pre-trained OSSL model.

        Args:
            model_name: Name of OSSL model (e.g., 'ossl.mir.cubist')

        Returns:
            OSSLModel instance

        Raises:
            ValueError: If model name is invalid or model not found
        """
        # Placeholder for OSSL model loading
        # Will be implemented in integration module
        raise NotImplementedError("OSSL model loading will be implemented in integration module")

    def _is_valid_ossl_format(self, df: pd.DataFrame, spectral_range: str) -> bool:
        """
        Validate OSSL dataframe format.

        Args:
            df: DataFrame to validate
            spectral_range: Expected spectral range

        Returns:
            True if valid OSSL format
        """
        # Check for expected columns
        required_columns = ["id.sample_local_c"]

        # Check for spectral columns
        if spectral_range == "mir":
            # MIR range: 600-4000 cm⁻¹
            spectral_prefix = "scan.mir."
        elif spectral_range == "nir":
            spectral_prefix = "scan.nir."
        elif spectral_range == "visnir":
            spectral_prefix = "scan.visnir."
        else:
            raise ValueError(f"Unknown spectral range: {spectral_range}")

        spectral_columns = [col for col in df.columns if col.startswith(spectral_prefix)]

        return all(col in df.columns for col in required_columns) and len(spectral_columns) > 0

    def extract_spectral_columns(
        self, df: pd.DataFrame, spectral_range: str = "mir"
    ) -> np.ndarray:
        """
        Extract spectral data as numpy array.

        Args:
            df: OSSL DataFrame
            spectral_range: Spectral range ('mir', 'nir', 'visnir')

        Returns:
            Array of shape (n_samples, n_wavelengths)
        """
        if spectral_range == "mir":
            spectral_prefix = "scan.mir."
        elif spectral_range == "nir":
            spectral_prefix = "scan.nir."
        elif spectral_range == "visnir":
            spectral_prefix = "scan.visnir."
        else:
            raise ValueError(f"Unknown spectral range: {spectral_range}")

        spectral_columns = [col for col in df.columns if col.startswith(spectral_prefix)]
        return df[spectral_columns].values

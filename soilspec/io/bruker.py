"""
Bruker OPUS file reader for spectroscopic data.

This module provides functionality to read Bruker Alpha II DRIFTS measurements
and other OPUS format spectral files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class Spectrum:
    """
    Container for spectral data with metadata.

    Attributes:
        wavenumbers: Array of wavenumber values (cm⁻¹)
        intensities: Array of intensity values (absorbance or reflectance)
        metadata: Dictionary containing measurement metadata
        spectrum_type: Type of spectrum ('absorbance', 'reflectance', etc.)
    """

    wavenumbers: np.ndarray
    intensities: np.ndarray
    metadata: Dict[str, any]
    spectrum_type: str = "absorbance"

    def __post_init__(self) -> None:
        """Validate spectrum data after initialization."""
        if len(self.wavenumbers) != len(self.intensities):
            raise ValueError("Wavenumbers and intensities must have the same length")
        if len(self.wavenumbers) == 0:
            raise ValueError("Spectrum cannot be empty")

    def to_dict(self) -> Dict:
        """Convert spectrum to dictionary format."""
        return {
            "wavenumbers": self.wavenumbers.tolist(),
            "intensities": self.intensities.tolist(),
            "metadata": self.metadata,
            "spectrum_type": self.spectrum_type,
        }


class BrukerReader:
    """
    Reader for Bruker OPUS binary files.

    Supports reading spectral data from Bruker Alpha II DRIFTS measurements
    and extracting associated metadata.

    Example:
        >>> reader = BrukerReader()
        >>> spectrum = reader.read_opus_file("sample.0")
        >>> print(f"Spectrum range: {spectrum.wavenumbers[0]}-{spectrum.wavenumbers[-1]} cm⁻¹")
    """

    def __init__(self, prefer_absorbance: bool = True):
        """
        Initialize BrukerReader.

        Args:
            prefer_absorbance: If True, prefer absorbance over reflectance when available
        """
        self.prefer_absorbance = prefer_absorbance

    def read_opus_file(self, filepath: Union[str, Path]) -> Spectrum:
        """
        Read a single OPUS file and return spectrum.

        Args:
            filepath: Path to OPUS file (.0, .1, .2, etc.)

        Returns:
            Spectrum object containing spectral data and metadata

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid or unsupported
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            from brukeropusreader import read_file

            opus_data = read_file(str(filepath))

            # Extract spectral data (prefer absorbance if available)
            if self.prefer_absorbance and "AB" in opus_data:
                spectrum_type = "absorbance"
                data = opus_data["AB"]
            elif "ScSm" in opus_data:  # Sample spectrum (reflectance)
                spectrum_type = "reflectance"
                data = opus_data["ScSm"]
            else:
                raise ValueError(f"No compatible spectrum data found in {filepath}")

            wavenumbers = np.array(data.x)
            intensities = np.array(data.y)

            # Extract metadata
            metadata = self._extract_metadata(opus_data)
            metadata["filename"] = filepath.name
            metadata["filepath"] = str(filepath)

            return Spectrum(
                wavenumbers=wavenumbers,
                intensities=intensities,
                metadata=metadata,
                spectrum_type=spectrum_type,
            )

        except ImportError:
            raise ImportError(
                "brukeropusreader library is required. Install with: pip install brukeropusreader"
            )
        except Exception as e:
            raise ValueError(f"Error reading OPUS file {filepath}: {str(e)}")

    def read_directory(
        self, dirpath: Union[str, Path], pattern: str = "*.0"
    ) -> List[Spectrum]:
        """
        Read all OPUS files matching pattern in a directory.

        Args:
            dirpath: Path to directory containing OPUS files
            pattern: Glob pattern for file matching (default: "*.0")

        Returns:
            List of Spectrum objects

        Example:
            >>> reader = BrukerReader()
            >>> spectra = reader.read_directory("data/spectra/", pattern="*.0")
            >>> print(f"Read {len(spectra)} spectra")
        """
        dirpath = Path(dirpath)
        if not dirpath.is_dir():
            raise NotADirectoryError(f"Directory not found: {dirpath}")

        spectra = []
        for filepath in sorted(dirpath.glob(pattern)):
            try:
                spectrum = self.read_opus_file(filepath)
                spectra.append(spectrum)
            except Exception as e:
                print(f"Warning: Failed to read {filepath}: {str(e)}")
                continue

        if not spectra:
            print(f"Warning: No spectra found in {dirpath} matching pattern {pattern}")

        return spectra

    def extract_metadata(self, opus_data: Dict) -> Dict[str, any]:
        """
        Extract metadata from OPUS data structure.

        Args:
            opus_data: Dictionary from brukeropusreader

        Returns:
            Dictionary containing extracted metadata
        """
        return self._extract_metadata(opus_data)

    def _extract_metadata(self, opus_data: Dict) -> Dict[str, any]:
        """
        Internal method to extract relevant metadata fields.

        Args:
            opus_data: Raw OPUS data dictionary

        Returns:
            Cleaned metadata dictionary
        """
        metadata = {}

        # Common metadata fields
        metadata_fields = [
            "acquisition_date",
            "instrument_type",
            "instrument_serial",
            "resolution",
            "scanner_velocity",
            "aperture_setting",
            "number_of_scans",
            "optical_velocity",
            "source_setting",
            "beamsplitter_setting",
            "detector_setting",
        ]

        for field in metadata_fields:
            if field in opus_data:
                metadata[field] = opus_data[field]

        return metadata

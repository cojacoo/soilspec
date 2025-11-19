"""
Unit tests for I/O module.

Tests for Bruker reader, OSSL reader, and format converters.
"""

import pytest
import numpy as np
from pathlib import Path


class TestBrukerReader:
    """Tests for BrukerReader class."""

    def test_bruker_reader_initialization(self):
        """Test BrukerReader can be instantiated."""
        from soilspec_pinn.io import BrukerReader

        reader = BrukerReader()
        assert reader is not None
        assert reader.prefer_absorbance is True

    def test_bruker_reader_with_options(self):
        """Test BrukerReader with different options."""
        from soilspec_pinn.io import BrukerReader

        reader = BrukerReader(prefer_absorbance=False)
        assert reader.prefer_absorbance is False


class TestSpectrum:
    """Tests for Spectrum data class."""

    def test_spectrum_creation(self):
        """Test creating a Spectrum object."""
        from soilspec_pinn.io import Spectrum

        wavenumbers = np.linspace(600, 4000, 1000)
        intensities = np.random.rand(1000)
        metadata = {"filename": "test.0"}

        spectrum = Spectrum(
            wavenumbers=wavenumbers,
            intensities=intensities,
            metadata=metadata,
            spectrum_type="absorbance",
        )

        assert len(spectrum.wavenumbers) == 1000
        assert len(spectrum.intensities) == 1000
        assert spectrum.spectrum_type == "absorbance"

    def test_spectrum_validation(self):
        """Test Spectrum validation."""
        from soilspec_pinn.io import Spectrum

        wavenumbers = np.linspace(600, 4000, 1000)
        intensities = np.random.rand(500)  # Wrong length
        metadata = {}

        with pytest.raises(ValueError):
            Spectrum(
                wavenumbers=wavenumbers, intensities=intensities, metadata=metadata
            )


class TestConverters:
    """Tests for format conversion functions."""

    def test_convert_to_absorbance_from_reflectance(self):
        """Test converting reflectance to absorbance."""
        from soilspec_pinn.io import convert_to_absorbance

        reflectance = np.array([0.5, 0.3, 0.7])
        absorbance = convert_to_absorbance(reflectance, input_type="reflectance")

        # A = log10(1/R)
        expected = np.log10(1.0 / reflectance)
        np.testing.assert_allclose(absorbance, expected)

    def test_convert_to_reflectance_from_absorbance(self):
        """Test converting absorbance to reflectance."""
        from soilspec_pinn.io import convert_to_reflectance

        absorbance = np.array([0.3, 0.5, 0.2])
        reflectance = convert_to_reflectance(absorbance, input_type="absorbance")

        # R = 10^(-A)
        expected = np.power(10.0, -absorbance)
        np.testing.assert_allclose(reflectance, expected)

    def test_conversion_round_trip(self):
        """Test that converting back and forth preserves data."""
        from soilspec_pinn.io import convert_to_absorbance, convert_to_reflectance

        original_reflectance = np.array([0.5, 0.3, 0.7])

        # Convert to absorbance and back
        absorbance = convert_to_absorbance(original_reflectance, input_type="reflectance")
        recovered_reflectance = convert_to_reflectance(absorbance, input_type="absorbance")

        np.testing.assert_allclose(recovered_reflectance, original_reflectance, rtol=1e-6)

"""
Tests for spectral band database parser.
"""

import pytest
import numpy as np
from soilspec_pinn.knowledge import SpectralBandDatabase


def test_load_database():
    """Test loading spectral_bands.csv"""
    bands = SpectralBandDatabase()
    assert len(bands.bands) > 100  # Should have 150+ bands
    assert 'band_start' in bands.bands.columns
    assert 'type' in bands.bands.columns


def test_get_bands_by_type():
    """Test filtering bands by type"""
    bands = SpectralBandDatabase()

    # Get organic bands
    org_bands = bands.get_bands(type='org')
    assert len(org_bands) > 0
    assert all(org_bands['type'] == 'org')

    # Get mineral bands
    min_bands = bands.get_bands(type='min')
    assert len(min_bands) > 0
    assert all(min_bands['type'] == 'min')


def test_get_bands_by_information():
    """Test filtering bands by information field"""
    bands = SpectralBandDatabase()

    # Search for clay
    clay_bands = bands.get_bands(information='Clay')
    assert len(clay_bands) > 0

    # Search for organic
    org_bands = bands.get_bands(information='Aliphatic')
    assert len(org_bands) > 0


def test_get_region_type():
    """Test getting region type for specific wavenumber"""
    bands = SpectralBandDatabase()

    # 2920 cm⁻¹ should be aliphatic C-H
    info = bands.get_region_type(2920)
    assert len(info) > 0
    assert 'Aliphatic' in info[0]['information'] or 'aliphatic' in info[0]['description'].lower()

    # 3620 cm⁻¹ should be clay OH
    info = bands.get_region_type(3620)
    assert len(info) > 0


def test_create_mask():
    """Test creating spectral masks"""
    bands = SpectralBandDatabase()

    wavenumbers = np.arange(600, 4001, 2)

    # Create organic mask
    org_mask = bands.create_mask(wavenumbers, type='org')
    assert org_mask.shape == wavenumbers.shape
    assert 0 <= org_mask.sum() <= len(wavenumbers)

    # Create mineral mask
    min_mask = bands.create_mask(wavenumbers, type='min')
    assert min_mask.shape == wavenumbers.shape


def test_get_key_regions():
    """Test getting predefined key regions"""
    bands = SpectralBandDatabase()

    regions = bands.get_key_regions()
    assert 'aliphatic_ch' in regions
    assert 'clay_oh' in regions
    assert 'carbohydrates' in regions

    # Check region format
    assert 'range' in regions['aliphatic_ch']
    assert 'description' in regions['aliphatic_ch']


def test_summarize():
    """Test database summary"""
    bands = SpectralBandDatabase()

    summary = bands.summarize()
    assert 'total_bands' in summary
    assert 'by_type' in summary
    assert 'wavenumber_range' in summary
    assert summary['total_bands'] > 100

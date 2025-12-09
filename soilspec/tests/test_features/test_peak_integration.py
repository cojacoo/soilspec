"""
Tests for peak integration feature extraction.
"""

import pytest
import numpy as np
from soilspec.features import PeakIntegrator, PhysicsInformedFeatures


@pytest.fixture
def synthetic_spectra():
    """Create synthetic spectra for testing"""
    wavenumbers = np.arange(600, 4001, 2)
    n_samples = 10
    n_wavelengths = len(wavenumbers)

    # Create random spectra
    spectra = np.random.rand(n_samples, n_wavelengths) * 0.5

    # Add some peaks at known positions
    # Peak at 2920 (aliphatic)
    idx_2920 = np.argmin(np.abs(wavenumbers - 2920))
    spectra[:, idx_2920-5:idx_2920+5] += 0.3

    # Peak at 1630 (amide/carboxylate)
    idx_1630 = np.argmin(np.abs(wavenumbers - 1630))
    spectra[:, idx_1630-5:idx_1630+5] += 0.2

    return wavenumbers, spectra


def test_peak_integrator_fit(synthetic_spectra):
    """Test fitting peak integrator"""
    wavenumbers, spectra = synthetic_spectra

    integrator = PeakIntegrator(region_selection='key')
    integrator.fit(spectra, wavenumbers=wavenumbers)

    assert integrator.regions_ is not None
    assert len(integrator.regions_) > 0
    assert integrator.wavenumbers_ is not None


def test_peak_integrator_transform(synthetic_spectra):
    """Test transforming spectra to features"""
    wavenumbers, spectra = synthetic_spectra

    integrator = PeakIntegrator(region_selection='key')
    integrator.fit(spectra, wavenumbers=wavenumbers)

    features = integrator.transform(spectra)

    assert features.shape[0] == spectra.shape[0]
    assert features.shape[1] > 0  # Should have multiple features
    assert features.shape[1] < spectra.shape[1]  # Fewer than wavelengths


def test_peak_integrator_feature_names(synthetic_spectra):
    """Test getting feature names"""
    wavenumbers, spectra = synthetic_spectra

    integrator = PeakIntegrator(region_selection='key')
    integrator.fit(spectra, wavenumbers=wavenumbers)
    features = integrator.transform(spectra)

    names = integrator.get_feature_names_out()
    assert len(names) == features.shape[1]
    assert 'aliphatic_ch' in names or 'amide' in names


def test_peak_integrator_region_selections(synthetic_spectra):
    """Test different region selection methods"""
    wavenumbers, spectra = synthetic_spectra

    # Key regions
    integrator_key = PeakIntegrator(region_selection='key')
    integrator_key.fit(spectra, wavenumbers=wavenumbers)
    features_key = integrator_key.transform(spectra)

    # Organic only
    integrator_org = PeakIntegrator(region_selection='organic')
    integrator_org.fit(spectra, wavenumbers=wavenumbers)
    features_org = integrator_org.transform(spectra)

    assert features_key.shape[1] > 0
    assert features_org.shape[1] > 0


def test_physics_informed_features(synthetic_spectra):
    """Test combined physics-informed features"""
    wavenumbers, spectra = synthetic_spectra

    extractor = PhysicsInformedFeatures(
        include_peaks=True,
        include_ratios=True,
        include_indices=True
    )

    extractor.fit(spectra, wavenumbers=wavenumbers)
    features = extractor.transform(spectra)

    assert features.shape[0] == spectra.shape[0]
    assert features.shape[1] > 0

    # Should have more features than just peaks (peaks + ratios + indices)
    names = extractor.get_feature_names_out()
    assert len(names) == features.shape[1]


def test_physics_informed_features_sklearn_compatible(synthetic_spectra):
    """Test sklearn pipeline compatibility"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    wavenumbers, spectra = synthetic_spectra

    pipeline = Pipeline([
        ('features', PhysicsInformedFeatures()),
        ('scaler', StandardScaler())
    ])

    # Fit with wavenumbers as parameter
    pipeline.fit(spectra, features__wavenumbers=wavenumbers)

    # Transform
    features = pipeline.transform(spectra)

    assert features.shape[0] == spectra.shape[0]
    # Features should be standardized (mean ≈ 0, std ≈ 1)
    assert np.abs(features.mean()) < 0.1

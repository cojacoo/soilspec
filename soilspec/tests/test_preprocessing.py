"""
Unit tests for preprocessing module.

Tests for SNV, MSC, derivatives, and other transformers.
"""

import pytest
import numpy as np
from sklearn.pipeline import Pipeline


class TestSNVTransformer:
    """Tests for SNV (Standard Normal Variate) transformer."""

    def test_snv_basic(self):
        """Test basic SNV transformation."""
        from soilspec.preprocessing import SNVTransformer

        # Create test data
        X = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 3, 3, 3, 3]])

        snv = SNVTransformer()
        X_transformed = snv.fit_transform(X)

        # Each row should have mean ≈ 0 and std ≈ 1
        means = np.mean(X_transformed, axis=1)
        stds = np.std(X_transformed, axis=1)

        np.testing.assert_allclose(means, 0, atol=1e-10)
        np.testing.assert_allclose(stds, 1, atol=1e-10)

    def test_snv_without_mean(self):
        """Test SNV without centering."""
        from soilspec.preprocessing import SNVTransformer

        X = np.array([[1, 2, 3, 4, 5]])

        snv = SNVTransformer(with_mean=False)
        X_transformed = snv.fit_transform(X)

        # Should have std ≈ 1 but mean not necessarily 0
        assert np.std(X_transformed[0]) == pytest.approx(1.0, abs=1e-10)


class TestMSCTransformer:
    """Tests for MSC (Multiplicative Scatter Correction) transformer."""

    def test_msc_basic(self):
        """Test basic MSC transformation."""
        from soilspec.preprocessing import MSCTransformer

        # Create test data with scatter effects
        reference = np.array([1, 2, 3, 4, 5])
        X = np.array(
            [
                reference * 1.5 + 0.5,  # Scaled and offset version
                reference * 2.0 + 1.0,
                reference * 0.8 + 0.2,
            ]
        )

        msc = MSCTransformer()
        X_transformed = msc.fit_transform(X)

        # Transformed spectra should be more similar to reference
        assert X_transformed.shape == X.shape


class TestDetrendTransformer:
    """Tests for baseline detrending transformer."""

    def test_detrend_linear(self):
        """Test linear detrending."""
        from soilspec.preprocessing import DetrendTransformer

        # Create spectrum with linear baseline
        x = np.arange(100)
        baseline = 2 * x + 10
        signal = np.sin(x / 10)
        spectrum = baseline + signal

        detrend = DetrendTransformer(degree=1)
        detrended = detrend.fit_transform(spectrum.reshape(1, -1))

        # Detrended spectrum should be closer to original signal
        assert detrended.shape == (1, 100)


class TestPipelineIntegration:
    """Test sklearn pipeline integration."""

    def test_pipeline_creation(self):
        """Test creating a preprocessing pipeline."""
        from soilspec.preprocessing import SNVTransformer, MSCTransformer

        pipeline = Pipeline([("snv", SNVTransformer()), ("msc", MSCTransformer())])

        assert pipeline is not None
        assert len(pipeline.steps) == 2

    def test_pipeline_transform(self):
        """Test transforming data through a pipeline."""
        from soilspec.preprocessing import SNVTransformer

        X = np.random.rand(10, 100)

        pipeline = Pipeline([("snv", SNVTransformer())])

        X_transformed = pipeline.fit_transform(X)

        assert X_transformed.shape == X.shape

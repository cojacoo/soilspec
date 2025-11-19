# SoilSpec: Evidence-Based Soil Spectroscopy Package

**Revised Plan**: 2025-11-19
**Philosophy**: Wrap proven tools (scipy, sklearn, PyTorch) with domain knowledge (spectral_bands.csv)

---

## Executive Summary

This package provides **physics-informed machine learning** for soil mid-infrared spectroscopy, combining:

1. **Domain Knowledge**: spectral_bands.csv with 150+ peak assignments
2. **Proven Methods**: PLS, Cubist, MBL (following TUBAFsoilFunctions/ADDRESS patterns)
3. **Modern ML**: Interpretable deep learning with physics-guided attention
4. **Clean Wrappers**: scipy/sklearn/PyTorch (not reimplemented)

**Target Application**: Bruker Alpha II DRIFTS measurements (600-4000 cm⁻¹)
**Core Principle**: Evidence-based, not hype-based

---

## 1. Package Architecture

### Directory Structure

```
soilspec/
├── __init__.py
├── io/                          # Data ingestion
│   ├── __init__.py
│   ├── bruker.py               # Bruker OPUS reader (brukeropusreader wrapper)
│   ├── ossl.py                 # OSSL CSV/HDF5 formats
│   ├── elementar.py            # Elementar soliTOCcube (optional)
│   ├── spectrolyzer.py         # Spectrolyzer UV-Vis (optional)
│   └── converters.py           # Format conversions
│
├── preprocessing/               # Spectral preprocessing
│   ├── __init__.py
│   ├── baseline.py             # SNV, MSC (sklearn-compatible)
│   ├── derivatives.py          # Savitzky-Golay (scipy.signal wrapper)
│   ├── smoothing.py            # Wavelet, filters (scipy.signal/pywavelets)
│   ├── resample.py             # Interpolation (scipy.interpolate wrapper)
│   ├── selection.py            # Kennard-Stone (sklearn.cluster wrapper)
│   └── decomposition.py        # Optional: Spectral unmixing
│
├── knowledge/                   # Domain knowledge integration ⭐ NEW
│   ├── __init__.py
│   ├── spectral_bands.csv      # 150+ peak assignments with references
│   ├── band_parser.py          # Query band database
│   ├── constraints.py          # Chemical rules (CEC ~ clay + SOC)
│   ├── visualization.py        # Annotate spectra with band info
│   └── indices.py              # Spectral indices library
│
├── features/                    # Feature engineering ⭐ NEW
│   ├── __init__.py
│   ├── peak_integration.py     # Integrate absorbance over band regions
│   ├── ratios.py               # Aliphatic/aromatic, organic/mineral
│   ├── functional_groups.py    # Extract functional group abundances
│   ├── transformers.py         # Sklearn-compatible feature extractors
│   └── automatic.py            # Automatic feature selection
│
├── models/
│   ├── __init__.py
│   │
│   ├── traditional/            # Classical chemometrics
│   │   ├── __init__.py
│   │   ├── pls.py              # sklearn.PLSRegression wrapper
│   │   ├── pls_enhanced.py     # PLS + physics features
│   │   ├── cubist.py           # OSSL Cubist (rpy2 or sklearn approximation)
│   │   ├── mbl.py              # Memory-based learning (sklearn KNN wrapper)
│   │   ├── random_forest.py    # With feature importance
│   │   └── ensemble.py         # Model stacking/averaging
│   │
│   ├── deep_learning/          # Modern neural networks
│   │   ├── __init__.py
│   │   ├── cnn1d.py            # 1D CNN for spectra
│   │   ├── attention.py        # Physics-guided attention ⭐ NEW
│   │   ├── multitask.py        # Multi-property learning ⭐ NEW
│   │   ├── resnet1d.py         # ResNet architecture for spectra
│   │   ├── transfer.py         # Transfer learning from OSSL
│   │   └── explainable.py      # Saliency maps, integrated gradients
│   │
│   └── hybrid/                 # Hybrid approaches ⭐ NEW
│       ├── __init__.py
│       ├── physics_regularized.py  # NN with chemical constraints
│       └── knowledge_distillation.py  # CNN → interpretable model
│
├── training/                    # Training infrastructure
│   ├── __init__.py
│   ├── trainer.py              # Generic training loop
│   ├── pinn_trainer.py         # If using spectral decomposition
│   ├── metrics.py              # R², RMSE, RPD, RPIQ, Lin's CCC
│   ├── callbacks.py            # Early stopping, LR scheduling
│   └── cross_validation.py     # Spatial/stratified CV
│
├── prediction/                  # Inference and uncertainty
│   ├── __init__.py
│   ├── predictor.py            # Unified prediction interface
│   ├── uncertainty.py          # Conformal prediction, ensembles
│   └── batch.py                # Batch processing utilities
│
├── interpretation/              # Explainable AI ⭐ NEW
│   ├── __init__.py
│   ├── feature_importance.py   # SHAP, permutation importance
│   ├── spectral_attribution.py # Integrated gradients, saliency
│   ├── chemistry_check.py      # Validate predictions chemically
│   └── report.py               # Generate interpretation reports
│
├── integration/                 # External tool integration
│   ├── __init__.py
│   ├── ossl_models.py          # Load OSSL pre-trained models
│   ├── model_zoo.py            # Pre-trained model registry
│   └── export.py               # Export to ONNX, TorchScript
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── spectral.py             # Spectral processing helpers
│   ├── validation.py           # Cross-validation tools
│   └── visualization.py        # Plotting utilities
│
├── datasets/                    # Dataset loaders
│   ├── __init__.py
│   ├── loaders.py              # OSSL, LUCAS, custom datasets
│   └── augmentation.py         # Spectral augmentation
│
├── cli.py                       # Command-line interface
└── tests/                       # Test suite
    ├── test_io.py
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_models.py
    └── test_knowledge.py
```

---

## 2. Core Modules (Detailed Specifications)

### 2.1 I/O Module (`io/`)

**Philosophy**: Use existing libraries, don't reimplement binary parsers

#### bruker.py
```python
"""
Bruker OPUS file reader using brukeropusreader library.
"""

from brukeropusreader import read_file
from dataclasses import dataclass
import numpy as np

@dataclass
class Spectrum:
    wavenumbers: np.ndarray
    intensities: np.ndarray
    metadata: dict
    spectrum_type: str = "absorbance"

class BrukerReader:
    """Wrap brukeropusreader with convenience methods."""

    def read_opus_file(self, filepath) -> Spectrum:
        """Read single OPUS file."""
        opus_data = read_file(str(filepath))

        # Prefer absorbance, fall back to reflectance
        if "AB" in opus_data:
            data = opus_data["AB"]
            spec_type = "absorbance"
        elif "ScSm" in opus_data:
            data = opus_data["ScSm"]
            spec_type = "reflectance"
        else:
            raise ValueError(f"No compatible spectrum in {filepath}")

        return Spectrum(
            wavenumbers=np.array(data.x),
            intensities=np.array(data.y),
            metadata=self._extract_metadata(opus_data),
            spectrum_type=spec_type
        )

    def read_directory(self, dirpath, pattern="*.0") -> list[Spectrum]:
        """Read all OPUS files in directory."""
        # Implementation using pathlib.Path.glob()
        pass
```

#### ossl.py
```python
"""
OSSL format handlers.
"""

import pandas as pd

class OSSLReader:
    """Read OSSL compressed spectra format."""

    def load_spectra(self, filepath) -> pd.DataFrame:
        """
        Load OSSL CSV with first 120 PCs.

        Returns:
            DataFrame with columns: sample_id, PC1, PC2, ..., PC120
        """
        return pd.read_csv(filepath)

    def load_properties(self, filepath) -> pd.DataFrame:
        """Load soil property measurements."""
        return pd.read_csv(filepath)

    def load_model(self, model_name: str):
        """Load pre-trained OSSL Cubist model."""
        # Download from OSSL model zoo if needed
        pass
```

---

### 2.2 Preprocessing Module (`preprocessing/`)

**Philosophy**: Wrap scipy/sklearn, don't reimplement algorithms

#### derivatives.py
```python
"""
Derivative calculations using scipy.signal.
"""

from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

class SavitzkyGolayDerivative(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay derivative transformer.

    Wraps scipy.signal.savgol_filter for sklearn compatibility.
    """

    def __init__(self, window_length=11, polyorder=2, deriv=0):
        """
        Args:
            window_length: Filter window size (must be odd)
            polyorder: Polynomial order
            deriv: Derivative order (0=smoothing, 1=1st deriv, 2=2nd deriv)
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Apply Savitzky-Golay filter to each spectrum (row)."""
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1  # Apply along wavelength dimension
        )
```

#### resample.py
```python
"""
Spectral resampling using scipy.interpolate.
"""

from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin

class SpectralResample(BaseEstimator, TransformerMixin):
    """
    Resample spectra to target wavenumbers.

    Uses scipy.interpolate.interp1d for interpolation.
    """

    def __init__(self, target_wavenumbers, kind='linear'):
        """
        Args:
            target_wavenumbers: Array of target wavenumber values
            kind: Interpolation method ('linear', 'cubic', 'quadratic')
        """
        self.target_wavenumbers = np.array(target_wavenumbers)
        self.kind = kind

    def fit(self, X, y=None, original_wavenumbers=None):
        """Store original wavenumbers."""
        if original_wavenumbers is None:
            raise ValueError("Must provide original_wavenumbers during fit")
        self.original_wavenumbers_ = original_wavenumbers
        return self

    def transform(self, X):
        """Interpolate each spectrum to target wavenumbers."""
        X_resampled = np.zeros((X.shape[0], len(self.target_wavenumbers)))

        for i in range(X.shape[0]):
            interpolator = interp1d(
                self.original_wavenumbers_,
                X[i, :],
                kind=self.kind,
                bounds_error=False,
                fill_value='extrapolate'
            )
            X_resampled[i, :] = interpolator(self.target_wavenumbers)

        return X_resampled
```

---

### 2.3 Knowledge Module (`knowledge/`) ⭐ **NEW**

**Philosophy**: Make spectral_bands.csv queryable and useful

#### band_parser.py
```python
"""
Parse and query spectral band assignments.
"""

import pandas as pd
import numpy as np

class SpectralBandDatabase:
    """
    Query spectral band assignments from spectral_bands.csv.

    Based on 150+ literature-referenced peak assignments.
    """

    def __init__(self, csv_path='spectral_bands.csv'):
        """Load band database."""
        self.bands = pd.read_csv(csv_path)

        # Clean up column names
        self.bands.columns = [
            'band_start', 'band_end', 'type',
            'information', 'description', 'reference'
        ]

    def get_bands(self, type=None, information=None) -> pd.DataFrame:
        """
        Query bands by type or information.

        Args:
            type: Filter by type ('org', 'min', 'water')
            information: Filter by chemical information (e.g., 'Clay minerals')

        Returns:
            DataFrame of matching band assignments
        """
        result = self.bands

        if type is not None:
            result = result[result['type'] == type]

        if information is not None:
            result = result[result['information'].str.contains(information, case=False)]

        return result

    def get_region_type(self, wavenumber: float) -> str:
        """
        Determine what a wavenumber represents.

        Args:
            wavenumber: Wavenumber in cm⁻¹

        Returns:
            Description of what this region represents
        """
        matches = self.bands[
            (self.bands['band_start'] <= wavenumber) &
            (self.bands['band_end'] >= wavenumber)
        ]

        if len(matches) == 0:
            return "Unknown region"

        # Prioritize by type: org > min > water
        for typ in ['org', 'min', 'water']:
            type_matches = matches[matches['type'] == typ]
            if len(type_matches) > 0:
                return type_matches.iloc[0]['description']

        return matches.iloc[0]['description']

    def create_mask(self, wavenumbers, type='org') -> np.ndarray:
        """
        Create binary mask for spectral regions of given type.

        Args:
            wavenumbers: Array of wavenumber values
            type: Band type ('org', 'min', 'water')

        Returns:
            Binary mask (1 where wavenumber is in type regions, 0 elsewhere)
        """
        mask = np.zeros_like(wavenumbers, dtype=float)
        type_bands = self.get_bands(type=type)

        for _, band in type_bands.iterrows():
            in_band = (wavenumbers >= band['band_start']) & \
                      (wavenumbers <= band['band_end'])
            mask[in_band] = 1.0

        return mask
```

#### constraints.py
```python
"""
Chemical consistency constraints based on domain knowledge.
"""

class ChemicalConstraints:
    """
    Encode known relationships between soil properties.

    These are empirical rules from soil science, not physics equations.
    """

    @staticmethod
    def cec_constraint(soc_percent, clay_percent):
        """
        CEC (cation exchange capacity) depends on SOC and clay.

        Empirical rule: CEC ≈ 0.5*clay + 2.0*SOC
        (coefficients vary by soil type)

        Returns:
            Expected CEC in cmol/kg
        """
        return 0.5 * clay_percent + 2.0 * soc_percent

    @staticmethod
    def texture_constraint(clay, silt, sand):
        """
        Texture fractions must sum to 100%.

        Returns:
            Residual (should be near 0)
        """
        return (clay + silt + sand) - 100.0

    @staticmethod
    def ph_buffering(soc_percent, clay_percent):
        """
        Higher SOC and clay → stronger pH buffering.

        This is qualitative, not quantitative.
        """
        return soc_percent > 2.0 or clay_percent > 20.0

    def validate_prediction(self, predictions: dict) -> dict:
        """
        Check if predictions satisfy chemical constraints.

        Args:
            predictions: {'SOC': 2.5, 'clay': 25, 'CEC': 15, ...}

        Returns:
            {'valid': True/False, 'warnings': [...]}
        """
        warnings = []

        # Check CEC consistency
        if 'SOC' in predictions and 'clay' in predictions and 'CEC' in predictions:
            expected_cec = self.cec_constraint(
                predictions['SOC'],
                predictions['clay']
            )
            if abs(predictions['CEC'] - expected_cec) > 5.0:
                warnings.append(
                    f"CEC ({predictions['CEC']}) inconsistent with "
                    f"SOC and clay (expected ~{expected_cec:.1f})"
                )

        # Check texture sum
        if 'clay' in predictions and 'silt' in predictions and 'sand' in predictions:
            residual = self.texture_constraint(
                predictions['clay'],
                predictions['silt'],
                predictions['sand']
            )
            if abs(residual) > 2.0:
                warnings.append(
                    f"Texture fractions sum to {100 + residual:.1f}%, not 100%"
                )

        return {
            'valid': len(warnings) == 0,
            'warnings': warnings
        }
```

---

### 2.4 Features Module (`features/`) ⭐ **NEW**

**Philosophy**: Extract chemically meaningful features using spectral_bands.csv

#### peak_integration.py
```python
"""
Integrate absorbance over chemically meaningful regions.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from ..knowledge import SpectralBandDatabase

class PeakIntegrator(BaseEstimator, TransformerMixin):
    """
    Extract features by integrating absorbance over spectral bands.

    Uses spectral_bands.csv to define integration regions.
    """

    def __init__(self, spectral_bands_csv='spectral_bands.csv'):
        """
        Args:
            spectral_bands_csv: Path to band assignment database
        """
        self.spectral_bands_csv = spectral_bands_csv
        self.band_db = None

    def fit(self, X, y=None, wavenumbers=None):
        """
        Fit by loading band database.

        Args:
            X: Spectra (n_samples, n_wavelengths)
            y: Target values (ignored)
            wavenumbers: Array of wavenumber values
        """
        if wavenumbers is None:
            raise ValueError("Must provide wavenumbers during fit")

        self.wavenumbers_ = wavenumbers
        self.band_db = SpectralBandDatabase(self.spectral_bands_csv)

        # Define integration regions (merge overlapping)
        self.regions_ = self._define_regions()

        return self

    def _define_regions(self):
        """Define distinct integration regions from band database."""
        regions = []

        # Key regions for soil spectroscopy
        region_defs = [
            ('org', 'Aliphates'),
            ('org', 'Aromates'),
            ('org', 'Carbohydrates'),
            ('org', 'Amide'),
            ('min', 'Clay minerals'),
            ('min', 'Carbonates'),
            ('min', 'Quartz'),
            ('min', 'Phyllosilicates'),
            ('water', 'Water'),
        ]

        for band_type, info_filter in region_defs:
            bands = self.band_db.get_bands(type=band_type, information=info_filter)
            if len(bands) > 0:
                regions.append({
                    'name': info_filter,
                    'type': band_type,
                    'bands': bands
                })

        return regions

    def transform(self, X):
        """
        Integrate absorbance over each defined region.

        Args:
            X: Spectra (n_samples, n_wavelengths)

        Returns:
            Feature matrix (n_samples, n_regions)
        """
        n_samples = X.shape[0]
        n_regions = len(self.regions_)
        features = np.zeros((n_samples, n_regions))

        for i, region in enumerate(self.regions_):
            # Integrate over all bands in this region
            region_integral = np.zeros(n_samples)

            for _, band in region['bands'].iterrows():
                # Find wavenumbers in this band
                mask = (self.wavenumbers_ >= band['band_start']) & \
                       (self.wavenumbers_ <= band['band_end'])

                if np.any(mask):
                    # Trapezoidal integration
                    for sample_idx in range(n_samples):
                        region_integral[sample_idx] += np.trapz(
                            X[sample_idx, mask],
                            self.wavenumbers_[mask]
                        )

            features[:, i] = region_integral

        return features

    def get_feature_names(self):
        """Get names of extracted features."""
        return [region['name'] for region in self.regions_]
```

#### ratios.py
```python
"""
Compute spectral ratios based on domain knowledge.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SpectralRatios(BaseEstimator, TransformerMixin):
    """
    Compute chemically meaningful spectral ratios.

    Based on known relationships in soil spectroscopy literature.
    """

    def fit(self, X, y=None):
        """
        Fit (no-op for ratios).

        Args:
            X: Features from PeakIntegrator (n_samples, n_features)
        """
        return self

    def transform(self, X):
        """
        Compute ratios between features.

        Args:
            X: Features (n_samples, n_features)
               Expected columns: Aliphates, Aromates, Carbohydrates,
                                Amide, Clay minerals, Carbonates, etc.

        Returns:
            Ratio features (n_samples, n_ratios)
        """
        # Assume X comes from PeakIntegrator with known column order
        features = {}

        # Aliphatic / Aromatic ratio
        # High ratio → fresh organic matter, Low ratio → humified
        if 'Aliphates' in self.feature_names_ and 'Aromates' in self.feature_names_:
            idx_aliph = self.feature_names_.index('Aliphates')
            idx_arom = self.feature_names_.index('Aromates')
            features['aliphatic_aromatic_ratio'] = \
                X[:, idx_aliph] / (X[:, idx_arom] + 1e-6)

        # Organic / Mineral ratio
        # Proxy for organic matter content
        org_cols = [i for i, name in enumerate(self.feature_names_)
                    if 'org' in self.feature_types_[i]]
        min_cols = [i for i, name in enumerate(self.feature_names_)
                    if 'min' in self.feature_types_[i]]

        if org_cols and min_cols:
            features['organic_mineral_ratio'] = \
                X[:, org_cols].sum(axis=1) / (X[:, min_cols].sum(axis=1) + 1e-6)

        # Carbohydrate / Amide ratio
        # Proxy for plant-derived vs microbial organic matter
        if 'Carbohydrates' in self.feature_names_ and 'Amide' in self.feature_names_:
            idx_carb = self.feature_names_.index('Carbohydrates')
            idx_amide = self.feature_names_.index('Amide')
            features['carbohydrate_amide_ratio'] = \
                X[:, idx_carb] / (X[:, idx_amide] + 1e-6)

        # Convert to array
        ratio_matrix = np.column_stack([features[k] for k in sorted(features.keys())])
        self.ratio_names_ = sorted(features.keys())

        return ratio_matrix

    def get_feature_names(self):
        """Get names of ratio features."""
        return self.ratio_names_
```

---

### 2.5 Models Module

#### Traditional Models (`models/traditional/`)

**pls_enhanced.py** - PLS with physics-informed features
```python
"""
Enhanced PLS using physics-informed features.
"""

from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from ...features import PeakIntegrator, SpectralRatios

class EnhancedPLS:
    """
    PLS regression with physics-informed feature engineering.

    Combines domain knowledge (spectral_bands.csv) with PLS modeling.
    """

    def __init__(self, n_components=10, spectral_bands_csv='spectral_bands.csv'):
        """
        Args:
            n_components: Number of PLS components
            spectral_bands_csv: Path to band assignment database
        """
        self.pipeline = Pipeline([
            ('peak_integration', PeakIntegrator(spectral_bands_csv)),
            ('ratios', SpectralRatios()),
            ('pls', PLSRegression(n_components=n_components))
        ])

    def fit(self, X, y, wavenumbers):
        """
        Fit PLS model with feature extraction.

        Args:
            X: Spectra (n_samples, n_wavelengths)
            y: Target values (n_samples,)
            wavenumbers: Wavenumber array
        """
        # Pass wavenumbers to peak integrator
        self.pipeline.named_steps['peak_integration'].fit(X, y, wavenumbers=wavenumbers)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Predict using fitted model."""
        return self.pipeline.predict(X)

    def score(self, X, y):
        """R² score."""
        return self.pipeline.score(X, y)

    def get_feature_importance(self):
        """
        Get importance of chemical features.

        Returns:
            DataFrame with feature names and PLS coefficients
        """
        peak_names = self.pipeline.named_steps['peak_integration'].get_feature_names()
        ratio_names = self.pipeline.named_steps['ratios'].get_feature_names()
        feature_names = peak_names + ratio_names

        pls_coef = self.pipeline.named_steps['pls'].coef_

        return pd.DataFrame({
            'feature': feature_names,
            'coefficient': pls_coef.flatten()
        }).sort_values('coefficient', ascending=False)
```

**mbl.py** - Memory-Based Learning
```python
"""
Memory-Based Learning for spectroscopy.

Based on resemble R package methodology (from ADDRESS/TUBAFsoilFunctions).
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import PLSRegression
import numpy as np

class MBLPredictor:
    """
    Memory-Based Learning predictor.

    Makes predictions using k-nearest neighbors in spectral space,
    fitting local PLS models.
    """

    def __init__(
        self,
        k_neighbors=50,
        similarity_metric='cosine',
        local_model='pls',
        n_components=10
    ):
        """
        Args:
            k_neighbors: Number of nearest neighbors
            similarity_metric: 'cosine', 'euclidean', 'mahalanobis'
            local_model: 'pls', 'ridge', 'mean'
            n_components: PLS components for local model
        """
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.local_model = local_model
        self.n_components = n_components

    def fit(self, X_cal, y_cal):
        """
        Store calibration (memory) set.

        Args:
            X_cal: Calibration spectra (n_samples, n_features)
            y_cal: Calibration values (n_samples,)
        """
        self.X_cal = X_cal
        self.y_cal = y_cal

        # Fit neighbor search
        self.nn = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            metric=self.similarity_metric
        )
        self.nn.fit(X_cal)

        return self

    def predict(self, X_pred):
        """
        Predict using local models.

        For each prediction sample:
        1. Find k nearest neighbors in calibration set
        2. Fit local PLS model on neighbors
        3. Predict using local model
        """
        predictions = np.zeros(X_pred.shape[0])

        for i in range(X_pred.shape[0]):
            spectrum = X_pred[i:i+1, :]

            # Find nearest neighbors
            distances, indices = self.nn.kneighbors(spectrum)
            neighbor_indices = indices[0]

            # Get neighbor data
            X_neighbors = self.X_cal[neighbor_indices]
            y_neighbors = self.y_cal[neighbor_indices]

            # Fit local model
            if self.local_model == 'pls':
                local_model = PLSRegression(n_components=self.n_components)
                local_model.fit(X_neighbors, y_neighbors)
                pred = local_model.predict(spectrum)
            elif self.local_model == 'mean':
                pred = np.mean(y_neighbors)
            else:
                raise ValueError(f"Unknown local model: {self.local_model}")

            predictions[i] = pred

        return predictions

    def predict_with_uncertainty(self, X_pred):
        """
        Predict with uncertainty from local variance.

        Returns:
            predictions, uncertainties
        """
        predictions = np.zeros(X_pred.shape[0])
        uncertainties = np.zeros(X_pred.shape[0])

        for i in range(X_pred.shape[0]):
            spectrum = X_pred[i:i+1, :]

            # Find nearest neighbors
            distances, indices = self.nn.kneighbors(spectrum)
            neighbor_indices = indices[0]

            # Get neighbor data
            y_neighbors = self.y_cal[neighbor_indices]

            # Prediction = local mean
            predictions[i] = np.mean(y_neighbors)

            # Uncertainty = local standard deviation
            uncertainties[i] = np.std(y_neighbors)

        return predictions, uncertainties
```

#### Deep Learning Models (`models/deep_learning/`)

**attention.py** - Physics-Guided Attention ⭐ **KEY INNOVATION**
```python
"""
Physics-guided attention mechanism for spectral analysis.

Uses spectral_bands.csv to guide where the network should focus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...knowledge import SpectralBandDatabase

class PhysicsGuidedAttention(nn.Module):
    """
    Attention mechanism guided by spectral band assignments.

    Learns to focus on chemically meaningful regions defined in
    spectral_bands.csv while remaining flexible.
    """

    def __init__(self, n_wavelengths, wavenumbers, spectral_bands_csv):
        """
        Args:
            n_wavelengths: Number of wavelength points
            wavenumbers: Array of wavenumber values
            spectral_bands_csv: Path to band database
        """
        super().__init__()

        self.n_wavelengths = n_wavelengths

        # Load spectral band knowledge
        band_db = SpectralBandDatabase(spectral_bands_csv)

        # Create masks for different chemical types
        self.register_buffer('organic_mask',
                           torch.from_numpy(band_db.create_mask(wavenumbers, 'org')).float())
        self.register_buffer('mineral_mask',
                           torch.from_numpy(band_db.create_mask(wavenumbers, 'min')).float())
        self.register_buffer('water_mask',
                           torch.from_numpy(band_db.create_mask(wavenumbers, 'water')).float())

        # Learnable attention network
        self.attention_net = nn.Sequential(
            nn.Linear(n_wavelengths, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_wavelengths),
            nn.Softmax(dim=-1)
        )

        # Physics-based attention priors
        self.organic_importance = nn.Parameter(torch.tensor(1.0))
        self.mineral_importance = nn.Parameter(torch.tensor(0.5))
        self.water_importance = nn.Parameter(torch.tensor(0.1))

    def forward(self, spectrum):
        """
        Apply physics-guided attention.

        Args:
            spectrum: Input spectrum (batch_size, n_wavelengths)

        Returns:
            attended_spectrum: Spectrum with attention applied
            attention_weights: Where the model is looking
            physics_loss: Regularization encouraging attention on known regions
        """
        # Compute attention weights from spectrum
        attention_weights = self.attention_net(spectrum)

        # Compute physics-based prior
        physics_prior = (
            self.organic_importance * self.organic_mask +
            self.mineral_importance * self.mineral_mask +
            self.water_importance * self.water_mask
        )
        physics_prior = physics_prior / physics_prior.sum()  # Normalize

        # Regularization: attention should correlate with physics prior
        physics_loss = F.kl_div(
            torch.log(attention_weights + 1e-10),
            physics_prior.unsqueeze(0).expand_as(attention_weights),
            reduction='batchmean'
        )

        # Apply attention
        attended_spectrum = spectrum * attention_weights

        return attended_spectrum, attention_weights, physics_loss


class CNN1D_with_PhysicsAttention(nn.Module):
    """
    1D CNN with physics-guided attention for soil property prediction.
    """

    def __init__(
        self,
        n_wavelengths,
        wavenumbers,
        n_outputs=1,
        spectral_bands_csv='spectral_bands.csv',
        physics_weight=0.1
    ):
        """
        Args:
            n_wavelengths: Number of wavelength points
            wavenumbers: Wavenumber array
            n_outputs: Number of properties to predict
            spectral_bands_csv: Path to band database
            physics_weight: Weight for physics regularization loss
        """
        super().__init__()

        self.physics_weight = physics_weight

        # Physics-guided attention
        self.attention = PhysicsGuidedAttention(
            n_wavelengths, wavenumbers, spectral_bands_csv
        )

        # 1D CNN backbone
        self.conv1 = nn.Conv1d(1, 32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=11, padding=5)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_outputs)
        )

    def forward(self, spectrum):
        """
        Forward pass with physics-guided attention.

        Returns:
            prediction, attention_weights, total_loss
        """
        # Apply physics-guided attention
        x, attention_weights, physics_loss = self.attention(spectrum)

        # Reshape for Conv1d (batch, channels, length)
        x = x.unsqueeze(1)

        # CNN processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.pool(x).squeeze(-1)

        # Prediction
        prediction = self.fc(x)

        return prediction, attention_weights, physics_loss

    def compute_loss(self, prediction, target, physics_loss):
        """
        Combined loss: prediction error + physics regularization.

        Args:
            prediction: Model output
            target: True values
            physics_loss: Physics regularization from attention

        Returns:
            total_loss, data_loss, weighted_physics_loss
        """
        data_loss = F.mse_loss(prediction, target)
        total_loss = data_loss + self.physics_weight * physics_loss

        return total_loss, data_loss, physics_loss
```

**multitask.py** - Multi-Task Learning with Chemical Constraints
```python
"""
Multi-task learning with chemical consistency constraints.
"""

import torch
import torch.nn as nn
from ...knowledge import ChemicalConstraints

class ChemicallyConstrainedMTL(nn.Module):
    """
    Multi-task CNN predicting multiple soil properties.

    Enforces chemical consistency (e.g., CEC ~ clay + SOC).
    """

    def __init__(self, n_wavelengths, tasks=['SOC', 'clay', 'CEC']):
        """
        Args:
            n_wavelengths: Number of spectral points
            tasks: List of properties to predict
        """
        super().__init__()

        self.tasks = tasks
        self.constraints = ChemicalConstraints()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Task-specific heads
        self.heads = nn.ModuleDict({
            task: nn.Linear(128, 1)
            for task in tasks
        })

    def forward(self, spectrum):
        """
        Predict all tasks.

        Returns:
            predictions: Dict {task: value}
        """
        # Reshape for Conv1d
        x = spectrum.unsqueeze(1)

        # Shared features
        features = self.encoder(x).squeeze(-1)

        # Task predictions
        predictions = {
            task: self.heads[task](features).squeeze(-1)
            for task in self.tasks
        }

        return predictions

    def compute_loss(self, predictions, targets):
        """
        Loss with chemical consistency regularization.

        Args:
            predictions: Dict of predicted values
            targets: Dict of true values

        Returns:
            total_loss, data_loss, chemistry_loss
        """
        # Standard prediction losses
        data_loss = sum(
            F.mse_loss(predictions[task], targets[task])
            for task in self.tasks
        )

        # Chemical consistency loss
        chemistry_loss = 0.0

        if 'SOC' in predictions and 'clay' in predictions and 'CEC' in predictions:
            # CEC should match chemical expectation
            expected_cec = self.constraints.cec_constraint(
                predictions['SOC'],
                predictions['clay']
            )
            chemistry_loss += F.mse_loss(predictions['CEC'], expected_cec)

        # Texture constraint (if predicting clay, silt, sand)
        if all(task in predictions for task in ['clay', 'silt', 'sand']):
            texture_sum = predictions['clay'] + predictions['silt'] + predictions['sand']
            texture_target = torch.ones_like(texture_sum) * 100.0
            chemistry_loss += F.mse_loss(texture_sum, texture_target)

        total_loss = data_loss + 0.1 * chemistry_loss

        return total_loss, data_loss, chemistry_loss
```

---

### 2.6 Interpretation Module (`interpretation/`) ⭐ **NEW**

**spectral_attribution.py** - Explain predictions via spectral bands
```python
"""
Explain model predictions using spectral band attribution.
"""

import torch
import numpy as np
from ..knowledge import SpectralBandDatabase

class SpectralSaliency:
    """
    Compute saliency maps showing which wavelengths influenced prediction.

    Maps attribution back to chemical bands from spectral_bands.csv.
    """

    def __init__(self, model, spectral_bands_csv='spectral_bands.csv'):
        """
        Args:
            model: Trained PyTorch model
            spectral_bands_csv: Path to band database
        """
        self.model = model
        self.band_db = SpectralBandDatabase(spectral_bands_csv)

    def compute_gradient_saliency(self, spectrum, wavenumbers):
        """
        Compute gradient-based saliency.

        Args:
            spectrum: Input spectrum (tensor)
            wavenumbers: Wavenumber array

        Returns:
            saliency: Importance of each wavelength
            top_bands: Chemical interpretation of top regions
        """
        spectrum = spectrum.clone().requires_grad_(True)

        # Forward pass
        prediction = self.model(spectrum.unsqueeze(0))

        # Backward pass
        prediction.backward()

        # Saliency = abs(gradient)
        saliency = spectrum.grad.abs().cpu().numpy()

        # Interpret top saliency regions
        top_indices = np.argsort(saliency)[-20:]  # Top 20 wavelengths
        top_wavenumbers = wavenumbers[top_indices]

        top_bands = []
        for wn in top_wavenumbers:
            description = self.band_db.get_region_type(wn)
            top_bands.append({
                'wavenumber': wn,
                'saliency': saliency[wn],
                'chemical_meaning': description
            })

        return saliency, top_bands

    def plot_with_bands(self, spectrum, wavenumbers, saliency):
        """
        Plot spectrum with saliency overlay and band annotations.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Spectrum
        ax1.plot(wavenumbers, spectrum, 'k-', linewidth=1)
        ax1.set_ylabel('Absorbance')
        ax1.set_title('Spectrum')

        # Annotate key bands
        key_bands = self.band_db.bands.sort_values('band_start').head(10)
        for _, band in key_bands.iterrows():
            ax1.axvspan(band['band_start'], band['band_end'],
                       alpha=0.2, label=band['information'])

        # Saliency map
        ax2.plot(wavenumbers, saliency, 'r-', linewidth=1)
        ax2.set_xlabel('Wavenumber (cm⁻¹)')
        ax2.set_ylabel('Importance')
        ax2.set_title('Spectral Attribution (what model uses for prediction)')

        plt.tight_layout()
        return fig
```

---

## 3. Implementation Roadmap (Revised)

### Phase 1: Core Infrastructure (Weeks 1-2) ✅ Mostly Done

- [x] Package structure
- [x] Bruker OPUS reader
- [x] OSSL integration basics
- [x] Basic preprocessing (SNV, MSC, Detrend)
- [ ] Complete preprocessing with scipy wrappers

**Deliverable**: Can read Bruker files and apply basic preprocessing

### Phase 2: Knowledge Integration (Weeks 3-4) ⭐ **PRIORITY**

- [ ] Load and parse spectral_bands.csv
- [ ] Query interface for band assignments
- [ ] Chemical constraints module
- [ ] Physics-informed feature extraction (peak integration, ratios)
- [ ] Visualization with band annotations

**Deliverable**: Extract ~50 chemically meaningful features from spectra

### Phase 3: Traditional ML (Weeks 5-6)

- [ ] PLS baseline (sklearn wrapper)
- [ ] Enhanced PLS with physics features
- [ ] MBL implementation
- [ ] Cubist wrapper (or sklearn approximation)
- [ ] Random Forest with feature importance
- [ ] Ensemble methods

**Deliverable**: Suite of traditional methods matching ADDRESS/TUBAFsoilFunctions

### Phase 4: Interpretable Deep Learning (Weeks 7-9)

- [ ] Basic 1D CNN
- [ ] Physics-guided attention mechanism
- [ ] Multi-task learning with chemical constraints
- [ ] Spectral saliency/attribution
- [ ] Attention visualization

**Deliverable**: Deep learning that explains its predictions

### Phase 5: Validation & Testing (Weeks 10-12)

- [ ] OSSL benchmark testing
- [ ] Comparison: PLS vs PLS+features vs MBL vs CNN
- [ ] Uncertainty quantification
- [ ] Transfer learning experiments
- [ ] Comprehensive test suite (>90% coverage)

**Deliverable**: Published benchmarks showing performance gains

---

## 4. Success Criteria (Evidence-Based)

### Technical Performance

| Method | Target R² (SOC) | Advantage Over Baseline |
|--------|----------------|-------------------------|
| PLS baseline | 0.80-0.82 | Industry standard |
| PLS + physics features | 0.83-0.85 | +3-5% from domain knowledge |
| Random Forest + features | 0.84-0.87 | Non-linear relationships |
| MBL | 0.82-0.88 | Instrument transfer |
| Cubist | 0.83-0.87 | OSSL compatibility |
| 1D CNN (10k+ samples) | 0.87-0.90 | Automatic feature learning |
| CNN + attention | 0.88-0.92 | Physics-guided + interpretable |
| Multi-task CNN | 0.90-0.93 | Leverage property correlations |

### Interpretability

✅ Can map predictions to chemical bands from spectral_bands.csv
✅ Attention visualizations show chemically meaningful regions
✅ Feature importance aligns with domain knowledge
✅ Predictions satisfy chemical constraints (CEC, texture sum, etc.)

### Software Quality

✅ Sklearn-compatible API for all transformers
✅ >90% test coverage
✅ Processing speed: >1000 spectra/sec (CPU preprocessing)
✅ Complete documentation with chemical context
✅ Example notebooks matching simplerspec workflows

---

## 5. Dependencies (Minimal, Proven)

### Core (Required)

```toml
[dependencies]
numpy = ">=1.24.0"
scipy = ">=1.10.0"              # For signal processing (not reimplemented)
pandas = ">=2.0.0"
scikit-learn = ">=1.3.0"        # For PLS, transformers, metrics
matplotlib = ">=3.7.0"
brukeropusreader = ">=1.3.0"    # For OPUS files
```

### Deep Learning (Optional)

```toml
[optional-dependencies.deep]
torch = ">=2.0.0"
lightning = ">=2.0.0"
```

### Advanced (Optional)

```toml
[optional-dependencies.advanced]
pywavelets = ">=1.4.0"          # For wavelet denoising
shap = ">=0.42.0"               # For feature importance
```

### Development

```toml
[optional-dependencies.dev]
pytest = ">=7.4.0"
pytest-cov = ">=4.1.0"
black = ">=23.0.0"
ruff = ">=0.1.0"
jupyter = ">=1.0.0"
```

**NO**: rdkit, torch-geometric, custom PINN libraries

---

## 6. Documentation Structure

```
docs/
├── index.md
├── getting_started/
│   ├── installation.md
│   ├── quick_start.md
│   └── bruker_workflow.md       # Like TUBAFsoilFunctions docs
│
├── tutorials/
│   ├── 01_reading_spectra.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb  # Using spectral_bands.csv
│   ├── 04_traditional_ml.ipynb       # PLS, MBL, Cubist
│   ├── 05_deep_learning.ipynb        # CNN with attention
│   ├── 06_interpretation.ipynb       # Explainable AI
│   └── 07_ossl_integration.ipynb
│
├── knowledge_base/
│   ├── spectral_bands_reference.md   # Detailed band explanations
│   ├── chemical_constraints.md       # Soil chemistry rules
│   └── literature_review.md          # What works in practice
│
└── api/
    ├── io.md
    ├── preprocessing.md
    ├── knowledge.md                  # NEW
    ├── features.md                   # NEW
    ├── models.md
    └── interpretation.md             # NEW
```

---

## 7. Key Differences from Original Plan

| Aspect | Original Plan | Revised Plan | Reason |
|--------|--------------|--------------|---------|
| **Core concept** | PINN for prediction | Physics-informed features + interpretable ML | No PDEs to enforce |
| **Preprocessing** | Custom implementations | Scipy/sklearn wrappers | Don't reinvent |
| **Features** | Raw spectra only | Chemical features from spectral_bands.csv | Domain knowledge |
| **Models** | PINN, MPNN, U-nets | PLS, MBL, CNN with attention | Evidence-based |
| **Interpretation** | Not emphasized | Central (spectral attribution) | Explainable AI |
| **Philosophy** | Novel deep learning | Hybrid: domain + data-driven | Practical |

---

## 8. Example Workflow Comparison

### ADDRESS/TUBAFsoilFunctions Pattern (R)

```r
# R workflow we're matching
library(simplerspec)
library(prospectr)

spectra <- read_opus_univ("bruker_files/")
spectra <- preprocess_spc(spectra, method = "savgol", deriv = 1)
model <- fit_pls(spectra, soil_properties)
predictions <- predict(model, new_spectra)
```

### Our Package (Python)

```python
# Equivalent Python workflow
from soilspec.io import BrukerReader
from soilspec.preprocessing import SavitzkyGolayDerivative
from soilspec.models.traditional import EnhancedPLS

# Read spectra
reader = BrukerReader()
spectra = reader.read_directory("bruker_files/")

# Preprocess (using scipy)
preprocessor = SavitzkyGolayDerivative(deriv=1)
spectra_processed = preprocessor.fit_transform(spectra)

# Model with physics-informed features
model = EnhancedPLS(spectral_bands_csv='spectral_bands.csv')
model.fit(spectra_processed, soil_properties, wavenumbers=wavenumbers)

# Predict
predictions = model.predict(new_spectra)

# Interpret (NEW)
feature_importance = model.get_feature_importance()
# Shows: "Aliphatic C-H (2920 cm⁻¹) most important for SOC"
```

---

## Conclusion

This revised plan:

✅ Uses spectral_bands.csv for **intelligent feature engineering**
✅ Wraps proven tools (scipy, sklearn, PyTorch) like ADDRESS/TUBAFsoilFunctions
✅ Focuses on **evidence-based methods** (PLS, MBL, Cubist, CNN)
✅ Adds **interpretable deep learning** (physics-guided attention)
✅ Provides **chemical validation** of predictions
✅ Skips PINN/MPNN for prediction (no differential equations)

**Bottom Line**: Build a practical, interpretable soil spectroscopy package that combines domain knowledge with modern ML, not hype-driven architectures without evidence.

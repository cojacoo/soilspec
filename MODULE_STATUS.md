# Module Status and Missing Components

## Overview

This document audits all modules for completeness after integrating with Albinet et al.'s packages.

## Status Summary

| Module | Status | Action Needed |
|--------|--------|---------------|
| **io/bruker.py** | ✅ Complete | None - works well |
| **io/converters.py** | ✅ Complete | None - useful utilities |
| **io/ossl.py** | ⚠️ Obsolete | Update or deprecate (replaced by datasets.OSSLDataset) |
| **utils/** | ❌ Broken imports | Implement missing files or update __init__.py |
| **integration/** | ❌ Broken imports | Implement or remove stubs |
| **prediction/** | ❌ Broken imports | Implement or remove stubs |
| **datasets/** | ✅ Complete | Working with soilspecdata integration |
| **models/** | ✅ Complete | Traditional + deep learning implemented |
| **training/** | ✅ Complete | Full infrastructure implemented |
| **preprocessing/** | ✅ Complete | All methods implemented |
| **features/** | ✅ Complete | Physics-informed features working |
| **knowledge/** | ✅ Complete | Spectral bands + constraints |
| **experimental/** | ✅ Complete | GADF with warnings |

## Detailed Analysis

### 1. io/ossl.py - OBSOLETE ⚠️

**Current state:**
- Stub for reading OSSL CSV files
- Basic validation of OSSL format
- Placeholder for model loading

**Problem:**
- **Replaced by `soilspec.datasets.OSSLDataset`** which uses Albinet's `soilspecdata` package
- `soilspecdata` handles download, caching, and data alignment automatically
- Our OSSLDataset is more powerful and easier to use

**Recommendation:**
```python
# OLD (io/ossl.py):
reader = OSSLReader()
df = reader.load_spectra("ossl_mir_data.csv")  # Manual CSV loading

# NEW (datasets.OSSLDataset):
from soilspec.datasets import OSSLDataset
ossl = OSSLDataset()
X, y, ids = ossl.load_mir(target='soc')  # Automatic download + alignment
```

**Action:**
- Mark `io/ossl.py` as deprecated
- Add deprecation warning pointing to `datasets.OSSLDataset`
- OR remove entirely since datasets.OSSLDataset is superior

### 2. utils/ - BROKEN IMPORTS ❌

**Current __init__.py references:**
```python
from soilspec.utils.spectral import interpolate_spectrum, baseline_als
from soilspec.utils.validation import train_test_split_spectral, cross_validate
from soilspec.utils.visualization import plot_spectrum, plot_predictions
```

**Problem:** None of these files exist!

**Analysis:**

**a) utils/spectral.py - NEEDED?**
- `interpolate_spectrum`: Already have `preprocessing.SpectralResample`
- `baseline_als`: Asymmetric Least Squares baseline - USEFUL, not implemented elsewhere

**b) utils/validation.py - MOSTLY REDUNDANT**
- `train_test_split_spectral`: We have `datasets.OSSLDataset.split_dataset()` and sklearn's `train_test_split`
- `cross_validate`: sklearn already provides this

**c) utils/visualization.py - NEEDED**
- `plot_spectrum`: Useful utility for quick spectrum plotting
- `plot_predictions`: Already have in `training.GenericTrainer.plot_predictions()`

**Recommendation:**
1. **Implement minimal utils**:
   - `spectral.py`: baseline_als (useful addition)
   - `visualization.py`: plot_spectrum, plot_spectra_comparison
2. **Remove redundant**:
   - validation.py (use sklearn + our OSSLDataset)

### 3. integration/ - BROKEN IMPORTS ❌

**Current __init__.py references:**
```python
from soilspec.integration.ossl_models import OSSLModelLoader
from soilspec.integration.model_zoo import ModelZoo, download_pretrained_model
```

**Problem:** Files don't exist

**Analysis:**

**What was this meant to do?**
- Load pre-trained OSSL Cubist models
- Model zoo for downloading pretrained models

**Current status:**
- OSSL models: We have `models.traditional.OSSLCubistPredictor` which trains from scratch
- Pre-trained models: No public repository of pre-trained soil spectroscopy models exists

**Recommendation:**
- **FUTURE FEATURE**: Implement when we have pre-trained models to share
- **For now**: Remove from __init__.py or mark as NotImplementedError with clear message

### 4. prediction/ - BROKEN IMPORTS ❌

**Current __init__.py references:**
```python
from soilspec.prediction.predictor import SpectralPredictor
from soilspec.prediction.uncertainty import (
    EnsembleUncertainty,
    ConformalPrediction,
    DropoutUncertainty,
)
```

**Problem:** Files don't exist

**Analysis:**

**What was this meant to do?**
- Unified prediction interface
- Uncertainty quantification methods

**Current status:**
- Prediction: Already works via `GenericTrainer.predict()` and `DeepLearningTrainer.predict()`
- Uncertainty: `MBLRegressor.predict_with_uncertainty()` provides local variance
- Conformal/Dropout: Advanced features not yet implemented

**Recommendation:**
- **SpectralPredictor**: Could be useful wrapper, but low priority
- **Uncertainty methods**: Advanced features for v0.2.0
- **For now**: Remove from __init__.py or implement basic wrappers

## Recommendations

### High Priority (Fix Now) ✅

1. **Fix broken imports** - package currently fails on import:
   ```python
   from soilspec import utils  # ImportError!
   from soilspec import integration  # ImportError!
   from soilspec import prediction  # ImportError!
   ```

2. **Deprecate io/ossl.py** - Replaced by better solution

3. **Implement minimal utils**:
   - baseline_als (useful)
   - plot_spectrum (convenience)

### Medium Priority (v0.1.1) 📋

4. **Integration module**: Implement basic model loading/saving utilities

5. **Prediction module**: Simple unified prediction wrapper

### Low Priority (v0.2.0) 📅

6. **Advanced uncertainty**: Conformal prediction, dropout uncertainty

7. **Model zoo**: Pre-trained model repository (requires infrastructure)

## Proposed Actions

### Option A: Fix Everything (Complete) 🔨

Implement all missing modules with full functionality.

**Pros:** Complete package, no broken imports
**Cons:** Adds complexity, duplicates sklearn functionality
**Effort:** ~500 lines of code

### Option B: Clean Up (Minimal) 🧹

Remove broken imports, keep only what's truly needed.

**Pros:** Clean, focused package
**Cons:** Some planned features removed
**Effort:** ~50 lines (mostly deletions)

### Option C: Hybrid (Recommended) ⚡

Fix critical issues, implement useful additions, remove redundancy.

**Action plan:**
1. ✅ Fix utils: Implement baseline_als, plot_spectrum
2. ✅ Update io/ossl.py: Add deprecation warning → use datasets.OSSLDataset
3. ❌ Remove integration __init__ imports (keep module for future)
4. ❌ Remove prediction __init__ imports (keep module for future)
5. ✅ Add TODO comments for future features

**Pros:** Package works, focused on core functionality, clear roadmap
**Cons:** Some modules incomplete
**Effort:** ~200 lines

## Current Import Status

**What works:**
```python
from soilspec.datasets import OSSLDataset  # ✅
from soilspec.models.traditional import MBLRegressor, CubistRegressor  # ✅
from soilspec.models.deep_learning import SimpleCNN1D  # ✅
from soilspec.training import GenericTrainer  # ✅
from soilspec.preprocessing import SNVTransformer  # ✅
from soilspec.features import PhysicsInformedFeatures  # ✅
```

**What fails:**
```python
from soilspec import utils  # ❌ ImportError
from soilspec import integration  # ❌ ImportError
from soilspec import prediction  # ❌ ImportError
from soilspec.io import OSSLReader  # ⚠️ Works but obsolete
```

## Decision Needed

Which approach should we take?
- **Option A**: Implement everything
- **Option B**: Remove unimplemented features
- **Option C**: Hybrid approach (recommended)

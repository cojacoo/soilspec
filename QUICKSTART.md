# Quick Start Guide

## Installation

```bash
cd soilspec
pip install -e ".[dev]"  # Install with development dependencies
```

## Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check CLI
soilspec version
```

## Example 1: Memory-Based Learning (MBL)

The MBL module is fully implemented and ready to use, based on the saxSSL methodology:

```python
import numpy as np
from soilspec_pinn.models.mbl import MBLPredictor

# Create synthetic data for demonstration
n_cal = 200  # Calibration samples
n_test = 50  # Test samples
n_features = 1701  # Spectral points (e.g., 600-4000 cm⁻¹ at 2 cm⁻¹ intervals)

# Simulate spectra (replace with real data)
X_cal = np.random.randn(n_cal, n_features)
y_cal = np.random.randn(n_cal)  # Soil property (e.g., SOC)

X_test = np.random.randn(n_test, n_features)
y_test = np.random.randn(n_test)

# Initialize MBL predictor
mbl = MBLPredictor(
    k_neighbors=50,                    # Number of nearest neighbors
    similarity_metric='mahalanobis',   # Spectral similarity metric
    weighting='gaussian',              # Neighbor weighting scheme
    local_model='pls',                 # Local model type
    n_components=10                    # PLS components
)

# Fit (store calibration data)
mbl.fit(X_cal, y_cal)

# Predict with uncertainty
predictions, uncertainties = mbl.predict(X_test, return_uncertainty=True)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")
```

## Example 2: Reading Bruker OPUS Files

```python
from soilspec_pinn.io import BrukerReader

# Initialize reader
reader = BrukerReader(prefer_absorbance=True)

# Read single file
spectrum = reader.read_opus_file("data/sample.0")

print(f"Wavenumber range: {spectrum.wavenumbers[0]:.1f} - {spectrum.wavenumbers[-1]:.1f} cm⁻¹")
print(f"Number of points: {len(spectrum.wavenumbers)}")
print(f"Spectrum type: {spectrum.spectrum_type}")
print(f"Metadata: {spectrum.metadata}")

# Read directory
spectra = reader.read_directory("data/spectra/", pattern="*.0")
print(f"Loaded {len(spectra)} spectra")
```

## Example 3: Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from soilspec_pinn.preprocessing import SNVTransformer, MSCTransformer
import numpy as np

# Create preprocessing pipeline
pipeline = Pipeline([
    ('snv', SNVTransformer()),
    ('msc', MSCTransformer())
])

# Load spectra (example with synthetic data)
X = np.random.randn(100, 1701)  # 100 spectra

# Apply preprocessing
X_preprocessed = pipeline.fit_transform(X)

# Use in MBL workflow
from soilspec_pinn.models.mbl import MBLPredictor

y = np.random.randn(100)  # Target values
mbl = MBLPredictor()
mbl.fit(X_preprocessed, y)
```

## Example 4: Exploring MBL Neighbors

```python
from soilspec_pinn.models.mbl import MBLPredictor
import numpy as np

# Setup (using synthetic data)
X_cal = np.random.randn(200, 1701)
y_cal = np.random.randn(200)

mbl = MBLPredictor(k_neighbors=50, similarity_metric='mahalanobis')
mbl.fit(X_cal, y_cal)

# Get neighbor information for a query
query_spectrum = np.random.randn(1701)
neighbor_info = mbl.get_neighbor_info(query_spectrum)

print(f"Number of neighbors selected: {neighbor_info['n_neighbors']}")
print(f"Neighbor similarities: {neighbor_info['similarities'][:5]}")  # Top 5
print(f"Neighbor weights: {neighbor_info['weights'][:5]}")
print(f"Neighbor values: {neighbor_info['neighbor_values'][:5]}")
```

## Example 5: Physics-Informed Neural Network (PINN)

```python
import torch
from soilspec_pinn.models.pinn import SpectralPINN, BeerLambertLaw, PhysicsInformedLoss

# Initialize physics constraint
physics = BeerLambertLaw()

# Create PINN model
model = SpectralPINN(
    input_dim=1701,           # Number of spectral points
    hidden_dims=[512, 512],   # Hidden layer sizes
    output_dim=5,             # Number of soil properties to predict
    physics_constraint=physics,
    dropout=0.1
)

# Create loss function
loss_fn = PhysicsInformedLoss(
    data_weight=1.0,
    physics_weight=0.1,      # Weight of physics constraint
    data_loss_type='mse'
)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X_train = torch.randn(100, 1701)  # Training spectra
y_train = torch.randn(100, 5)     # Training targets

for epoch in range(10):
    optimizer.zero_grad()

    y_pred = model(X_train)

    # Compute physics residual (example)
    physics_residual = torch.randn(100, 1701)  # Replace with actual physics computation

    loss = loss_fn(y_pred, y_train, physics_residual)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## Example 6: Comparing Different Similarity Metrics

```python
from soilspec_pinn.models.mbl.similarity import SpectralSimilarity
import numpy as np

# Create sample spectra
query = np.random.randn(1701)
references = np.random.randn(100, 1701)

# Test different metrics
metrics = ['euclidean', 'mahalanobis', 'cosine', 'correlation', 'sid']

for metric in metrics:
    sim = SpectralSimilarity(metric=metric)
    sim.fit(references)  # Fit for Mahalanobis

    similarities = sim.compute(query, references, return_similarity=True)

    print(f"\n{metric.upper()}")
    print(f"  Min similarity: {similarities.min():.4f}")
    print(f"  Max similarity: {similarities.max():.4f}")
    print(f"  Mean similarity: {similarities.mean():.4f}")
```

## Example 7: Adaptive Neighbor Selection

```python
from soilspec_pinn.models.mbl import MBLPredictor
import numpy as np

# Compare fixed vs adaptive neighbor selection
X_cal = np.random.randn(200, 1701)
y_cal = np.random.randn(200)
X_test = np.random.randn(50, 1701)
y_test = np.random.randn(50)

# Fixed k
mbl_fixed = MBLPredictor(
    k_neighbors=50,
    selection_method='fixed',
    similarity_metric='mahalanobis'
)
mbl_fixed.fit(X_cal, y_cal)
pred_fixed = mbl_fixed.predict(X_test)

# Adaptive k
mbl_adaptive = MBLPredictor(
    k_neighbors=50,          # Initial k
    selection_method='adaptive',
    min_neighbors=10,
    max_neighbors=100,
    similarity_metric='mahalanobis'
)
mbl_adaptive.fit(X_cal, y_cal)
pred_adaptive = mbl_adaptive.predict(X_test)

# Compare performance
from sklearn.metrics import r2_score
print(f"Fixed k R²: {r2_score(y_test, pred_fixed):.3f}")
print(f"Adaptive k R²: {r2_score(y_test, pred_adaptive):.3f}")
```

## Testing with Real Data

When you have Bruker Alpha II DRIFTS data:

```python
from soilspec_pinn.io import BrukerReader
from soilspec_pinn.preprocessing import SNVTransformer
from soilspec_pinn.models.mbl import MBLPredictor
import numpy as np
import pandas as pd

# 1. Load spectra
reader = BrukerReader()
spectra = reader.read_directory("path/to/bruker/spectra/", pattern="*.0")

# Extract intensities and wavenumbers
X = np.array([s.intensities for s in spectra])
wavenumbers = spectra[0].wavenumbers

# 2. Load reference values (e.g., from CSV)
ref_data = pd.read_csv("reference_values.csv")
y = ref_data['SOC'].values  # Soil Organic Carbon

# 3. Preprocess
snv = SNVTransformer()
X_preprocessed = snv.fit_transform(X)

# 4. Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

# 5. Train MBL model
mbl = MBLPredictor(
    k_neighbors=50,
    similarity_metric='mahalanobis',
    weighting='gaussian',
    local_model='pls'
)
mbl.fit(X_train, y_train)

# 6. Predict and evaluate
predictions, uncertainties = mbl.predict(X_test, return_uncertainty=True)

from sklearn.metrics import r2_score, mean_squared_error
print(f"R² = {r2_score(y_test, predictions):.3f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, predictions)):.3f}")

# 7. Analyze results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Measured SOC')
plt.ylabel('Predicted SOC')
plt.title('MBL Predictions')

plt.subplot(1, 2, 2)
plt.scatter(predictions, uncertainties, alpha=0.5)
plt.xlabel('Predicted SOC')
plt.ylabel('Uncertainty')
plt.title('Prediction Uncertainty')

plt.tight_layout()
plt.savefig('mbl_results.png')
```

## Next Steps

1. **Test with your data**: Replace synthetic examples with real Bruker spectra
2. **Optimize parameters**: Tune k_neighbors, similarity metrics, weighting schemes
3. **Compare models**: Test MBL vs PLS vs PINN
4. **Extend preprocessing**: Add derivatives, smoothing, resampling
5. **Build workflows**: Create end-to-end analysis pipelines

## Documentation

- See `README.md` for comprehensive package overview
- See `soilspec_package_plan.md` for detailed architecture and roadmap
- See `PACKAGE_STATUS.md` for implementation status

## Getting Help

For issues or questions:
1. Check the implementation in `soilspec_pinn/`
2. Review test examples in `tests/`
3. Consult the plan in `soilspec_package_plan.md`

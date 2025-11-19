# SoilSpec-PINN: Physics-Informed Neural Networks for Soil Spectroscopy

A comprehensive Python package for analyzing soil mid-infrared spectra using physics-informed neural networks (PINNs), message passing neural networks (MPNNs), and traditional machine learning methods.

## Features

### Data I/O
- **Bruker Alpha II DRIFTS Support**: Read OPUS binary files with metadata extraction
- **OSSL Integration**: Compatible with Open Soil Spectral Library formats and pre-trained models
- **Flexible Format Support**: CSV, HDF5, and custom spectral formats

### Preprocessing
- **Baseline Corrections**: SNV (Standard Normal Variate), MSC (Multiplicative Scatter Correction)
- **Derivatives**: Savitzky-Golay smoothing and derivatives (1st, 2nd order)
- **Smoothing**: Wavelet denoising, Savitzky-Golay filters
- **Transformations**: Absorbance/reflectance conversions, spectral resampling
- **Scikit-learn Compatible**: Full pipeline integration with sklearn

### Models

#### Physics-Informed Neural Networks (PINNs)
- **Physical Constraints**: Beer-Lambert law, Kubelka-Munk radiative transfer
- **Custom Loss Functions**: Combined data and physics losses
- **Unsupervised Learning**: Extract spectral information without labels

#### Message Passing Neural Networks (MPNNs)
- **Chemprop-IR Architecture**: Directed MPNN for composition-to-spectra prediction
- **Spectral Information Divergence**: SID loss function for robust training
- **Ensemble Methods**: 10-model ensembles for uncertainty quantification

#### Hierarchical Networks
- **Multi-Resolution U-nets**: Physics-guided hierarchical architectures
- **Transfer Learning**: Progressive knowledge transfer across resolutions

#### Traditional ML
- **PLS Regression**: Partial Least Squares for calibration
- **Cubist Models**: Integration with OSSL pre-trained Cubist models
- **Ensemble Methods**: Random forests, gradient boosting

### Training & Prediction
- **Flexible Training Pipeline**: Support for all model types
- **Uncertainty Quantification**: Conformal prediction, ensemble uncertainty, dropout uncertainty
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **GPU Acceleration**: PyTorch Lightning backend

## Installation

### Basic Installation

```bash
pip install soilspec-pinn
```

### Development Installation

```bash
git clone https://github.com/yourusername/soilspec-pinn.git
cd soilspec-pinn
pip install -e ".[dev]"
```

### Full Installation (with all optional dependencies)

```bash
pip install "soilspec-pinn[all]"
```

## Quick Start

### Reading Bruker Spectra

```python
from soilspec_pinn.io import BrukerReader

# Read a single OPUS file
reader = BrukerReader()
spectrum = reader.read_opus_file("path/to/spectrum.0")

# Read a directory of spectra
spectra = reader.read_directory("path/to/spectra_folder")
```

### Preprocessing Pipeline

```python
from soilspec_pinn.preprocessing import (
    SNVTransformer,
    SavitzkyGolayDerivative,
    SpectralResample,
)
from sklearn.pipeline import Pipeline

# Create preprocessing pipeline
pipeline = Pipeline([
    ('snv', SNVTransformer()),
    ('derivative', SavitzkyGolayDerivative(window_length=11, polyorder=2, deriv=1)),
    ('resample', SpectralResample(wavenumbers=range(600, 4001, 2))),
])

# Apply preprocessing
preprocessed_spectra = pipeline.fit_transform(spectra)
```

### Training a PINN Model

```python
from soilspec_pinn.models.pinn import SpectralPINN
from soilspec_pinn.models.pinn.physics import BeerLambertLaw
from soilspec_pinn.training import PINNTrainer

# Define physics constraint
physics = BeerLambertLaw()

# Create model
model = SpectralPINN(
    input_dim=1801,
    hidden_dims=[512, 512, 512],
    output_dim=10,  # number of soil properties
    physics_constraint=physics
)

# Train model
trainer = PINNTrainer(
    model=model,
    physics_constraint=physics,
    data_weight=1.0,
    physics_weight=0.1,
    lr=1e-3
)

trainer.fit(train_loader, val_loader, physics_params, epochs=100)
```

### Using OSSL Pre-trained Models

```python
from soilspec_pinn.integration import OSSLModelLoader

# Load pre-trained OSSL Cubist model
loader = OSSLModelLoader()
model = loader.load_model("ossl.mir.cubist")

# Make predictions
predictions = model.predict(spectra)
predictions_with_ci = model.predict_with_uncertainty(spectra)
```

### Making Predictions with Uncertainty

```python
from soilspec_pinn.prediction import SpectralPredictor

predictor = SpectralPredictor(model, preprocessing_pipeline=pipeline)

# Point predictions
predictions = predictor.predict(raw_spectra)

# Predictions with uncertainty
mean, std = predictor.predict_with_uncertainty(raw_spectra, method='ensemble')
```

## Documentation

Full documentation is available at [https://soilspec-pinn.readthedocs.io](https://soilspec-pinn.readthedocs.io)

### Tutorials

- [Getting Started](docs/tutorials/01_getting_started.ipynb)
- [Preprocessing Spectra](docs/tutorials/02_preprocessing.ipynb)
- [Training PINN Models](docs/tutorials/03_pinn_training.ipynb)
- [Using MPNN for Molecular Spectra](docs/tutorials/04_mpnn.ipynb)
- [OSSL Integration](docs/tutorials/05_ossl_integration.ipynb)
- [Uncertainty Quantification](docs/tutorials/06_uncertainty.ipynb)

## Project Structure

```
soilspec_pinn/
├── io/                    # Data input/output
│   ├── bruker.py         # Bruker OPUS file readers
│   ├── ossl.py           # OSSL format handlers
│   └── converters.py     # Format conversions
├── preprocessing/         # Spectral preprocessing
│   ├── baseline.py       # SNV, MSC, detrending
│   ├── derivatives.py    # Savitzky-Golay derivatives
│   ├── smoothing.py      # Wavelet, SG smoothing
│   └── pipeline.py       # Sklearn pipelines
├── models/               # Machine learning models
│   ├── pinn/            # Physics-informed neural networks
│   ├── mpnn/            # Message passing neural networks
│   ├── traditional/     # PLS, Cubist, ensembles
│   └── hierarchical/    # Hierarchical U-nets
├── training/            # Training infrastructure
├── prediction/          # Prediction and uncertainty
├── integration/         # OSSL and external tools
├── utils/              # Utilities
└── datasets/           # Dataset loaders
```

## Research Background

This package implements methods from several key research papers:

1. **Physics-Informed Neural Networks for Spectroscopy**
   - Unsupervised spectra information extraction using PINNs
   - Handling non-linearities in spectroscopic data

2. **Chemprop-IR** (Zubatyuk et al., JCIM)
   - Message passing neural networks for IR spectra prediction
   - Spectral Information Divergence loss function
   - Ensemble methods for uncertainty quantification

3. **Physics-Guided Hierarchical Neural Networks**
   - Hierarchical U-net architectures with physics constraints
   - Transfer learning across resolution levels

## Citation

If you use this package in your research, please cite:

```bibtex
@software{soilspec_pinn,
  title = {SoilSpec-PINN: Physics-Informed Neural Networks for Soil Spectroscopy},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/soilspec-pinn}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Open Soil Spectral Library (OSSL) for pre-trained models and datasets
- R prospectr package for preprocessing reference implementations
- SoilSpecTfm package for scikit-learn compatible transformers
- Chemprop-IR authors for MPNN architecture insights

## Contact

For questions and support, please open an issue on [GitHub Issues](https://github.com/yourusername/soilspec-pinn/issues).

# SoilSpec-PINN: Physics-Informed Neural Networks for Soil Spectroscopy

## Executive Summary

This document outlines a comprehensive plan for developing a Python package that applies physics-informed neural networks (PINNs), memory-based learning (MBL), and advanced machine learning techniques to soil mid-infrared spectroscopy analysis. The package integrates proven methodologies from doctoral research at TU Bergakademie Freiberg with cutting-edge deep learning approaches, supporting Bruker Alpha II DRIFTS measurements, OSSL integration, and state-of-the-art preprocessing and modeling capabilities.

**Target Application**: Analysis of soil mid-IR spectra from DRIFTS (Diffuse Reflectance Infrared Fourier Transform Spectroscopy)
**Instruments**: Bruker Alpha II, Elementar soliTOCcube, Spectrolyzer UV-Vis
**Spectral Range**: 600-4000 cm⁻¹ (mid-infrared)
**Core Methodologies**: PINNs, Memory-Based Learning (MBL), MPNNs, Traditional ML

---

## 1. Literature Review & Theoretical Foundation

### 1.1 Physics-Informed Neural Networks for Spectroscopy

**Source**: Nature Scientific Reports (s41598-025-25573-5.pdf)

**Key Concepts**:
- Unsupervised spectra information extraction using PINNs
- Handling non-linearities in spectroscopic data
- Multi-agent problem solving for complex spectral systems
- Embedding physical constraints in neural network training

**Relevance**: Provides theoretical foundation for incorporating Beer-Lambert law, radiative transfer equations, and other physical models into neural network training for soil spectroscopy.

### 1.2 Chemprop-IR: Message Passing Neural Networks for IR Spectra

**Source**: Journal of Chemical Information and Modeling (Chemprop_IR_JCIM.pdf)

**Key Architecture Details**:
- **Model Type**: Directed Message Passing Neural Network (D-MPNN)
- **Training Data**: 56,955 experimental IR spectra from NIST, PNNL, AIST, Coblentz Society
- **Pretraining**: 85,232 computed spectra using GFN2-xTB quantum chemistry
- **Spectral Representation**: 1801 normalized absorbances (2 cm⁻¹ intervals, 400-4000 cm⁻¹)
- **Loss Function**: Spectral Information Divergence (SID)
- **Network Architecture**:
  - MPNN: depth 6, hidden size 2200
  - Feed-forward: depth 3, hidden size 2200
  - Total parameters: 23,957,601
  - Dropout: 0.05
  - Optimizer: Adam
- **Ensemble Approach**: 10 sub-models for robust predictions

**Relevance**: Demonstrates how graph neural networks can learn molecular-to-spectra relationships. Can be adapted for soil composition-to-spectra prediction.

### 1.3 Physics-Guided Hierarchical Neural Networks

**Source**: ACS Photonics (physics-guided-hierarchical-neural-networks.pdf)

**Key Concepts**:
- Hierarchical U-net architecture with physics constraints
- Maxwell's equations embedded in loss function
- Transfer learning between different resolution networks
- **Loss Components**:
  - L_φ: Mean squared error on field components
  - L_rz: Hybrid loss including physics-based layer outputs
  - L_ph: Physics consistency loss (residual of physical equations)

**Relevance**: Provides methodology for creating hierarchical models with physics constraints, applicable to multi-scale spectral analysis and transfer learning from low to high-resolution spectra.

---

## 2. Existing Tools Analysis

### 2.1 SoilSpecTfm

**Repository**: https://github.com/franckalbinet/soilspectfm

**Features**:
- Scikit-learn compatible transformers
- **Baseline Corrections**: SNV (Standard Normal Variate), MSC (Multiplicative Scatter Correction)
- **Derivatives**: TakeDerivative with Savitzky-Golay smoothing
- **Smoothing**: WaveletDenoise, SavGolSmooth
- **Transformations**: ToAbsorbance, Resample, Trim

**Assessment**: Well-designed preprocessing library but lacks ML modeling capabilities and physics-informed approaches.

### 2.2 OSSL Models

**Repository**: https://github.com/soilspectroscopy/ossl-models

**Specifications**:
- **Algorithm**: Cubist (decision tree + linear regression)
- **Available Models**: 5 pre-trained models for MIR, NIR, VisNIR
- **Input**: First 120 principal components of compressed spectra
- **Spectral Ranges**:
  - MIR: 600-4000 cm⁻¹ (relevant for our application)
  - NIR: 1350-2550 nm
  - VisNIR: 400-2500 nm
- **Uncertainty**: Conformal prediction intervals

**Assessment**: Provides baseline models using traditional ML. Our package should integrate with these models while offering advanced PINN alternatives.

### 2.3 OSSL Training Pipeline

**Repository**: https://github.com/soilspectroscopy/predictive-soilspec-training

**Structure**: R-based Quarto educational materials with modules for:
- Spectral processing
- Chemometrics
- Model training and evaluation

**Assessment**: Well-structured training pipeline but implemented in R. Our Python package should provide similar functionality with added PINN capabilities.

### 2.4 R Prospectr Package

**Capabilities** (from documentation):
- Extensive preprocessing functions
- Derivative computations
- Smoothing and filtering
- Baseline corrections
- Spectral transformations

**Assessment**: Industry-standard preprocessing library. Our Python package should match or exceed its capabilities.

### 2.5 TUBAFsoilFunctions (Student Research Package)

**Repository**: https://github.com/seanadamhdh/TUBAFsoilFunctions

**Purpose**: R package developed during doctoral research at TU Bergakademie Freiberg's soil working group

**Key Features**:
- **Bruker Alpha DRIFTS Loading**: Specialized loaders for Bruker Alpha measurements
- **Elementar soliTOCcube Integration**: Carbon/nitrogen analysis data handling
- **Spectrolyzer UV-Vis Processing**: UV-Vis spectra processing utilities
- **Chemometric Modeling**: Caret framework implementation for MIR and UV-Vis calibration
- **Respiration Analysis**: SEMACHv3 chamber and PriEco Pri8800 incubator data processing
- **Soil Profile Utilities**: Data standardization and sample labeling tools

**Dependencies**: caret, prospectr, resemble, simplerspec

**Assessment**: Provides field-tested workflows specifically for TUBAF soil lab instruments and procedures. Critical to integrate these proven methodologies, especially Bruker-specific handling.

### 2.6 saxSSL Code (Memory-Based Learning Application)

**Repository**: https://github.com/seanadamhdh/saxSSL_code

**Purpose**: Interactive application for soil property prediction using Memory-Based Learning (MBL)

**Key Methodology - Memory-Based Learning (MBL)**:
- **Concept**: Local modeling approach where predictions are made using only the most similar spectra from the calibration set
- **Library**: Based on `resemble` R package for spectral chemometrics
- **Advantages**:
  - No global model training required
  - Adapts to local spectral space
  - Handles non-linear relationships effectively
  - Natural uncertainty quantification through local variance
  - Computationally efficient for prediction

**Features**:
- Interactive web interface for spectral analysis
- Real-time MBL prediction
- Integration with simplerspec workflow
- Spectral similarity computation
- Local model selection and weighting

**Assessment**: MBL is a powerful complementary approach to global models (PLS, PINN). Essential for scenarios with:
- Limited training data
- Non-stationary spectral characteristics
- Need for interpretable predictions
- Transfer learning between instruments/labs

---

## 3. Proposed Package Architecture

### 3.1 Package Name: `soilspec_pinn`

### 3.2 Module Structure

```
soilspec_pinn/
├── __init__.py
├── io/
│   ├── __init__.py
│   ├── bruker.py          # Bruker OPUS & Alpha II file readers
│   ├── elementar.py       # Elementar soliTOCcube readers
│   ├── spectrolyzer.py    # Spectrolyzer UV-Vis readers
│   ├── ossl.py            # OSSL format handlers
│   └── converters.py      # Format conversions
├── preprocessing/
│   ├── __init__.py
│   ├── baseline.py        # SNV, MSC, detrending
│   ├── derivatives.py     # Savitzky-Golay derivatives
│   ├── smoothing.py       # Wavelet, SG smoothing
│   ├── transforms.py      # Absorbance, reflectance conversions
│   ├── resample.py        # Spectral resampling
│   ├── similarity.py      # Spectral similarity metrics (for MBL)
│   └── pipeline.py        # Scikit-learn compatible pipelines
├── models/
│   ├── __init__.py
│   ├── pinn/
│   │   ├── __init__.py
│   │   ├── physics.py     # Physical laws (Beer-Lambert, etc.)
│   │   ├── losses.py      # Physics-informed loss functions
│   │   ├── networks.py    # PINN architectures
│   │   └── constraints.py # Physical constraints
│   ├── mpnn/
│   │   ├── __init__.py
│   │   ├── chemprop.py    # Chemprop-IR style MPNN
│   │   ├── graph.py       # Molecular graph construction
│   │   └── sid_loss.py    # Spectral Information Divergence
│   ├── mbl/
│   │   ├── __init__.py
│   │   ├── similarity.py  # Distance/similarity metrics
│   │   ├── selection.py   # Neighbor selection strategies
│   │   ├── weighting.py   # Local model weighting
│   │   └── predictor.py   # MBL prediction engine
│   ├── traditional/
│   │   ├── __init__.py
│   │   ├── pls.py         # Partial Least Squares
│   │   ├── cubist.py      # Cubist wrapper (OSSL compatibility)
│   │   └── ensemble.py    # Ensemble methods
│   └── hierarchical/
│       ├── __init__.py
│       ├── unet.py        # Hierarchical U-net
│       └── transfer.py    # Transfer learning utilities
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Generic training loops
│   ├── pinn_trainer.py    # PINN-specific training
│   ├── callbacks.py       # Training callbacks
│   └── metrics.py         # Evaluation metrics
├── prediction/
│   ├── __init__.py
│   ├── predictor.py       # Prediction interface
│   └── uncertainty.py     # Conformal prediction, ensembles
├── integration/
│   ├── __init__.py
│   ├── ossl_models.py     # OSSL model integration
│   └── model_zoo.py       # Pre-trained model registry
├── utils/
│   ├── __init__.py
│   ├── spectral.py        # Spectral processing utilities
│   ├── validation.py      # Cross-validation tools
│   └── visualization.py   # Plotting utilities
└── datasets/
    ├── __init__.py
    ├── loaders.py         # Dataset loaders
    └── augmentation.py    # Spectral augmentation
```

---

## 4. Technical Design Specifications

### 4.1 Data Ingestion Module (`io/`)

**Bruker Alpha II Support**:
- Read OPUS binary files (.0, .1, .2, etc.)
- Extract metadata (instrument parameters, measurement conditions)
- Handle multiple measurement blocks (absorbance, reflectance, interferograms)
- Support batch processing of measurement series

**OSSL Integration**:
- Read OSSL CSV format (compressed spectra)
- Load pre-trained OSSL models
- Convert between OSSL and native formats

**Key Classes**:
```python
class BrukerReader:
    def read_opus_file(self, filepath) -> Spectrum
    def read_directory(self, dirpath) -> List[Spectrum]
    def extract_metadata(self) -> dict

class OSSLReader:
    def load_spectra(self, filepath) -> DataFrame
    def load_model(self, model_name) -> OSSLModel
```

### 4.2 Preprocessing Module (`preprocessing/`)

**Scikit-learn Compatible Transformers**:

```python
class SNVTransformer(TransformerMixin):
    """Standard Normal Variate correction"""

class MSCTransformer(TransformerMixin):
    """Multiplicative Scatter Correction"""

class SavitzkyGolayDerivative(TransformerMixin):
    """SG smoothing with derivative calculation"""
    def __init__(self, window_length, polyorder, deriv=0)

class WaveletDenoise(TransformerMixin):
    """Wavelet-based denoising"""
    def __init__(self, wavelet='sym8', level=3)

class SpectralResample(TransformerMixin):
    """Resample spectra to target wavenumbers"""
    def __init__(self, target_wavenumbers)
```

**Preprocessing Pipeline**:
```python
from sklearn.pipeline import Pipeline

preprocessing_pipeline = Pipeline([
    ('trim', TrimTransformer(min_wn=600, max_wn=4000)),
    ('to_absorbance', ToAbsorbance()),
    ('snv', SNVTransformer()),
    ('savgol', SavitzkyGolayDerivative(window_length=11, polyorder=2, deriv=1)),
    ('resample', SpectralResample(target_wn=np.arange(600, 4001, 2)))
])
```

### 4.3 Physics-Informed Neural Network Module (`models/pinn/`)

**Physical Laws Implementation**:

```python
class BeerLambertLaw:
    """A = ε * c * l"""
    def compute_residual(self, absorbance, concentration, path_length, epsilon):
        return absorbance - epsilon * concentration * path_length

class RadiativeTransferEquation:
    """Kubelka-Munk theory for diffuse reflectance"""
    def compute_reflectance(self, k, s):
        # k: absorption coefficient, s: scattering coefficient
        return (1 + k/s) - sqrt((k/s)**2 + 2*k/s)
```

**PINN Architecture**:

```python
class SpectralPINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, physics_constraint):
        super().__init__()
        self.network = build_feedforward(input_dim, hidden_dims, output_dim)
        self.physics_constraint = physics_constraint

    def forward(self, x):
        prediction = self.network(x)
        return prediction

    def compute_loss(self, x, y_true, physics_params):
        y_pred = self.forward(x)

        # Data loss
        data_loss = F.mse_loss(y_pred, y_true)

        # Physics loss
        physics_residual = self.physics_constraint.compute_residual(
            y_pred, physics_params
        )
        physics_loss = torch.mean(physics_residual ** 2)

        # Combined loss
        total_loss = data_loss + self.physics_weight * physics_loss
        return total_loss, data_loss, physics_loss
```

**Loss Functions**:

```python
class PhysicsInformedLoss:
    def __init__(self, data_weight=1.0, physics_weight=1.0):
        self.data_weight = data_weight
        self.physics_weight = physics_weight

    def __call__(self, y_pred, y_true, physics_residual):
        L_data = F.mse_loss(y_pred, y_true)
        L_physics = torch.mean(physics_residual ** 2)
        return self.data_weight * L_data + self.physics_weight * L_physics
```

### 4.4 Message Passing Neural Network Module (`models/mpnn/`)

**Chemprop-IR Inspired Architecture**:

```python
class SoilMPNN(nn.Module):
    """
    Directed MPNN for soil composition to spectra prediction
    Adapted from Chemprop-IR architecture
    """
    def __init__(
        self,
        mpnn_depth=6,
        mpnn_hidden_size=2200,
        ffnn_depth=3,
        ffnn_hidden_size=2200,
        dropout=0.05,
        output_size=1801  # spectral points
    ):
        super().__init__()
        self.mpnn = DirectedMPNN(depth=mpnn_depth, hidden_size=mpnn_hidden_size)
        self.ffnn = FeedForwardNetwork(
            input_size=mpnn_hidden_size,
            hidden_size=ffnn_hidden_size,
            output_size=output_size,
            depth=ffnn_depth,
            dropout=dropout
        )

    def forward(self, molecular_graph):
        # Message passing on molecular/composition graph
        node_features = self.mpnn(molecular_graph)

        # Aggregate to graph-level representation
        graph_features = self.readout(node_features)

        # Predict spectrum
        spectrum = self.ffnn(graph_features)
        return spectrum

class SpectralInformationDivergence(nn.Module):
    """SID loss as used in Chemprop-IR"""
    def forward(self, pred_spectrum, true_spectrum):
        # Normalize to probability distributions
        p = F.softmax(true_spectrum, dim=-1) + 1e-10
        q = F.softmax(pred_spectrum, dim=-1) + 1e-10

        # SID = KL(p||q) + KL(q||p)
        kl_pq = torch.sum(p * torch.log(p / q), dim=-1)
        kl_qp = torch.sum(q * torch.log(q / p), dim=-1)
        sid = kl_pq + kl_qp
        return torch.mean(sid)
```

### 4.5 Hierarchical Network Module (`models/hierarchical/`)

**Multi-Resolution U-net with Transfer Learning**:

```python
class HierarchicalSpectralUNet(nn.Module):
    """
    Physics-guided hierarchical network for spectral analysis
    Inspired by Maxwell's equation solver architecture
    """
    def __init__(self, resolution_levels=[64, 128, 256, 512, 1801]):
        super().__init__()
        self.levels = resolution_levels
        self.encoders = nn.ModuleList([
            UNetEncoder(in_ch, out_ch)
            for in_ch, out_ch in zip(resolution_levels[:-1], resolution_levels[1:])
        ])
        self.decoders = nn.ModuleList([
            UNetDecoder(in_ch, out_ch)
            for in_ch, out_ch in zip(reversed(resolution_levels[1:]),
                                     reversed(resolution_levels[:-1]))
        ])

    def forward(self, low_res_spectrum):
        # Progressive upsampling with skip connections
        x = low_res_spectrum
        skip_connections = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        return x

    def compute_physics_loss(self, spectrum):
        # Implement physical consistency checks
        # e.g., baseline constraints, peak shape constraints
        pass
```

### 4.6 Memory-Based Learning (MBL) Module (`models/mbl/`)

**Concept and Implementation**:

Memory-Based Learning (MBL), also known as k-Nearest Neighbors regression or local modeling, makes predictions using only the most similar spectra from the calibration set. This approach is particularly effective for:
- Non-linear spectral-property relationships
- Transfer learning between instruments
- Handling spectral drift
- Providing interpretable predictions

**Core MBL Components**:

```python
class MBLPredictor:
    """
    Memory-Based Learning predictor for spectral data.

    Based on the resemble R package methodology, adapted from saxSSL application.
    Implements local modeling where predictions are made using k-nearest neighbors.
    """

    def __init__(
        self,
        k_neighbors=50,
        similarity_metric='mahalanobis',
        weighting='gaussian',
        local_model='pls',
        n_components=10
    ):
        """
        Initialize MBL predictor.

        Args:
            k_neighbors: Number of nearest neighbors to use
            similarity_metric: 'euclidean', 'mahalanobis', 'cosine', 'correlation'
            weighting: 'uniform', 'distance', 'gaussian'
            local_model: 'pls', 'ridge', 'wls' (weighted least squares)
            n_components: Number of PLS components for local model
        """
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.weighting = weighting
        self.local_model = local_model
        self.n_components = n_components

        # Calibration data storage (memory)
        self.X_cal = None
        self.y_cal = None

    def fit(self, X_cal, y_cal):
        """
        Store calibration (memory) set.

        Args:
            X_cal: Calibration spectra (n_samples, n_features)
            y_cal: Calibration property values (n_samples, n_properties)
        """
        self.X_cal = X_cal
        self.y_cal = y_cal
        return self

    def predict(self, X_pred):
        """
        Predict using memory-based learning.

        For each prediction spectrum:
        1. Compute similarity to all calibration spectra
        2. Select k nearest neighbors
        3. Fit local model on neighbors
        4. Make prediction

        Args:
            X_pred: Prediction spectra (n_samples, n_features)

        Returns:
            Predictions and uncertainty estimates
        """
        predictions = []
        uncertainties = []

        for i in range(X_pred.shape[0]):
            spectrum = X_pred[i:i+1, :]

            # Compute similarities
            similarities = self._compute_similarity(spectrum, self.X_cal)

            # Select k nearest neighbors
            neighbor_indices = np.argsort(similarities)[::-1][:self.k_neighbors]

            # Extract neighbor data
            X_neighbors = self.X_cal[neighbor_indices]
            y_neighbors = self.y_cal[neighbor_indices]

            # Compute weights
            neighbor_similarities = similarities[neighbor_indices]
            weights = self._compute_weights(neighbor_similarities)

            # Fit local model
            local_model = self._fit_local_model(
                X_neighbors, y_neighbors, weights
            )

            # Make prediction
            pred = local_model.predict(spectrum)
            predictions.append(pred)

            # Estimate uncertainty from local variance
            uncertainty = np.std(y_neighbors)
            uncertainties.append(uncertainty)

        return np.array(predictions), np.array(uncertainties)

    def _compute_similarity(self, spectrum, calibration_set):
        """Compute spectral similarity/distance."""
        if self.similarity_metric == 'euclidean':
            distances = np.linalg.norm(calibration_set - spectrum, axis=1)
            return 1.0 / (1.0 + distances)  # Convert to similarity

        elif self.similarity_metric == 'mahalanobis':
            # Compute Mahalanobis distance with covariance
            cov = np.cov(calibration_set.T)
            inv_cov = np.linalg.pinv(cov)
            diff = calibration_set - spectrum
            distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            return 1.0 / (1.0 + distances)

        elif self.similarity_metric == 'cosine':
            # Cosine similarity
            norm_spectrum = spectrum / np.linalg.norm(spectrum)
            norm_cal = calibration_set / np.linalg.norm(calibration_set, axis=1, keepdims=True)
            return np.dot(norm_cal, norm_spectrum.T).flatten()

        elif self.similarity_metric == 'correlation':
            # Pearson correlation
            return np.array([
                np.corrcoef(spectrum.flatten(), cal_spec)[0, 1]
                for cal_spec in calibration_set
            ])

    def _compute_weights(self, similarities):
        """Compute neighbor weights based on similarities."""
        if self.weighting == 'uniform':
            return np.ones_like(similarities) / len(similarities)

        elif self.weighting == 'distance':
            # Inverse distance weighting
            weights = similarities / np.sum(similarities)
            return weights

        elif self.weighting == 'gaussian':
            # Gaussian kernel weighting
            sigma = np.std(similarities)
            weights = np.exp(-(1 - similarities)**2 / (2 * sigma**2))
            return weights / np.sum(weights)

    def _fit_local_model(self, X, y, weights):
        """Fit local regression model on neighbors."""
        if self.local_model == 'pls':
            from sklearn.cross_decomposition import PLSRegression
            model = PLSRegression(n_components=self.n_components)
            # Apply weights through sample duplication or weighted loss
            model.fit(X, y)
            return model

        elif self.local_model == 'ridge':
            from sklearn.linear_model import Ridge
            model = Ridge()
            model.fit(X, y, sample_weight=weights)
            return model

        elif self.local_model == 'wls':
            # Weighted least squares
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            # Apply weights by scaling samples
            X_weighted = X * np.sqrt(weights[:, np.newaxis])
            y_weighted = y * np.sqrt(weights)
            model.fit(X_weighted, y_weighted)
            return model


class SpectralSimilarity:
    """
    Spectral similarity metrics for MBL and sample selection.
    """

    @staticmethod
    def euclidean_distance(spec1, spec2):
        """Euclidean distance between spectra."""
        return np.linalg.norm(spec1 - spec2)

    @staticmethod
    def mahalanobis_distance(spec1, spec2, covariance):
        """Mahalanobis distance accounting for spectral covariance."""
        diff = spec1 - spec2
        inv_cov = np.linalg.pinv(covariance)
        return np.sqrt(diff @ inv_cov @ diff.T)

    @staticmethod
    def spectral_angle(spec1, spec2):
        """Spectral angle mapper (SAM) distance."""
        dot_product = np.dot(spec1, spec2)
        norms = np.linalg.norm(spec1) * np.linalg.norm(spec2)
        cos_angle = dot_product / norms
        return np.arccos(np.clip(cos_angle, -1, 1))

    @staticmethod
    def spectral_information_divergence(spec1, spec2):
        """SID distance between spectra."""
        # Normalize to probability distributions
        p = spec1 / np.sum(spec1) + 1e-10
        q = spec2 / np.sum(spec2) + 1e-10

        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))

        return kl_pq + kl_qp
```

### 4.7 Traditional ML Module (`models/traditional/`)

**OSSL Cubist Integration**:

```python
class CubistWrapper:
    """Wrapper for OSSL Cubist models"""
    def __init__(self, model_path):
        self.model = self.load_ossl_model(model_path)

    def predict(self, spectra):
        # Convert to first 120 PCs as OSSL expects
        pcs = self.compute_pcs(spectra, n_components=120)
        return self.model.predict(pcs)

    def predict_with_uncertainty(self, spectra):
        # Conformal prediction intervals
        predictions = self.predict(spectra)
        intervals = self.compute_conformal_intervals(spectra)
        return predictions, intervals
```

**Ensemble Methods**:

```python
class SpectralEnsemble:
    """
    Ensemble of 10 models as in Chemprop-IR
    """
    def __init__(self, model_class, n_models=10):
        self.models = [model_class() for _ in range(n_models)]

    def fit(self, X, y):
        for model in self.models:
            # Train with different random seeds
            model.fit(X, y)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)
```

### 4.8 Training Module (`training/`)

**Generic Trainer with PINN Support**:

```python
class PINNTrainer:
    def __init__(
        self,
        model,
        physics_constraint,
        data_weight=1.0,
        physics_weight=1.0,
        optimizer='adam',
        lr=1e-3
    ):
        self.model = model
        self.physics_constraint = physics_constraint
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.optimizer = self._build_optimizer(optimizer, lr)

    def train_epoch(self, train_loader, physics_params_loader):
        self.model.train()
        total_loss = 0

        for (X, y), physics_params in zip(train_loader, physics_params_loader):
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(X)

            # Compute losses
            data_loss = F.mse_loss(y_pred, y)
            physics_residual = self.physics_constraint.compute_residual(
                y_pred, physics_params
            )
            physics_loss = torch.mean(physics_residual ** 2)

            # Combined loss
            loss = self.data_weight * data_loss + self.physics_weight * physics_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def fit(self, train_loader, val_loader, physics_params, epochs=100):
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, physics_params)
            val_loss = self.validate(val_loader, physics_params)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint()

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}")
```

### 4.9 Prediction & Uncertainty Module (`prediction/`)

```python
class SpectralPredictor:
    def __init__(self, model, preprocessing_pipeline=None):
        self.model = model
        self.preprocessing = preprocessing_pipeline

    def predict(self, raw_spectra):
        if self.preprocessing:
            spectra = self.preprocessing.transform(raw_spectra)
        else:
            spectra = raw_spectra

        return self.model.predict(spectra)

    def predict_with_uncertainty(self, raw_spectra, method='ensemble'):
        if method == 'ensemble':
            return self._ensemble_uncertainty(raw_spectra)
        elif method == 'conformal':
            return self._conformal_uncertainty(raw_spectra)
        elif method == 'dropout':
            return self._dropout_uncertainty(raw_spectra)

    def _ensemble_uncertainty(self, spectra):
        # Use ensemble of models
        predictions = [model.predict(spectra) for model in self.model.models]
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

**Sprint 1.1: Package Setup**
- Initialize Python package structure
- Setup development environment (poetry/pip, pytest, black, mypy)
- Configure CI/CD (GitHub Actions)
- Write initial documentation structure

**Sprint 1.2: Data I/O Module**
- Implement Bruker OPUS file reader
  - Binary format parsing
  - Metadata extraction
  - Support for multiple data blocks
- Implement OSSL format handlers
- Write unit tests for file readers

**Sprint 1.3: Preprocessing Module**
- Implement baseline corrections (SNV, MSC)
- Implement derivatives (Savitzky-Golay)
- Implement smoothing (wavelet, SG)
- Implement transformations (absorbance, resample)
- Create scikit-learn compatible pipeline
- Validate against R prospectr outputs

### Phase 2: Traditional ML and MBL Models (Weeks 4-6)

**Sprint 2.1: Memory-Based Learning (MBL)**
- Implement spectral similarity metrics (Euclidean, Mahalanobis, cosine, correlation)
- Create neighbor selection strategies
- Implement weighting schemes (uniform, distance, Gaussian)
- Build MBL predictor with local PLS/Ridge regression
- Validate against saxSSL application outputs

**Sprint 2.2: Traditional Models**
- Implement PLS regression
- Create Cubist wrapper for OSSL models
- Implement ensemble methods
- Add cross-validation utilities

**Sprint 2.3: OSSL Integration**
- Load and test OSSL pre-trained models
- Implement PCA compression (120 components)
- Add conformal prediction intervals
- Validate predictions against OSSL benchmarks

### Phase 3: Physics-Informed Models (Weeks 6-9)

**Sprint 3.1: Physics Module**
- Implement Beer-Lambert law
- Implement Kubelka-Munk theory
- Create physics constraint interface
- Write physics residual calculators

**Sprint 3.2: PINN Architecture**
- Build basic PINN network
- Implement physics-informed loss functions
- Create PINN trainer
- Test on synthetic data with known physics

**Sprint 3.3: Advanced PINN Features**
- Implement multi-agent PINN (from Nature paper)
- Add support for non-linearities
- Integrate unsupervised learning approaches
- Validate on real soil spectra

### Phase 4: Advanced Neural Networks (Weeks 10-14)

**Sprint 4.1: MPNN Foundation**
- Implement directed message passing layers
- Create molecular/composition graph builder
- Build graph readout functions

**Sprint 4.2: Chemprop-IR Style Model**
- Implement full Chemprop-IR architecture
  - MPNN depth 6, hidden 2200
  - FFNN depth 3, hidden 2200
- Implement SID loss function
- Create ensemble training (10 models)

**Sprint 4.3: Hierarchical Networks**
- Implement U-net architecture
- Add multi-resolution support
- Implement transfer learning utilities
- Create hierarchical training pipeline

### Phase 5: Training Infrastructure (Weeks 15-16)

**Sprint 5.1: Training Pipeline**
- Create generic trainer interface
- Implement training callbacks (early stopping, LR scheduling)
- Add metric tracking (RMSE, R², RPD, RPIQ)
- Integrate with MLflow/Weights & Biases

**Sprint 5.2: Data Augmentation**
- Implement spectral augmentation techniques
- Add noise injection
- Create synthetic spectra generator

### Phase 6: Prediction & Deployment (Weeks 17-18)

**Sprint 6.1: Prediction Module**
- Create unified prediction interface
- Implement uncertainty quantification
- Add batch prediction support

**Sprint 6.2: Model Zoo**
- Create pre-trained model registry
- Add model versioning
- Implement model download utilities

### Phase 7: Documentation & Testing (Weeks 19-20)

**Sprint 7.1: Documentation**
- Complete API documentation
- Write tutorials (Jupyter notebooks)
- Create example workflows
- Write deployment guide

**Sprint 7.2: Testing & Validation**
- Comprehensive unit tests (>90% coverage)
- Integration tests
- Validate on public datasets (LUCAS, OSSL)
- Benchmark against existing tools

---

## 6. Key Dependencies

### Core Libraries
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

### Deep Learning
```
torch>=2.0.0
torch-geometric>=2.3.0  # For MPNN
lightning>=2.0.0        # For training infrastructure
```

### Spectroscopy
```
brukeropusreader>=1.3.0  # Bruker file reading
pywavelets>=1.4.0        # Wavelet transforms
```

### Utilities
```
pydantic>=2.0.0         # Data validation
typer>=0.9.0            # CLI interface
rich>=13.0.0            # Terminal formatting
mlflow>=2.8.0           # Experiment tracking
```

### Development
```
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
mypy>=1.5.0
ruff>=0.1.0
```

---

## 7. Success Criteria

### Technical Metrics

1. **Preprocessing Accuracy**
   - Match R prospectr outputs within 0.1% error
   - Process 1000 spectra/second on CPU

2. **Model Performance (MIR soil property prediction)**
   - R² > 0.85 for major properties (SOC, clay, sand)
   - RMSE competitive with OSSL Cubist models
   - Physics-informed models show improved extrapolation

3. **Uncertainty Quantification**
   - Conformal prediction coverage > 90%
   - Ensemble uncertainty correlates with prediction error

4. **Compatibility**
   - Successfully load and use OSSL pre-trained models
   - Read all Bruker Alpha II file formats
   - Scikit-learn pipeline integration

### Usability Metrics

1. **Documentation**
   - Complete API reference
   - 10+ tutorial notebooks
   - Deployment examples

2. **Testing**
   - Unit test coverage > 90%
   - Integration tests for all major workflows
   - Continuous integration passing

3. **Performance**
   - Training time < 2 hours for standard dataset (10k spectra)
   - Prediction latency < 100ms for single spectrum

---

## 8. Risk Analysis & Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Bruker file format incompatibility | High | Medium | Use existing brukeropusreader library, extensive testing |
| Physics constraints too restrictive | Medium | Medium | Adjustable physics weights, ablation studies |
| PINN training instability | High | Medium | Careful hyperparameter tuning, curriculum learning |
| MPNN computational cost | Medium | High | Optimize with JIT compilation, consider smaller architectures |
| OSSL model integration issues | Medium | Low | Direct collaboration with OSSL team |

### Resource Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Limited training data | High | Medium | Data augmentation, transfer learning from OSSL |
| GPU requirements | Medium | Low | Support CPU training, cloud GPU options |
| Development timeline | Medium | Medium | Phased approach, MVP focus |

---

## 9. Future Extensions

### Short-term (6 months)
- Web API for model serving
- GUI for interactive analysis
- Integration with cloud storage (S3, GCS)
- Support for other spectrometer brands (Nicolet, Shimadzu)

### Medium-term (1 year)
- Real-time spectral acquisition and prediction
- Active learning for optimal sampling
- Federated learning across institutions
- Mobile app for field measurements

### Long-term (2+ years)
- Foundation models for spectroscopy (pretrain on massive datasets)
- Multi-modal learning (combine spectra with images, text metadata)
- Causal inference for soil property relationships
- Automated experimental design and optimization

---

## 10. Conclusion

This comprehensive plan outlines the development of a state-of-the-art Python package for physics-informed machine learning applied to soil spectroscopy. By combining:

1. **Solid engineering** - Robust I/O, preprocessing, and traditional ML
2. **Cutting-edge research** - PINNs, MPNNs, hierarchical networks
3. **Domain expertise** - Spectroscopy physics, soil science
4. **Community integration** - OSSL compatibility, open source

The `soilspec_pinn` package will provide researchers and practitioners with powerful tools for analyzing soil mid-IR spectra from Bruker Alpha II instruments and beyond.

The phased implementation approach ensures early deliverables while building toward advanced capabilities. Integration with existing tools (OSSL, prospectr) ensures immediate utility, while novel PINN and MPNN models push the boundaries of what's possible in spectral analysis.

**Next Step**: Begin Phase 1 implementation with package setup and I/O module development.

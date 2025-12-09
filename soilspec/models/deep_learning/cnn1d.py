"""
1D Convolutional Neural Networks for soil spectroscopy.

Implements 1D CNNs that directly process spectral data without
converting to 2D images (more natural than GADF + 2D CNN approach).

Scientific Background
---------------------
1D CNNs learn local spectral features through convolution filters,
similar to how chemists identify absorption peaks and bands. This is
more natural for spectral data than converting to 2D images.

**Advantages over 2D approaches:**

* Direct 1D processing - no artificial 2D transformation
* Fewer parameters than 2D CNNs
* Faster training and inference
* More interpretable filters (correspond to spectral features)

**Performance:**

With sufficient data (>5000 samples), 1D CNNs can match or exceed
traditional methods (MBL, Cubist). For small datasets (<1000 samples),
traditional methods typically perform better.

References
----------
.. [1] Padarian, J., et al. (2019). Using deep learning for digital soil
       mapping. Soil 5(1):79-89.
.. [2] Tsakiridis, N.L., et al. (2020). Simultaneous prediction of soil
       properties from VNIR-SWIR spectra using a localized multi-channel 1-D
       CNN. Geoderma 367:114208.
.. [3] Liu, L., et al. (2019). Transferability of a visible and near-infrared
       model for soil organic carbon estimation in riparian landscapes.
       Remote Sensing 11(20):2438.
"""

import numpy as np
from typing import Optional, List, Tuple

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # Dummy for type hints


class SimpleCNN1D(nn.Module if TORCH_AVAILABLE else object):
    """
    Simple 1D CNN for spectral regression.

    **Architecture:**

    * 3-4 convolutional layers with increasing filters
    * MaxPooling for downsampling
    * Fully connected layers for regression
    * Dropout for regularization

    **Use Cases:**

    * Good baseline for spectral regression
    * Works well with 5k-50k samples
    * Faster than complex architectures

    Parameters
    ----------
    input_size : int
        Number of wavelengths (e.g., 1801 for MIR)
    n_filters : list of int, default=[32, 64, 128]
        Number of filters in each conv layer
    kernel_sizes : list of int, default=[7, 5, 3]
        Kernel sizes for each conv layer
    pool_sizes : list of int, default=[2, 2, 2]
        Pooling sizes after each conv layer
    fc_sizes : list of int, default=[128, 64]
        Sizes of fully connected layers
    dropout : float, default=0.3
        Dropout probability
    n_outputs : int, default=1
        Number of output values (1 for single property, >1 for multi-task)

    Examples
    --------
    >>> import torch
    >>> from soilspec.models.deep_learning import SimpleCNN1D
    >>>
    >>> # Create model for MIR spectra (1801 wavelengths)
    >>> model = SimpleCNN1D(
    >>>     input_size=1801,
    >>>     n_filters=[32, 64, 128],
    >>>     kernel_sizes=[7, 5, 3]
    >>> )
    >>>
    >>> # Forward pass
    >>> X = torch.randn(16, 1, 1801)  # (batch, channels, wavelengths)
    >>> y_pred = model(X)
    >>> print(y_pred.shape)  # (16, 1)
    >>>
    >>> # Multi-task (predict SOC, clay, pH)
    >>> model_multi = SimpleCNN1D(input_size=1801, n_outputs=3)
    >>> y_pred = model_multi(X)  # (16, 3)

    Notes
    -----
    Input shape: (batch_size, 1, n_wavelengths)
    Output shape: (batch_size, n_outputs)
    """

    def __init__(
        self,
        input_size: int,
        n_filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3],
        pool_sizes: List[int] = [2, 2, 2],
        fc_sizes: List[int] = [128, 64],
        dropout: float = 0.3,
        n_outputs: int = 1
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        super().__init__()

        self.input_size = input_size
        self.n_filters = n_filters
        self.n_outputs = n_outputs

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for n_filter, kernel_size, pool_size in zip(n_filters, kernel_sizes, pool_sizes):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, n_filter,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(n_filter),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_size),
                    nn.Dropout(dropout)
                )
            )
            in_channels = n_filter

        # Calculate size after convolutions
        conv_output_size = self._get_conv_output_size()

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = conv_output_size

        for fc_size in fc_sizes:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, fc_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_features = fc_size

        # Output layer
        self.output = nn.Linear(in_features, n_outputs)

    def _get_conv_output_size(self) -> int:
        """Calculate output size after conv layers."""
        x = torch.zeros(1, 1, self.input_size)
        for conv in self.conv_layers:
            x = conv(x)
        return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 1, wavelengths)
            Input spectra

        Returns
        -------
        out : Tensor of shape (batch, n_outputs)
            Predictions
        """
        # Convolutional layers
        for conv in self.conv_layers:
            x = conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        for fc in self.fc_layers:
            x = fc(x)

        # Output
        out = self.output(x)

        return out


class ResNet1D(nn.Module if TORCH_AVAILABLE else object):
    """
    1D ResNet for spectral regression with skip connections.

    **Architecture:**

    * Residual blocks with skip connections
    * Prevents vanishing gradients in deep networks
    * Better feature learning than simple CNNs

    **Use Cases:**

    * Large datasets (>10k samples)
    * Complex spectral patterns
    * Multi-task learning

    Parameters
    ----------
    input_size : int
        Number of wavelengths
    n_blocks : int, default=3
        Number of residual blocks
    n_filters : int, default=64
        Base number of filters (doubles in each block)
    kernel_size : int, default=7
        Kernel size for convolutions
    dropout : float, default=0.2
        Dropout probability
    n_outputs : int, default=1
        Number of outputs

    Examples
    --------
    >>> from soilspec.models.deep_learning import ResNet1D
    >>>
    >>> model = ResNet1D(
    >>>     input_size=1801,
    >>>     n_blocks=4,
    >>>     n_filters=64
    >>> )
    >>>
    >>> # Forward pass
    >>> X = torch.randn(32, 1, 1801)
    >>> y_pred = model(X)

    Notes
    -----
    ResNets perform best with larger datasets (>10k samples) and
    sufficient training epochs (50-200 epochs with early stopping).
    """

    def __init__(
        self,
        input_size: int,
        n_blocks: int = 3,
        n_filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.2,
        n_outputs: int = 1
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        super().__init__()

        self.input_size = input_size

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(n_filters),
            nn.ReLU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_filters = n_filters

        for i in range(n_blocks):
            out_filters = in_filters * 2 if i > 0 else in_filters
            self.res_blocks.append(
                ResidualBlock1D(
                    in_filters, out_filters,
                    kernel_size=kernel_size,
                    downsample=(i > 0),
                    dropout=dropout
                )
            )
            in_filters = out_filters

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Output layer
        self.fc = nn.Linear(in_filters, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.gap(x).squeeze(-1)
        out = self.fc(x)

        return out


class ResidualBlock1D(nn.Module if TORCH_AVAILABLE else object):
    """
    Residual block for 1D spectral data.

    Implements skip connection: output = F(x) + x

    where F(x) is two conv layers with batch norm and ReLU.
    """

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        kernel_size: int = 7,
        downsample: bool = False,
        dropout: float = 0.2
    ):
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv1d(
            in_filters, out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = nn.Conv1d(
            out_filters, out_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.dropout = nn.Dropout(dropout)

        # Skip connection adjustment
        if in_filters != out_filters or downsample:
            self.skip = nn.Sequential(
                nn.Conv1d(in_filters, out_filters, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_filters)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with skip connection."""
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out


class MultiScaleCNN1D(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-scale 1D CNN with parallel conv paths.

    **Architecture:**

    * Multiple parallel convolutional paths with different kernel sizes
    * Captures features at multiple scales (narrow and broad peaks)
    * Concatenates multi-scale features before FC layers

    **Inspiration:**

    Similar to Inception architecture but for 1D spectral data.

    Parameters
    ----------
    input_size : int
        Number of wavelengths
    kernel_sizes : list of int, default=[3, 7, 15]
        Multiple kernel sizes for parallel paths
    n_filters : int, default=64
        Number of filters per path
    n_blocks : int, default=2
        Number of multi-scale blocks
    dropout : float, default=0.3
        Dropout probability
    n_outputs : int, default=1
        Number of outputs

    Examples
    --------
    >>> from soilspec.models.deep_learning import MultiScaleCNN1D
    >>>
    >>> # Capture narrow (3), medium (7), and broad (15) spectral features
    >>> model = MultiScaleCNN1D(
    >>>     input_size=1801,
    >>>     kernel_sizes=[3, 7, 15],
    >>>     n_filters=32
    >>> )

    Notes
    -----
    Multi-scale CNNs work well for spectra with features at different scales
    (sharp peaks + broad absorption bands).
    """

    def __init__(
        self,
        input_size: int,
        kernel_sizes: List[int] = [3, 7, 15],
        n_filters: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.3,
        n_outputs: int = 1
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        super().__init__()

        self.input_size = input_size
        self.kernel_sizes = kernel_sizes

        # Multi-scale blocks
        self.ms_blocks = nn.ModuleList()
        in_channels = 1

        for i in range(n_blocks):
            self.ms_blocks.append(
                MultiScaleBlock1D(
                    in_channels, n_filters,
                    kernel_sizes=kernel_sizes,
                    dropout=dropout
                )
            )
            in_channels = n_filters * len(kernel_sizes)

        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for ms_block in self.ms_blocks:
            x = ms_block(x)

        x = self.gap(x).squeeze(-1)
        out = self.fc(x)

        return out


class MultiScaleBlock1D(nn.Module if TORCH_AVAILABLE else object):
    """Multi-scale block with parallel convolutions."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes: List[int] = [3, 7, 15],
        dropout: float = 0.3
    ):
        super().__init__()

        self.paths = nn.ModuleList()

        for kernel_size in kernel_sizes:
            self.paths.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, n_filters,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )

        # Pooling path (max pool + 1x1 conv)
        self.pool_path = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel paths + concatenation."""
        outputs = [path(x) for path in self.paths]
        outputs.append(self.pool_path(x))

        # Concatenate along channel dimension
        out = torch.cat(outputs, dim=1)

        return out


def create_cnn1d(
    model_type: str = 'simple',
    input_size: int = 1801,
    n_outputs: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create 1D CNN models.

    Parameters
    ----------
    model_type : str, default='simple'
        Model architecture: 'simple', 'resnet', or 'multiscale'
    input_size : int, default=1801
        Number of wavelengths (1801 for MIR, varies for VISNIR)
    n_outputs : int, default=1
        Number of output properties
    **kwargs
        Additional arguments passed to model constructor

    Returns
    -------
    model : nn.Module
        Initialized model

    Examples
    --------
    >>> from soilspec.models.deep_learning import create_cnn1d
    >>>
    >>> # Simple CNN
    >>> model = create_cnn1d('simple', input_size=1801)
    >>>
    >>> # ResNet for large dataset
    >>> model = create_cnn1d('resnet', input_size=1801, n_blocks=4)
    >>>
    >>> # Multi-scale for complex spectra
    >>> model = create_cnn1d('multiscale', input_size=1801, kernel_sizes=[3, 7, 15, 31])
    """
    models = {
        'simple': SimpleCNN1D,
        'resnet': ResNet1D,
        'multiscale': MultiScaleCNN1D
    }

    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](input_size=input_size, n_outputs=n_outputs, **kwargs)

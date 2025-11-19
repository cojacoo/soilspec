"""
Format converters for spectral data.

Provides functions to convert between different spectral representations
(absorbance, reflectance, transmittance, etc.).
"""

import numpy as np
from typing import Union


def convert_to_absorbance(
    data: np.ndarray, input_type: str = "reflectance"
) -> np.ndarray:
    """
    Convert spectral data to absorbance.

    Args:
        data: Input spectral data array
        input_type: Type of input data ('reflectance', 'transmittance')

    Returns:
        Absorbance array

    Note:
        - From reflectance: A = log10(1/R)
        - From transmittance: A = log10(1/T) = -log10(T)
    """
    if input_type == "reflectance":
        # Avoid division by zero and log of zero
        data_clipped = np.clip(data, 1e-10, 1.0)
        return np.log10(1.0 / data_clipped)
    elif input_type == "transmittance":
        data_clipped = np.clip(data, 1e-10, 1.0)
        return -np.log10(data_clipped)
    elif input_type == "absorbance":
        return data
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def convert_to_reflectance(
    data: np.ndarray, input_type: str = "absorbance"
) -> np.ndarray:
    """
    Convert spectral data to reflectance.

    Args:
        data: Input spectral data array
        input_type: Type of input data ('absorbance', 'transmittance')

    Returns:
        Reflectance array

    Note:
        - From absorbance: R = 10^(-A)
        - From transmittance: R ≈ T (approximation for diffuse reflectance)
    """
    if input_type == "absorbance":
        return np.power(10.0, -data)
    elif input_type == "transmittance":
        # This is an approximation - actual conversion depends on geometry
        return data
    elif input_type == "reflectance":
        return data
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def convert_to_transmittance(
    data: np.ndarray, input_type: str = "absorbance"
) -> np.ndarray:
    """
    Convert spectral data to transmittance.

    Args:
        data: Input spectral data array
        input_type: Type of input data ('absorbance', 'reflectance')

    Returns:
        Transmittance array

    Note:
        - From absorbance: T = 10^(-A)
    """
    if input_type == "absorbance":
        return np.power(10.0, -data)
    elif input_type == "reflectance":
        # This is an approximation
        return data
    elif input_type == "transmittance":
        return data
    else:
        raise ValueError(f"Unknown input type: {input_type}")

"""
Data input/output module for soil spectroscopy data.

This module provides readers and converters for various spectral file formats,
with a focus on Bruker OPUS files and OSSL formats.
"""

from soilspec_pinn.io.bruker import BrukerReader, Spectrum
from soilspec_pinn.io.ossl import OSSLReader
from soilspec_pinn.io.converters import convert_to_absorbance, convert_to_reflectance

__all__ = [
    "BrukerReader",
    "Spectrum",
    "OSSLReader",
    "convert_to_absorbance",
    "convert_to_reflectance",
]

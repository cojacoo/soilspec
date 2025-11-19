"""
Integration with external tools and model repositories.

Provides access to OSSL pre-trained models and compatibility
with other spectroscopy tools.
"""

from soilspec.integration.ossl_models import OSSLModelLoader
from soilspec.integration.model_zoo import ModelZoo, download_pretrained_model

__all__ = [
    "OSSLModelLoader",
    "ModelZoo",
    "download_pretrained_model",
]

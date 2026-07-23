"""Subset-DIC: C++ implementation of Subset-DIC algorithms with Python bindings."""

from .dic import SubsetDIC, DicResult
from .config import load_config

__version__ = "0.1.1"
__all__ = ["SubsetDIC", "DicResult", "load_config"]

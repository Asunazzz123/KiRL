"""Dataset module for pluggable datasets."""

from .base import BaseDataset
from .factory import build_dataset

__all__ = ["BaseDataset", "build_dataset"]

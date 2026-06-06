"""Dataset implementations."""
from .base import MultimodalEODataset
from .houston import HoustonDataset
from .trento import TrentoDataset
from .muufl import MUUFLDataset
from .patched import (
    PatchedMultimodalDataset,
    HoustonPatchedDataset,
    TrentoPatchedDataset,
    MUUFLPatchedDataset,
)
from .registry import DATASET_REGISTRY, DatasetSpec, get_spec

__all__ = [
    "MultimodalEODataset",
    "HoustonDataset",
    "TrentoDataset",
    "MUUFLDataset",
    # Patched format (MFT style, pre-extracted patches)
    "PatchedMultimodalDataset",
    "HoustonPatchedDataset",
    "TrentoPatchedDataset",
    "MUUFLPatchedDataset",
    # Per-dataset spec registry (single source of truth)
    "DATASET_REGISTRY",
    "DatasetSpec",
    "get_spec",
]

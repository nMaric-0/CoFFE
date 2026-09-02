"""Dataset implementations."""

from .base import MultimodalEODataset
from .houston import HoustonDataset
from .muufl import MUUFLDataset
from .patched import (
    HoustonPatchedDataset,
    MUUFLPatchedDataset,
    PatchedMultimodalDataset,
    TrentoPatchedDataset,
)
from .registry import DATASET_REGISTRY, DatasetSpec, get_spec
from .trento import TrentoDataset

# Grouped by dataset family, with the comments below carrying the grouping.
__all__ = [  # noqa: RUF022
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

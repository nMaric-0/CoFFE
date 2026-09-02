"""
Self-supervised pretraining for CoFFE and the MFT control.

Unified masked autoencoder pretraining: HSI and auxiliary (LiDAR/SAR) bands
are concatenated along the channel dim, per-(pixel, band) masking hides a
configurable fraction of entries, and a single decoder reconstructs the full
combined tensor.

Main components:
- SimMIMPretrainModel: SimMIM-style in-place masked pretraining model
- UnifiedBandMasking: Per-(pixel, band) Bernoulli masking over HSI+aux
- MLPDecoder: Lightweight decoder for reconstruction
- PretrainDataset, CombinedPretrainDataset: Dataset wrappers
"""

from typing import Any

from .decoders import (
    SimpleMLPDecoder,
    TransformerDecoder,
    TwoLayerMLPDecoder,
    build_decoder,
)
from .masked_modeling import (
    CombinedPretrainDataset,
    MLPDecoder,
    PretrainDataset,
    SpatialTokenMasking,
    UnifiedBandMasking,
)
from .simmim import (
    SimMIMPretrainModel,
)

# Moved here in phase 5 from the dissolved top-level ``trainers`` package
# (``trainers/pretrain_trainer.py`` -> ``coffe/pretrain/trainer.py``). That
# package also exported ``create_pretrain_dataloaders``, deleted at the phase-5
# gate: it had no caller and a broken relative import (see CHANGES.md).
from .trainer import PretrainTrainer

__all__ = [
    "CombinedPretrainDataset",
    "MLPDecoder",
    "PretrainDataset",
    "PretrainTrainer",
    "SimMIMPretrainModel",
    "SimpleMLPDecoder",
    "SpatialTokenMasking",
    "TransformerDecoder",
    "TwoLayerMLPDecoder",
    "UnifiedBandMasking",
    "build_decoder",
]


# Legacy class names (PAPER_CANON §1) still resolve, with a DeprecationWarning,
# so notebooks and scripts written before the rename keep importing. The alias
# table lives in one place: coffe.compat.LEGACY_CLASSES.
_LEGACY_ALIASES = {"EnhancedMaskedSpectralSpatialModel": "SimMIMPretrainModel"}


def __getattr__(name: str) -> Any:
    canonical = _LEGACY_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .. import compat

    return getattr(compat, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LEGACY_ALIASES))

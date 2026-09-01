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

from .masked_modeling import (
    UnifiedBandMasking,
    SpatialTokenMasking,
    MLPDecoder,
    PretrainDataset,
    CombinedPretrainDataset,
)

from .simmim import (
    SimMIMPretrainModel,
)

from .decoders import (
    TransformerDecoder,
    SimpleMLPDecoder,
    TwoLayerMLPDecoder,
    build_decoder,
)

__all__ = [
    "SimMIMPretrainModel",
    "UnifiedBandMasking",
    "SpatialTokenMasking",
    "MLPDecoder",
    "PretrainDataset",
    "CombinedPretrainDataset",
    "TransformerDecoder",
    "SimpleMLPDecoder",
    "TwoLayerMLPDecoder",
    "build_decoder",
]


# Legacy class names (PAPER_CANON §1) still resolve, with a DeprecationWarning,
# so notebooks and scripts written before the rename keep importing. The alias
# table lives in one place: coffe_compat.LEGACY_CLASSES.
_LEGACY_ALIASES = {"EnhancedMaskedSpectralSpatialModel": "SimMIMPretrainModel"}


def __getattr__(name):
    canonical = _LEGACY_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import coffe_compat

    return getattr(coffe_compat, name)


def __dir__():
    return sorted(set(__all__) | set(_LEGACY_ALIASES))

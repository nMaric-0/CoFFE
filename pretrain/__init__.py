"""
Self-supervised pretraining module for MFT-CPEA.

Unified masked autoencoder pretraining: HSI and auxiliary (LiDAR/SAR) bands
are concatenated along the channel dim, per-(pixel, band) masking hides a
configurable fraction of entries, and a single decoder reconstructs the full
combined tensor.

Main components:
- EnhancedMaskedSpectralSpatialModel: Unified masked pretraining model
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

from .masked_modeling_enhanced import (
    EnhancedMaskedSpectralSpatialModel,
)

from .decoders import (
    TransformerDecoder,
    SimpleMLPDecoder,
    TwoLayerMLPDecoder,
    build_decoder,
)

__all__ = [
    "EnhancedMaskedSpectralSpatialModel",
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

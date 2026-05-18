"""
Backbone models for feature extraction.

Available backbones:
- MFTChannelBackbone: MFT with channel tokenization for LiDAR
- MFTPixelBackbone: MFT with pixel tokenization for LiDAR
- HSIBaselineBackbone: HSI-only with learnable CLS token

All backbones output:
- patch_emb: [B, num_tokens, embed_dim] for dense similarity
- cls_emb: [B, embed_dim] for CPEA adaptation
"""
from .base import BackboneBase
from .mft_channel import MFTChannelBackbone
from .mft_pixel import MFTPixelBackbone
from .hsi_baseline import HSIBaselineBackbone

__all__ = [
    'BackboneBase',
    'MFTChannelBackbone',
    'MFTPixelBackbone',
    'HSIBaselineBackbone',
]

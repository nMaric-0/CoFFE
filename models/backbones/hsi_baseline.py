"""
HSI Baseline Backbone (No Multimodal Fusion).

Adapted from: https://github.com/srinadh99/Transformer-Models-for-Multimodal-Remote-Sensing-Data

This is the HSI-only baseline that uses a learnable CLS token instead of
deriving it from auxiliary data (LiDAR). Useful for:
- Ablation studies comparing multimodal vs unimodal
- Datasets without auxiliary modalities
- Understanding contribution of LiDAR fusion

Key features:
- 3D Conv + HetConv for HSI feature extraction (same as MFT)
- Learnable CLS token (randomly initialized)
- MCrossAttention or standard self-attention (configurable)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .base import BackboneBase
from ..components.mft_blocks import (
    HSI3DConv, HetConv, LearnableTokenizer,
    MFTEncoder
)


class HSIBaselineBackbone(BackboneBase):
    """
    HSI-only backbone with learnable CLS token.

    Architecture:
        HSI: Conv3D -> HetConv -> Flatten -> LearnableTokenizer -> N tokens
        CLS: Learnable parameter [1, 1, D]
        Encoding: Concat [CLS, HSI_tokens] -> MFTEncoder -> (patch_emb, cls_emb)

    Note: Auxiliary input (LiDAR) is IGNORED in this backbone.
    The forward_features still accepts aux for API compatibility but doesn't use it.

    Args:
        hsi_channels: Number of HSI spectral bands
        aux_channels: Ignored (kept for API compatibility)
        embed_dim: Embedding dimension
        num_tokens: Number of HSI tokens (default: 8)
        num_heads: Number of attention heads
        depth: Number of transformer layers
        mlp_dim: MLP hidden dimension (default: 4x embed_dim)
        patch_size: Input patch spatial size
        dropout: Dropout rate
        attention_type: "mcross" or "standard"
    """

    def __init__(
        self,
        hsi_channels: int,
        aux_channels: int = 1,  # Ignored, kept for API compatibility
        embed_dim: int = 128,
        num_tokens: int = 8,
        num_heads: int = 8,
        depth: int = 2,
        mlp_dim: Optional[int] = None,
        patch_size: int = 11,
        dropout: float = 0.1,
        attention_type: str = "mcross"
    ):
        super().__init__(embed_dim=embed_dim, num_tokens=num_tokens)

        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels  # Stored but not used
        self.patch_size = patch_size
        self.attention_type = attention_type

        # Feature multiplier
        self.fm = embed_dim // 4

        # ========== HSI Feature Extraction ==========
        self.hsi_conv3d = HSI3DConv(
            in_channels=hsi_channels,
            out_channels=self.fm,
            spectral_kernel=7,
            spatial_kernel=3
        )

        self.hsi_hetconv = HetConv(
            in_channels=self.fm * hsi_channels,
            out_channels=self.fm * 4,
            kernel_size=3,
            groups=self.fm
        )

        self.hsi_tokenizer = LearnableTokenizer(
            input_dim=self.fm * 4,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            token_type="channel"
        )

        # ========== Learnable CLS Token ==========
        # This is the key difference from MFT backbones
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )

        # ========== Positional Embedding ==========
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens + 1, embed_dim) * 0.02
        )

        # ========== Transformer Encoder ==========
        self.encoder = MFTEncoder(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_type=attention_type
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_hsi_features(self, hsi: torch.Tensor) -> torch.Tensor:
        """
        Extract HSI features using 3D Conv + HetConv pipeline.

        Args:
            hsi: [B, C, H, W] hyperspectral data

        Returns:
            features: [B, N_spatial, D] flattened features
        """
        B, C, H, W = hsi.shape

        # 3D convolution
        x = self.hsi_conv3d(hsi)

        # Reshape for HetConv
        x = x.reshape(B, -1, H, W)

        # HetConv
        x = self.hsi_hetconv(x)

        # Flatten spatial
        x = x.flatten(2).transpose(1, 2)

        return x

    def forward_tokens_only(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract tokens BEFORE transformer encoder (for MAE pretraining).

        Returns:
            hsi_tokens: [B, num_tokens, D] HSI tokens (pre-encoder, target for MAE)
            cls_token: [B, 1, D] CLS token (learnable, pre-encoder)
        """
        B = hsi.shape[0]
        hsi_features = self.extract_hsi_features(hsi)
        hsi_tokens = self.hsi_tokenizer(hsi_features)
        cls_token = self.cls_token.expand(B, -1, -1)

        return hsi_tokens, cls_token

    def forward_with_mask(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with token masking for MAE pretraining.

        Args:
            hsi: [B, C, H, W] hyperspectral data
            aux: Ignored (for API compatibility)
            mask: [B, num_tokens] boolean mask (True = masked)

        Returns:
            encoded_visible: [B, num_visible+1, D] encoded visible tokens + CLS
            pre_encoder_tokens: [B, num_tokens, D] original tokens (target)
            mask: [B, num_tokens] the mask used
        """
        hsi_tokens, cls_token = self.forward_tokens_only(hsi, aux)
        pre_encoder_tokens = hsi_tokens.clone()

        B = hsi.shape[0]

        visible_tokens_list = []
        for i in range(B):
            visible = hsi_tokens[i, ~mask[i]]
            visible_tokens_list.append(visible)

        num_visible = visible_tokens_list[0].shape[0]
        visible_tokens = torch.stack(visible_tokens_list)

        tokens = torch.cat([cls_token, visible_tokens], dim=1)

        cls_pos = self.pos_embed[:, 0:1, :]
        visible_pos_list = []
        for i in range(B):
            visible_indices = (~mask[i]).nonzero(as_tuple=True)[0]
            pos = self.pos_embed[:, visible_indices + 1, :]
            visible_pos_list.append(pos.squeeze(0))

        visible_pos = torch.stack(visible_pos_list)
        pos_embed = torch.cat([cls_pos.expand(B, -1, -1), visible_pos], dim=1)

        tokens = tokens + pos_embed
        encoded = self.encoder(tokens)

        return encoded, pre_encoder_tokens, mask

    def forward_features(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor = None  # Ignored
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from HSI input (aux is ignored).

        Args:
            hsi: [B, C, H, W] hyperspectral data
            aux: Ignored (kept for API compatibility)

        Returns:
            patch_emb: [B, num_tokens, embed_dim] HSI patch embeddings
            cls_emb: [B, embed_dim] class embedding (learnable)
        """
        B = hsi.shape[0]

        # Extract HSI features
        hsi_features = self.extract_hsi_features(hsi)

        # Tokenize HSI
        hsi_tokens = self.hsi_tokenizer(hsi_features)  # [B, num_tokens, D]

        # Expand learnable CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        # Concatenate: [CLS, HSI_tokens]
        tokens = torch.cat([cls_tokens, hsi_tokens], dim=1)

        # Add positional embedding
        tokens = tokens + self.pos_embed

        # Transformer encoding
        tokens = self.encoder(tokens)

        # Split output
        cls_emb = tokens[:, 0]
        patch_emb = tokens[:, 1:]

        return patch_emb, cls_emb

    def forward_hsi_only(self, hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for HSI-only forward pass.
        Same as forward_features but makes it explicit that aux is not needed.
        """
        return self.forward_features(hsi, aux=None)

    # Alternative interface for backward compatibility
    # def forward_features_full(
    #     self,
    #     hsi: torch.Tensor,
    #     aux: torch.Tensor = None
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     patch_emb, cls_emb = self.forward_features(hsi, aux)
    #     # Return cls_emb as both cls and aux (no separate aux in this backbone)
    #     return patch_emb, cls_emb, cls_emb

"""
MFT Channel Tokenization Backbone.

Adapted from: https://github.com/srinadh99/Transformer-Models-for-Multimodal-Remote-Sensing-Data

This backbone uses channel-wise attention tokenization for the auxiliary
modality (LiDAR), creating a fusion token that serves as the CLS token.

Key features:
- 3D Conv + HetConv for HSI feature extraction
- Channel tokenization for LiDAR -> single fusion/CLS token
- Learnable tokenization for HSI -> configurable number of tokens
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


class MFTChannelBackbone(BackboneBase):
    """
    MFT backbone with channel tokenization for multimodal fusion.

    Architecture:
        HSI: Conv3D -> HetConv -> Flatten -> LearnableTokenizer -> N tokens
        LiDAR: Conv2D -> Flatten -> ChannelTokenizer -> 1 token (CLS)
        Fusion: Concat [CLS, HSI_tokens] -> MFTEncoder -> (patch_emb, cls_emb)

    Args:
        hsi_channels: Number of HSI spectral bands
        aux_channels: Number of auxiliary data channels (e.g., LiDAR)
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
        aux_channels: int = 1,
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
        self.aux_channels = aux_channels
        self.patch_size = patch_size
        self.attention_type = attention_type

        # Feature multiplier (as in MFT)
        self.fm = embed_dim // 4  # 32 for embed_dim=128

        # ========== HSI Feature Extraction ==========
        # 3D Conv for spectral-spatial processing
        self.hsi_conv3d = HSI3DConv(
            in_channels=hsi_channels,
            out_channels=self.fm,
            spectral_kernel=7,
            spatial_kernel=3
        )

        # HetConv for efficient spatial processing
        self.hsi_hetconv = HetConv(
            in_channels=self.fm * hsi_channels,  # After reshape from 3D
            out_channels=self.fm * 4,  # = embed_dim
            kernel_size=3,
            groups=self.fm
        )

        # Learnable tokenization for HSI
        self.hsi_tokenizer = LearnableTokenizer(
            input_dim=self.fm * 4,  # embed_dim
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            token_type="channel"
        )

        # ========== Auxiliary (LiDAR) Feature Extraction ==========
        # Conv2D for LiDAR
        self.aux_conv = nn.Sequential(
            nn.Conv2d(aux_channels, self.fm * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.fm * 4),
            nn.GELU()
        )

        # Channel tokenization for LiDAR -> 1 token (serves as CLS)
        self.aux_tokenizer = LearnableTokenizer(
            input_dim=self.fm * 4,
            embed_dim=embed_dim,
            num_tokens=1,  # Single fusion token
            token_type="channel"
        )

        # ========== Positional Embedding ==========
        # +1 for CLS/fusion token
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

        # 3D convolution: [B, C, H, W] -> [B, fm, C, H, W]
        x = self.hsi_conv3d(hsi)

        # Reshape for HetConv: [B, fm, C, H, W] -> [B, fm*C, H, W]
        x = x.reshape(B, -1, H, W)

        # HetConv: [B, fm*C, H, W] -> [B, fm*4, H, W]
        x = self.hsi_hetconv(x)

        # Flatten spatial: [B, D, H, W] -> [B, H*W, D]
        x = x.flatten(2).transpose(1, 2)

        return x

    def extract_aux_features(self, aux: torch.Tensor) -> torch.Tensor:
        """
        Extract auxiliary (LiDAR) features.

        Args:
            aux: [B, C_aux, H, W] auxiliary data

        Returns:
            features: [B, N_spatial, D] flattened features
        """
        # Conv2D: [B, C_aux, H, W] -> [B, D, H, W]
        x = self.aux_conv(aux)

        # Flatten spatial: [B, D, H, W] -> [B, H*W, D]
        x = x.flatten(2).transpose(1, 2)

        return x

    def forward_tokens_only(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract tokens BEFORE transformer encoder (for MAE pretraining).

        Returns:
            hsi_tokens: [B, num_tokens, D] HSI tokens (pre-encoder, target for MAE)
            cls_token: [B, 1, D] CLS/fusion token (pre-encoder)
        """
        # Extract raw features
        hsi_features = self.extract_hsi_features(hsi)  # [B, H*W, D]
        aux_features = self.extract_aux_features(aux)  # [B, H*W, D]

        # Tokenize (this is what we want to reconstruct in MAE)
        hsi_tokens = self.hsi_tokenizer(hsi_features)  # [B, num_tokens, D]
        cls_token = self.aux_tokenizer(aux_features)   # [B, 1, D]

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
            aux: [B, C_aux, H, W] auxiliary data
            mask: [B, num_tokens] boolean mask (True = masked, to be reconstructed)

        Returns:
            encoded_visible: [B, num_visible+1, D] encoded visible tokens + CLS
            pre_encoder_tokens: [B, num_tokens, D] original tokens (target)
            mask: [B, num_tokens] the mask used
        """
        # Get pre-encoder tokens (these are the reconstruction targets)
        hsi_tokens, cls_token = self.forward_tokens_only(hsi, aux)
        pre_encoder_tokens = hsi_tokens.clone()  # [B, num_tokens, D]

        B = hsi.shape[0]

        # Apply mask: keep only visible tokens
        visible_tokens_list = []
        for i in range(B):
            visible = hsi_tokens[i, ~mask[i]]  # [num_visible, D]
            visible_tokens_list.append(visible)

        # Pad to same length (all samples have same mask ratio, so same length)
        num_visible = visible_tokens_list[0].shape[0]
        visible_tokens = torch.stack(visible_tokens_list)  # [B, num_visible, D]

        # Concatenate CLS + visible tokens
        tokens = torch.cat([cls_token, visible_tokens], dim=1)  # [B, 1+num_visible, D]

        # Add positional embedding (only for visible positions)
        # CLS always gets position 0
        cls_pos = self.pos_embed[:, 0:1, :]  # [1, 1, D]

        # Get positions for visible tokens
        visible_pos_list = []
        for i in range(B):
            visible_indices = (~mask[i]).nonzero(as_tuple=True)[0]  # indices of visible tokens
            # +1 because pos_embed[0] is CLS
            pos = self.pos_embed[:, visible_indices + 1, :]  # [1, num_visible, D]
            visible_pos_list.append(pos.squeeze(0))

        visible_pos = torch.stack(visible_pos_list)  # [B, num_visible, D]
        pos_embed = torch.cat([cls_pos.expand(B, -1, -1), visible_pos], dim=1)

        tokens = tokens + pos_embed

        # Transformer encoding (only on visible tokens)
        encoded = self.encoder(tokens)  # [B, 1+num_visible, D]

        return encoded, pre_encoder_tokens, mask

    def forward_features(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from multimodal input.

        Args:
            hsi: [B, C, H, W] hyperspectral data
            aux: [B, C_aux, H, W] auxiliary data (LiDAR)

        Returns:
            patch_emb: [B, num_tokens, embed_dim] HSI patch embeddings
            cls_emb: [B, embed_dim] class/fusion embedding (from LiDAR)
        """
        # Extract raw features
        hsi_features = self.extract_hsi_features(hsi)  # [B, H*W, D]
        aux_features = self.extract_aux_features(aux)  # [B, H*W, D]

        # Tokenize
        hsi_tokens = self.hsi_tokenizer(hsi_features)  # [B, num_tokens, D]
        cls_token = self.aux_tokenizer(aux_features)   # [B, 1, D]

        # Concatenate: [CLS, HSI_tokens]
        tokens = torch.cat([cls_token, hsi_tokens], dim=1)  # [B, 1+num_tokens, D]

        # Add positional embedding
        tokens = tokens + self.pos_embed

        # Transformer encoding
        tokens = self.encoder(tokens)

        # Split output
        cls_emb = tokens[:, 0]           # [B, D]
        patch_emb = tokens[:, 1:]        # [B, num_tokens, D]

        return patch_emb, cls_emb

    # Alternative interface for backward compatibility with original MFTCPEA
    # def forward_features_full(
    #     self,
    #     hsi: torch.Tensor,
    #     aux: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Extract features with separate aux embedding.
    #
    #     Returns:
    #         patch_emb: [B, num_tokens, D]
    #         cls_emb: [B, D]
    #         aux_emb: [B, D] (same as cls_emb in this architecture)
    #     """
    #     patch_emb, cls_emb = self.forward_features(hsi, aux)
    #     return patch_emb, cls_emb, cls_emb  # aux_emb = cls_emb

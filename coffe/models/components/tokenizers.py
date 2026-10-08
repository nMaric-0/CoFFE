"""Tokenizers for multimodal EO data."""

import torch
import torch.nn as nn


class ChannelTokenizer(nn.Module):
    """
    Channel tokenizer for HSI data.
    Converts spectral bands to embedding dimension.
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] HSI data
        Returns:
            [B, D, H, W] embedded features
        """
        return self.conv(x)


class SpatialTokenizer(nn.Module):
    """
    Spatial tokenizer.
    Applies spatial convolution and flattens to tokens.
    """

    def __init__(self, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W] embedded features
        Returns:
            [B, H*W, D] spatial tokens
        """
        x = self.conv(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, D]
        return x


class AuxTokenizer(nn.Module):
    """
    Tokenizer for auxiliary modality (LiDAR/SAR/DSM).
    Creates a single token representing the auxiliary data.

    Deprecated: Use SpatialAuxTokenizer for spatial-preserving tokenization.
    """

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.flatten_dim = in_channels * patch_size * patch_size
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] auxiliary data
        Returns:
            [B, 1, D] auxiliary token
        """
        x = x.flatten(1)  # [B, C*H*W]
        x = self.mlp(x)  # [B, D]
        return x.unsqueeze(1)  # [B, 1, D]


class SpatialAuxTokenizer(nn.Module):
    """
    Spatial tokenizer for auxiliary data (LiDAR/SAR/DSM).

    Produces N spatial tokens (same as HSI) preserving spatial structure,
    rather than collapsing to a single token. This allows the transformer
    to attend to spatially-resolved auxiliary features.
    """

    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.GELU()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_aux, H, W] auxiliary data
        Returns:
            [B, H*W, D] spatial auxiliary tokens
        """
        x = self.channel_conv(x)  # [B, D, H, W]
        x = self.spatial_conv(x)  # [B, D, H, W]
        B, D, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # [B, H*W, D]

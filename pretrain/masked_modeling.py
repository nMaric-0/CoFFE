"""
Masked Spectral-Spatial Modeling for Self-Supervised Pretraining.

This module implements masked autoencoder pretraining for multimodal 
Earth observation data (HSI + LiDAR/SAR).

Based on:
- MAE: Masked Autoencoders Are Scalable Vision Learners (He et al., 2022)
- SS-MAE: Spatial-Spectral Masked Autoencoder (Lin et al., 2023)

Key design decisions:
- Reconstruction target: Normalized raw HSI pixels
- Decoder: 2-layer MLP (lightweight, forces encoder to learn)
- Spatial masking: Random token masking (configurable ratio)
- Spectral masking: Optional, random band masking
- LiDAR: Can be used as unmasked context OR also masked
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

from utils.spatial_weights import make_center_weights


class UnifiedBandMasking(nn.Module):
    """
    Per-(pixel, band) Bernoulli masking over a combined HSI+aux tensor.

    Unlike the older token-level / band-level masks, each (band, spatial) entry
    is sampled independently. With mask_ratio=0.9, ~90% of all values (HSI
    bands *and* LiDAR bands) are hidden from the encoder.

    Masked entries are replaced with a learnable per-channel fill value so the
    1x1 channel tokenizer receives a well-defined input at masked positions.
    """

    def __init__(self, num_channels: int, mask_ratio: float = 0.9):
        super().__init__()
        self.num_channels = num_channels
        self.mask_ratio = mask_ratio
        # Learnable fill value per channel (shape [1, C, 1, 1] so it broadcasts over H, W)
        self.mask_value = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        nn.init.normal_(self.mask_value, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] combined HSI+aux input

        Returns:
            x_masked: [B, C, H, W] input with masked entries replaced by the
                      learnable fill value
            mask: [B, C, H, W] binary mask (1 = masked, 0 = visible)
        """
        B, C, H, W = x.shape
        assert C == self.num_channels, (
            f"UnifiedBandMasking expected {self.num_channels} channels, got {C}"
        )
        noise = torch.rand(B, C, H, W, device=x.device, dtype=x.dtype)
        mask = (noise < self.mask_ratio).to(x.dtype)
        x_masked = x * (1.0 - mask) + self.mask_value * mask
        return x_masked, mask


class SpatialTokenMasking(nn.Module):
    """
    MAE-style spatial masking applied after tokenization.

    A random fraction of pixel tokens (whole-pixel, all bands) is replaced with
    a learnable `mask_token`. No sequence gather/scatter is performed; the
    transformer still sees N tokens, but `mask_ratio` of them carry the mask
    token instead of the pixel embedding.
    """

    def __init__(self, embed_dim: int, mask_ratio: float = 0.75):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: [B, N, D] patch-token embeddings (CLS not included).

        Returns:
            tokens_out: [B, N, D] with ~mask_ratio fraction replaced by mask_token.
            token_mask: [B, N] binary (1 = masked, 0 = visible).
        """
        B, N, D = tokens.shape
        num_mask = int(round(self.mask_ratio * N))
        if num_mask <= 0:
            return tokens, torch.zeros(B, N, device=tokens.device, dtype=tokens.dtype)

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)
        masked_idx = ids_shuffle[:, N - num_mask:]
        token_mask = torch.zeros(B, N, device=tokens.device, dtype=tokens.dtype)
        token_mask.scatter_(1, masked_idx, 1.0)

        mask_tokens = self.mask_token.expand(B, N, D)
        tokens_out = torch.where(token_mask.unsqueeze(-1).bool(), mask_tokens, tokens)
        return tokens_out, token_mask


class MLPDecoder(nn.Module):
    """
    Lightweight 2-layer MLP decoder for reconstruction.
    
    Following the principle that a lightweight decoder forces
    the encoder to learn meaningful representations.
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (e.g., num_bands for HSI reconstruction)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., embed_dim] encoded features
        Returns:
            [..., output_dim] reconstructed values
        """
        return self.decoder(x)


class PretrainDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for pretraining.
    
    Provides all pixels (including unlabeled) for self-supervised learning.
    """
    
    def __init__(
        self,
        base_dataset,
        include_unlabeled: bool = True
    ):
        """
        Args:
            base_dataset: Base multimodal EO dataset
            include_unlabeled: Whether to include unlabeled pixels
        """
        self.base_dataset = base_dataset
        self.patch_size = base_dataset.patch_size
        self.pad = self.patch_size // 2
        
        # Build index of all valid pixels
        H, W = base_dataset.labels.shape
        self.valid_coords = []
        
        for y in range(self.pad, H - self.pad):
            for x in range(self.pad, W - self.pad):
                if include_unlabeled or base_dataset.labels[y, x] > 0:
                    self.valid_coords.append((y, x))
    
    def __len__(self) -> int:
        return len(self.valid_coords)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        y, x = self.valid_coords[idx]
        hsi, aux = self.base_dataset.extract_patch(y, x)
        
        return {
            "hsi": hsi,
            "aux": aux,
            "coords": torch.tensor([y, x], dtype=torch.long)
        }


class CombinedPretrainDataset(torch.utils.data.Dataset):
    """
    Combines multiple datasets for pretraining.
    
    Useful for pretraining on Houston + Trento + MUUFL together.
    Handles different numbers of spectral bands via padding/truncation.
    """
    
    def __init__(
        self,
        datasets: list,
        target_hsi_channels: int = 144,
        target_aux_channels: int = 2
    ):
        """
        Args:
            datasets: List of PretrainDataset instances
            target_hsi_channels: Pad/truncate to this many HSI channels
            target_aux_channels: Pad/truncate to this many aux channels
        """
        self.datasets = datasets
        self.target_hsi_channels = target_hsi_channels
        self.target_aux_channels = target_aux_channels
        
        # Build cumulative index
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def _find_dataset(self, idx: int) -> Tuple[int, int]:
        """Find which dataset and local index for global index."""
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                local_idx = idx - (self.cumulative_sizes[i-1] if i > 0 else 0)
                return i, local_idx
        raise IndexError(f"Index {idx} out of range")
    
    def _pad_or_truncate(
        self,
        tensor: torch.Tensor,
        target_channels: int,
        dim: int = 0
    ) -> torch.Tensor:
        """Pad or truncate tensor to target number of channels."""
        current_channels = tensor.shape[dim]
        
        if current_channels == target_channels:
            return tensor
        elif current_channels < target_channels:
            # Pad with zeros
            pad_size = target_channels - current_channels
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=dim)
        else:
            # Truncate
            indices = [slice(None)] * tensor.ndim
            indices[dim] = slice(0, target_channels)
            return tensor[tuple(indices)]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, local_idx = self._find_dataset(idx)
        sample = self.datasets[ds_idx][local_idx]
        
        # Normalize channel counts
        sample["hsi"] = self._pad_or_truncate(
            sample["hsi"], self.target_hsi_channels, dim=0
        )
        sample["aux"] = self._pad_or_truncate(
            sample["aux"], self.target_aux_channels, dim=0
        )
        sample["dataset_idx"] = torch.tensor(ds_idx, dtype=torch.long)
        
        return sample

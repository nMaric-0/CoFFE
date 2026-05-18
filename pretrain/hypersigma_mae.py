"""MAE-style adaptation wrapper for HyperSIGMA on Houston (Level-2).

Wraps a ``HyperSIGMADual`` encoder for masked reconstruction. The
small randomly-initialized components of the dual model
(``patch_embed.proj``, both ``pos_embed`` parameters, ``spat_map``,
``l1``, and the SEM) are MAE-trained so they can speak to the frozen
SpatViT / SpecViT transformer bodies coherently.

Recipe:

* Mask is applied at the raw HSI input via ``UnifiedBandMasking`` from
  ``pretrain.masked_modeling`` (reusing the same learnable per-channel
  fill values as MFT-CPEA's pretraining).
* The masked HSI is fed through the dual encoder. The 512-d SEM output
  is the bottleneck; a small MLP decoder expands it to the
  ``C * patch_size * patch_size`` reconstruction target.
* Loss is computed on the masked (pixel, band) entries only, optionally
  center-weighted via ``utils.spatial_weights.make_center_weights``.

This mirrors the hyperparameter recipe in
``configs/pretrain/houston_pretrain_enhanced.yaml`` (band_mask_ratio,
sigma, decoder hidden dim, optimizer config) so the comparison with
MFT-CPEA is apples-to-apples.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hypersigma.hypersigma_dual import HyperSIGMADual
from pretrain.masked_modeling import UnifiedBandMasking, MLPDecoder
from utils.spatial_weights import make_center_weights

logger = logging.getLogger(__name__)


class HyperSIGMAMaskedAdaptation(nn.Module):
    """MAE-adaptation head sitting on top of a ``HyperSIGMADual`` encoder.

    Args:
        dual:                The ``HyperSIGMADual`` to adapt.
        hsi_channels:        Number of HSI bands in the raw input.
        patch_size:          Spatial patch size (11 for Houston).
        decoder_hidden_dim:  Hidden width of the MLP decoder.
        band_mask_ratio:     Bernoulli mask ratio at the input.
        recon_sigma:         Gaussian sigma for center-weighted recon
                             loss; ``None`` = uniform weighting.
    """

    def __init__(
        self,
        dual: HyperSIGMADual,
        hsi_channels: int = 144,
        patch_size: int = 11,
        decoder_hidden_dim: int = 256,
        band_mask_ratio: float = 0.9,
        recon_sigma: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        if band_mask_ratio <= 0.0:
            raise ValueError("band_mask_ratio must be > 0 for MAE adaptation")

        self.dual = dual
        self.hsi_channels = hsi_channels
        self.patch_size = patch_size
        self.band_mask_ratio = band_mask_ratio

        self.input_masking = UnifiedBandMasking(
            num_channels=hsi_channels, mask_ratio=band_mask_ratio,
        )

        # SEM output -> [B, hsi_channels * patch_size * patch_size]
        feature_dim = dual.num_stages * dual.dr_dim
        target_dim = hsi_channels * patch_size * patch_size
        self.decoder = MLPDecoder(
            embed_dim=feature_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=target_dim,
            dropout=0.1,
        )

        if recon_sigma is not None:
            weights = make_center_weights(patch_size, recon_sigma, normalize=True)
            # Stored as [1, 1, H, W] so it broadcasts over (B, C).
            self.register_buffer("recon_weights", weights.view(1, 1, patch_size, patch_size))
        else:
            self.recon_weights = None

    def trainable_parameters(self):
        """Yield (name, param) pairs for trainable parameters across the
        whole adaptation module (dual + decoder + masking fills)."""
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield n, p

    def forward(self, hsi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the MAE reconstruction loss.

        Args:
            hsi: ``[B, C, H, W]`` raw normalized HSI patch.

        Returns:
            dict with ``loss``, ``mask`` (binary mask, 1 = masked),
            ``recon`` (reconstructed patch), and ``target``.
        """
        B, C, H, W = hsi.shape
        assert C == self.hsi_channels and H == W == self.patch_size, (
            f"expected [B,{self.hsi_channels},{self.patch_size},{self.patch_size}], got {hsi.shape}"
        )

        x_masked, mask = self.input_masking(hsi)          # [B, C, H, W], [B, C, H, W]

        out = self.dual(x_masked)
        fused = out["fused"]                              # [B, 512]
        recon_flat = self.decoder(fused)                  # [B, C*H*W]
        recon = recon_flat.view(B, C, H, W)

        diff_sq = (recon - hsi).pow(2) * mask              # only masked entries contribute
        if self.recon_weights is not None:
            diff_sq = diff_sq * self.recon_weights
            denom = (mask * self.recon_weights).sum().clamp(min=1.0)
        else:
            denom = mask.sum().clamp(min=1.0)
        loss = diff_sq.sum() / denom

        return {
            "loss": loss,
            "mask": mask,
            "recon": recon,
            "target": hsi,
        }

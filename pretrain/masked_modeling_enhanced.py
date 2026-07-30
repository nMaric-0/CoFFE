"""
Masked pretraining with a unified HSI+LiDAR token space.

HSI and auxiliary (LiDAR/SAR/DSM) bands are concatenated along the channel
dimension, forming a single [B, C_hsi + C_aux, H, W] tensor. Two masking
regimes are supported and can be combined:

* Per-(pixel, band) Bernoulli masking (``band_mask_ratio``) hides a fraction
  of individual (pixel, band) entries of the raw input.
* Spatial token masking (``spatial_mask_ratio``) is the original MAE recipe:
  a fraction of whole pixel tokens are replaced with a learnable mask token
  after tokenization.

A lightweight MLP decoder reconstructs the full combined tensor; loss is
computed only on the masked entries (union of the two masks when both are on).

Historical note: previous revisions of this file ran three separate masks
(spatial / spectral / LiDAR) plus a denoising head. All of that is replaced
by the unified mask below.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .masked_modeling import UnifiedBandMasking, SpatialTokenMasking, MLPDecoder
from utils.spatial_weights import make_center_weights


class EnhancedMaskedSpectralSpatialModel(nn.Module):
    """
    Unified masked autoencoder over concatenated HSI + aux bands.

    Supports two masking regimes, selectable (and combinable) via constructor
    arguments:

    * ``band_mask_ratio > 0`` — per-(pixel, band) Bernoulli masking on the raw
      input, replacing masked entries with a learnable per-channel fill value.
    * ``spatial_mask_ratio > 0`` — MAE-style masking at the token level; a
      random fraction of pixel tokens is replaced with a learnable
      ``mask_token``.

    When both are enabled they compose: band masking is applied first, then
    tokenization, then spatial token masking; the reconstruction loss is
    computed over the union of the two masks.

    Args:
        encoder: The MFT-CPEA encoder to pretrain (must expose ``tokenize``,
                 ``encoder``, ``norm``, ``projection``, ``class_agnostic_emb``,
                 ``pos_embed``).
        hsi_channels: Number of HSI spectral bands.
        aux_channels: Number of auxiliary channels (e.g. 1 for LiDAR).
        use_aux: If True (default), aux bands are concatenated with HSI, masked,
                 and reconstructed jointly. If False, the model is HSI-only: aux
                 is ignored and masking/reconstruction cover HSI bands only.
        patch_size: Spatial patch size.
        embed_dim: Encoder embedding dimension.
        decoder_hidden_dim: Decoder hidden dimension.
        band_mask_ratio: Fraction of (pixel, band) entries to mask. Set to 0
                         to disable band masking.
        spatial_mask_ratio: Fraction of whole pixel tokens to mask (MAE style).
                            Set to 0 to disable.
        recon_sigma: Optional Gaussian sigma for center-weighted reconstruction
                     loss. None = uniform weighting.
        recon_loss: Reconstruction loss on masked entries. One of "mse"
                    (default), "l1", or "smooth_l1"/"huber" (SmoothL1, beta=1.0).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hsi_channels: int,
        aux_channels: int = 1,
        use_aux: bool = True,
        patch_size: int = 11,
        embed_dim: int = 128,
        decoder_hidden_dim: int = 256,
        band_mask_ratio: float = 0.9,
        spatial_mask_ratio: float = 0.0,
        recon_sigma: Optional[float] = None,
        recon_loss: str = "mse",
    ):
        super().__init__()

        recon_loss = (recon_loss or "mse").lower()
        if recon_loss not in ("mse", "l1", "smooth_l1", "huber"):
            raise ValueError(
                f"recon_loss must be one of mse/l1/smooth_l1/huber, got {recon_loss!r}"
            )
        self.recon_loss = recon_loss

        if band_mask_ratio <= 0 and spatial_mask_ratio <= 0:
            raise ValueError(
                "At least one of band_mask_ratio / spatial_mask_ratio must be > 0"
            )

        self.encoder = encoder
        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels
        self.use_aux = use_aux
        self.total_channels = hsi_channels + aux_channels if use_aux else hsi_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_tokens = patch_size * patch_size
        self.band_mask_ratio = band_mask_ratio
        self.spatial_mask_ratio = spatial_mask_ratio
        self.recon_sigma = recon_sigma

        self.use_band_mask = band_mask_ratio > 0
        self.use_spatial_mask = spatial_mask_ratio > 0

        if recon_sigma is not None:
            rw = make_center_weights(patch_size, recon_sigma, normalize=False)
            rw = rw / rw.mean()  # mean weight 1.0 so loss scale is preserved
            self.register_buffer('_recon_center_weights', rw)

        if self.use_band_mask:
            self.band_masking = UnifiedBandMasking(
                num_channels=self.total_channels,
                mask_ratio=band_mask_ratio,
            )

        if self.use_spatial_mask:
            self.spatial_masking = SpatialTokenMasking(
                embed_dim=embed_dim,
                mask_ratio=spatial_mask_ratio,
            )

        # Single decoder reconstructs HSI + aux values for every pixel.
        self.decoder = MLPDecoder(
            embed_dim=embed_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=self.total_channels,
        )

        self.enc_to_dec = nn.Linear(embed_dim, embed_dim)

    def forward_combined(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified forward pass.

        Args:
            hsi: [B, C_hsi, H, W]
            aux: [B, C_aux, H, W]

        Returns:
            total_loss: scalar reconstruction loss over masked entries
            info: dict with 'band_recon' (= total_loss), 'pred', 'target', 'mask'
        """
        B, _, H, W = hsi.shape
        N = self.num_tokens
        assert H * W == N, f"Patch size mismatch: H*W={H*W}, num_tokens={N}"

        # 1. Concatenate HSI + aux bands (HSI-only when use_aux is False).
        if self.use_aux:
            combined = torch.cat([hsi, aux], dim=1)             # [B, C_total, H, W]
        else:
            combined = hsi                                      # [B, C_hsi, H, W]

        # 2. Per-(pixel, band) masking (if enabled).
        if self.use_band_mask:
            combined_masked, band_mask = self.band_masking(combined)  # [B, C_total, H, W]
        else:
            combined_masked = combined
            band_mask = None

        # 3. Split back and tokenize through the encoder (which re-concatenates).
        if self.use_aux:
            hsi_masked = combined_masked[:, :self.hsi_channels]
            aux_masked = combined_masked[:, self.hsi_channels:]
            patch_tokens = self.encoder.tokenize(hsi_masked, aux_masked)  # [B, N, D]
        else:
            patch_tokens = self.encoder.tokenize(combined_masked)         # [B, N, D]

        # 4. Spatial token masking (if enabled).
        if self.use_spatial_mask:
            patch_tokens, token_mask = self.spatial_masking(patch_tokens)  # token_mask [B, N]
        else:
            token_mask = None

        # 5. Build [CLS, patch_tokens] and encode.
        cls_token = self.encoder.class_agnostic_emb.expand(B, -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)    # [B, 1+N, D]
        tokens = tokens + self.encoder.pos_embed
        encoded = self.encoder.encoder(tokens)
        encoded = self.encoder.norm(encoded)

        patch_encoded = encoded[:, 1:]                          # [B, N, D]

        # Run through the projection head so its gradients are shaped by the
        # reconstruction objective, matching the previous behavior.
        patch_encoded = self.encoder.projection(patch_encoded)
        patch_encoded = self.enc_to_dec(patch_encoded)

        # 6. Decode.
        pred = self.decoder(patch_encoded)                      # [B, N, C_total]

        # 7. Target and per-entry mask (union of band + spatial).
        target = combined.flatten(2).transpose(1, 2)            # [B, N, C_total]

        if band_mask is not None:
            mask_per_entry = band_mask.flatten(2).transpose(1, 2)  # [B, N, C_total]
        else:
            mask_per_entry = torch.zeros_like(target)

        if token_mask is not None:
            spatial_entry = token_mask.unsqueeze(-1).expand(-1, -1, self.total_channels)
            mask_per_entry = torch.maximum(mask_per_entry, spatial_entry)

        # 8. Reconstruction loss on masked entries only (per-element error,
        #    then optional center-weighting, then masked mean).
        if self.recon_loss == "mse":
            sq = (pred - target) ** 2                           # [B, N, C_total]
        elif self.recon_loss == "l1":
            sq = (pred - target).abs()
        else:  # smooth_l1 / huber (beta=1.0)
            sq = nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
        if self.recon_sigma is not None:
            sq = sq * self._recon_center_weights.view(1, N, 1)
        denom = mask_per_entry.sum().clamp_min(1.0)
        loss = (sq * mask_per_entry).sum() / denom

        info = {
            "band_recon": loss,
            "pred": pred.detach(),
            "target": target.detach(),
            "mask": mask_per_entry.detach(),
        }
        return loss, info

    def forward(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.forward_combined(hsi, aux)

    def get_encoder(self) -> nn.Module:
        return self.encoder

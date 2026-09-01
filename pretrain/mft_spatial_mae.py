"""
"Spatial" masked-modeling pretraining for the original-MFT baseline
(:class:`models.mft_original.MFTOriginal`).

This replicates the masking + loss of CoFFE's SimMIM-token regime
— the SimMIM objective with ``band_mask_ratio=0.0, spatial_mask_ratio=0.75,
recon_center_sigma=1.0`` (see
:class:`pretrain.simmim.SimMIMPretrainModel`):

* SimMIM-style *in-place* spatial token masking
  (:class:`pretrain.masked_modeling.SpatialTokenMasking`): 75% of the 121 pixel
  tokens are replaced with a learnable mask token; the encoder still sees all N
  tokens.
* A lightweight 2-layer :class:`pretrain.masked_modeling.MLPDecoder` reconstructs
  the full per-pixel HSI+LiDAR band vector.
* Center-weighted MSE (``recon_center_sigma``) computed **only on masked entries**.

It differs from ``SimMIMPretrainModel`` only where the original
MFT architecture demands it:

1. **Data-dependent CLS.** The enhanced wrapper prepends a learnable
   ``class_agnostic_emb``; the MFT CLS is instead built from the auxiliary
   modality via ``encoder.make_cls(aux)`` (the external LiDAR CLS — the essence
   of MFT's multimodal fusion). There is also no projection head (the original
   MFT has none), so the ``encoder.projection`` step is dropped.

2. **CLS injection before the decoder.** mCrossPA updates only the CLS token and
   the MLP decoder reconstructs from the *patch* tokens (CLS dropped), so a pure
   mCrossPA encoder would receive no reconstruction gradient and the per-pixel
   LiDAR bands could not be reconstructed. The encoded CLS is therefore
   broadcast-added into every encoded patch token before the decoder — routing a
   gradient into the mCrossPA projections and making the LiDAR signal available
   for reconstruction. This mirrors the cross-attention adaptation used by the
   original-MFT standard-MAE wrapper (:mod:`pretrain.mft_mae`) and leaves the
   masking, decoder type, loss, and target identical to the Spatial version.

Only spatial masking is supported (band masking is ill-defined for the original
MFT, where the auxiliary modality enters only through the CLS token).

After pretraining the decoder / masking modules are discarded; eval rebuilds the
bare ``MFTOriginal`` and ``fix_state_dict_keys`` strips the ``encoder.``
prefix and skips ``spatial_masking`` / ``decoder``.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .masked_modeling import SpatialTokenMasking, MLPDecoder
from utils.spatial_weights import make_center_weights


class MFTSpatialMaskPretrainModel(nn.Module):
    """
    Spatial-mask (SimMIM-style) pretraining wrapper around an ``MFTOriginal``.

    Args:
        encoder: The MFT encoder. Must expose ``tokenize(hsi)``, ``make_cls(aux)``,
            ``encoder``, ``norm`` and ``pos_embed`` (an ``MFTOriginal`` does).
        hsi_channels: Number of HSI spectral bands.
        aux_channels: Number of auxiliary channels (e.g. 1 for LiDAR).
        use_aux: Must be True (MFT reconstructs HSI+aux jointly and needs aux for
            the CLS token).
        patch_size: Spatial patch size (N = patch_size ** 2 tokens).
        embed_dim: Encoder embedding dimension.
        decoder_hidden_dim: Hidden width of the MLP decoder.
        spatial_mask_ratio: Fraction of whole pixel tokens replaced with a
            learnable mask token (0.75 in the Spatial version).
        recon_sigma: Gaussian sigma for the center-weighted reconstruction loss
            (1.0 in the Spatial version). None = uniform weighting.
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
        spatial_mask_ratio: float = 0.75,
        recon_sigma: Optional[float] = 1.0,
    ):
        super().__init__()

        if not use_aux:
            raise ValueError(
                "MFTSpatialMaskPretrainModel requires use_aux=True (the MFT CLS "
                "token is derived from the auxiliary modality)."
            )
        if not (0.0 < spatial_mask_ratio < 1.0):
            raise ValueError(
                f"spatial_mask_ratio must be in (0, 1), got {spatial_mask_ratio}"
            )

        self.encoder = encoder
        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels
        self.use_aux = use_aux
        self.total_channels = hsi_channels + aux_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_tokens = patch_size * patch_size
        self.spatial_mask_ratio = spatial_mask_ratio
        self.recon_sigma = recon_sigma

        if recon_sigma is not None:
            rw = make_center_weights(patch_size, recon_sigma, normalize=False)
            rw = rw / rw.mean()  # mean weight 1.0 so the loss scale is preserved
            self.register_buffer("_recon_center_weights", rw)

        # Same masking + decoder building blocks as CoFFE's SimMIM-token recipe.
        self.spatial_masking = SpatialTokenMasking(
            embed_dim=embed_dim,
            mask_ratio=spatial_mask_ratio,
        )
        self.decoder = MLPDecoder(
            embed_dim=embed_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=self.total_channels,
        )

    def forward(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hsi: [B, C_hsi, H, W]
            aux: [B, C_aux, H, W]

        Returns:
            loss: scalar center-weighted MSE over masked entries.
            info: dict with 'band_recon' (= loss), 'pred', 'target', 'mask'.
        """
        B, _, H, W = hsi.shape
        N = self.num_tokens
        assert H * W == N, f"Patch size mismatch: H*W={H * W}, num_tokens={N}"

        # 1. Reconstruction target: full HSI + aux per pixel, original token order.
        combined = torch.cat([hsi, aux], dim=1)             # [B, C_total, H, W]
        target = combined.flatten(2).transpose(1, 2)        # [B, N, C_total]

        # 2. Tokenize HSI -> 121 spatial tokens, then in-place spatial masking.
        patch_tokens = self.encoder.tokenize(hsi)           # [B, N, D]
        patch_tokens, token_mask = self.spatial_masking(patch_tokens)  # token_mask [B, N]

        # 3. Prepend the data-dependent external LiDAR CLS, add pos embed, encode.
        cls = self.encoder.make_cls(aux)                    # [B, 1, D]
        tokens = torch.cat([cls, patch_tokens], dim=1)      # [B, 1+N, D]
        tokens = tokens + self.encoder.pos_embed
        encoded = self.encoder.encoder(tokens)
        encoded = self.encoder.norm(encoded)

        # 4. CLS injection: add the encoded CLS into every encoded patch token so
        #    the mCrossPA projections get a reconstruction gradient and the LiDAR
        #    signal is available for reconstruction (see module docstring).
        patch_enc = encoded[:, 1:] + encoded[:, :1]         # [B, N, D]

        # 5. Decode to per-pixel band predictions.
        pred = self.decoder(patch_enc)                      # [B, N, C_total]

        # 6. Center-weighted MSE on masked entries only (union over all bands of a
        #    masked pixel token), matching the Spatial version's loss.
        mask_per_entry = token_mask.unsqueeze(-1).expand(-1, -1, self.total_channels)
        sq = (pred - target) ** 2                           # [B, N, C_total]
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

    def get_encoder(self) -> nn.Module:
        return self.encoder

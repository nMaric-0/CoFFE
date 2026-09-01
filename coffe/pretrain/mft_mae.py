"""
Standard-MAE pretraining for the original-MFT baseline (:class:`models.mft_original.MFTOriginal`).

This is the He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
recipe — token removal (75%), encode the visible subset only, an asymmetric
transformer decoder re-inserts learnable mask tokens and reconstructs the
per-pixel band values, MSE on masked tokens only with ``norm_pix_loss`` — applied
to the 121 spatial HSI tokens of the faithful MFT encoder. It mirrors
:class:`pretrain.mae_pretrain.MAEPretrainModel` (which targets the unified
``CoFFE``) and shares its decoder (:mod:`pretrain.decoders`) and masking
convention, with two MFT-specific changes:

1. **Data-dependent CLS.** ``MAEPretrainModel`` prepends a learnable
   ``class_agnostic_emb``; the MFT CLS token is instead computed from the
   auxiliary modality via ``encoder.make_cls(aux)`` (the external LiDAR CLS).

2. **Decoder cross-attends to the encoded CLS.** mCrossPA updates only the CLS
   token and leaves the HSI tokens unchanged, and the canonical MAE decoder drops
   the CLS before decoding — so a pure mCrossPA encoder would receive *no*
   reconstruction gradient (only the conv front-end would train). To make the
   fusion transformer actually learn, the decoder is built with
   ``use_cross_attention=True`` and the encoded sequence (which carries the
   trained CLS) is passed as the cross-attention context. This is a small,
   deliberate deviation from vanilla MAE required by the mCrossPA architecture.

After pretraining the decoder / ``enc_to_dec`` are discarded; eval rebuilds the
bare ``MFTOriginal`` and the evaluator's ``fix_state_dict_keys`` strips the
``encoder.`` prefix and skips ``decoder`` / ``enc_to_dec`` (same as the
``CoFFE`` MAE path).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .decoders import TransformerDecoder
from coffe.utils.spatial_weights import make_center_weights


class MFTMAEPretrainModel(nn.Module):
    """
    Masked-autoencoder pretraining wrapper around an :class:`MFTOriginal`.

    Args:
        encoder: The MFT encoder to pretrain. Must expose ``tokenize(hsi)``,
            ``make_cls(aux)``, ``encoder``, ``norm`` and ``pos_embed``
            (an ``MFTOriginal`` does).
        hsi_channels: Number of HSI spectral bands.
        aux_channels: Number of auxiliary channels (e.g. 1 for LiDAR).
        use_aux: Must be True (MFT reconstructs HSI+aux jointly and needs aux for
            the CLS token).
        patch_size: Spatial patch size (N = patch_size ** 2 tokens).
        embed_dim: Encoder embedding dimension.
        mask_ratio: Fraction of spatial tokens to remove before encoding (0.75).
        decoder_dim: Transformer decoder width (asymmetric: narrower than encoder).
        decoder_depth: Number of transformer decoder layers.
        decoder_heads: Number of decoder attention heads (must divide decoder_dim).
        decoder_mlp_ratio: Decoder MLP expansion ratio.
        decoder_dropout: Decoder dropout.
        norm_pix_loss: If True, normalize each token's channel vector before MSE.
        recon_sigma: Optional Gaussian sigma for center-weighted reconstruction
            loss. None = uniform weighting (the plain MAE recipe).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hsi_channels: int,
        aux_channels: int = 1,
        use_aux: bool = True,
        patch_size: int = 11,
        embed_dim: int = 128,
        mask_ratio: float = 0.75,
        decoder_dim: int = 64,
        decoder_depth: int = 4,
        decoder_heads: int = 4,
        decoder_mlp_ratio: float = 4.0,
        decoder_dropout: float = 0.0,
        norm_pix_loss: bool = True,
        recon_sigma: Optional[float] = None,
    ):
        super().__init__()

        if not use_aux:
            raise ValueError(
                "MFTMAEPretrainModel requires use_aux=True (the MFT CLS token is "
                "derived from the auxiliary modality)."
            )
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if decoder_dim % decoder_heads != 0:
            raise ValueError(
                f"decoder_dim ({decoder_dim}) must be divisible by "
                f"decoder_heads ({decoder_heads})"
            )

        self.encoder = encoder
        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels
        self.use_aux = use_aux
        self.total_channels = hsi_channels + aux_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_tokens = patch_size * patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.recon_sigma = recon_sigma

        self.len_keep = int(round(self.num_tokens * (1.0 - mask_ratio)))
        if not (1 <= self.len_keep < self.num_tokens):
            raise ValueError(
                f"mask_ratio={mask_ratio} leaves {self.len_keep} visible tokens "
                f"out of {self.num_tokens}; need 1 <= len_keep < N"
            )

        if recon_sigma is not None:
            rw = make_center_weights(patch_size, recon_sigma, normalize=False)
            rw = rw / rw.mean()
            self.register_buffer("_recon_center_weights", rw)

        # Encoder dim -> decoder dim. Named ``enc_to_dec`` so the evaluator's
        # fix_state_dict_keys skip-list drops it at load time.
        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim)

        # Canonical MAE transformer decoder + cross-attention to the encoded CLS
        # (see module docstring, deviation #2). Named ``decoder`` so it is skipped
        # at eval.
        self.decoder = TransformerDecoder(
            embed_dim=decoder_dim,
            output_dim=self.total_channels,
            num_tokens=self.num_tokens,
            num_layers=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            dropout=decoder_dropout,
            use_cross_attention=True,
        )

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample random masking by shuffle (He et al. 2022).

        Visible tokens occupy the FRONT of the shuffled order and
        ``ids_restore = argsort(ids_shuffle)`` is the inverse permutation — the
        exact convention :meth:`TransformerDecoder.forward` assumes.

        Returns ``(x_visible [B, len_keep, D], mask [B, N] (1=masked), ids_restore [B, N])``.
        """
        B, N, D = x.shape
        len_keep = int(round(N * (1.0 - mask_ratio)))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)        # ascending; front = kept
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse permutation

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

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
            loss: scalar MSE over masked tokens.
            info: dict with 'band_recon' (= loss), 'pred', 'target', 'mask'.
        """
        B, _, H, W = hsi.shape
        N = self.num_tokens
        assert H * W == N, f"Patch size mismatch: H*W={H * W}, num_tokens={N}"

        # 1. Reconstruction target (HSI + aux) in original token order.
        combined = torch.cat([hsi, aux], dim=1)             # [B, C_total, H, W]
        target = combined.flatten(2).transpose(1, 2)        # [B, N, C_total]

        # 2. Tokenize HSI into 121 spatial tokens (CLS built separately).
        patch_tokens = self.encoder.tokenize(hsi)           # [B, N, D]

        # 3. Add PATCH positional embedding (original order) before masking, so
        #    each surviving token keeps the position it has at eval. The encoder
        #    pos_embed is [1, 1+N, D]: [:, :1] = CLS, [:, 1:] = patches.
        patch_tokens = patch_tokens + self.encoder.pos_embed[:, 1:]

        # 4. Remove mask_ratio of tokens; encode visible only.
        x_visible, mask, ids_restore = self.random_masking(patch_tokens, self.mask_ratio)

        # 5. Prepend the data-dependent external CLS (+ its positional slice).
        cls = self.encoder.make_cls(aux)                    # [B, 1, D]
        cls = cls + self.encoder.pos_embed[:, :1]
        x = torch.cat([cls, x_visible], dim=1)              # [B, 1+len_keep, D]

        # 6. Encode the visible-only sequence (mCrossPA updates only the CLS).
        x = self.encoder.encoder(x)
        x = self.encoder.norm(x)

        # 7. Project the full encoded sequence to decoder width. The CLS carries
        #    the fused signal; passing it as cross-attention context is what gives
        #    the mCrossPA transformer a reconstruction gradient (deviation #2).
        context = self.enc_to_dec(x)                        # [B, 1+len_keep, decoder_dim]
        visible_dec = context[:, 1:]                        # [B, len_keep, decoder_dim]

        # 8. Decode to all-token predictions in original order, cross-attending to
        #    the encoded sequence (incl. the trained CLS).
        pred = self.decoder(visible_dec, ids_restore, encoder_output=context)  # [B, N, C_total]

        # 9. MSE loss on masked tokens only.
        if self.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target_used = (target - mu) / torch.sqrt(var + 1e-6)
        else:
            target_used = target

        per_token = ((pred - target_used) ** 2).mean(dim=-1)   # [B, N]
        if self.recon_sigma is not None:
            per_token = per_token * self._recon_center_weights.view(1, N)
        denom = mask.sum().clamp_min(1.0)
        loss = (per_token * mask).sum() / denom

        info = {
            "band_recon": loss,
            "pred": pred.detach(),
            "target": target.detach(),
            "mask": mask.detach(),
        }
        return loss, info

    def get_encoder(self) -> nn.Module:
        return self.encoder

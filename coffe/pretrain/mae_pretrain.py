"""
Token-drop MAE pretraining for the CoFFE encoder.

This is the He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
recipe, adapted to the unified HSI+LiDAR token space:

* Spatial masking only. A random fraction (``mask_ratio``, 0.75 in the paper) of
  whole pixel tokens are *removed* — the encoder sees only the visible 25% of
  tokens (plus the CLS / class-agnostic token), not a masked-in-place sequence.
* A lightweight, asymmetric *transformer* decoder re-inserts learnable mask
  tokens at the removed positions, adds its own positional embeddings, runs a
  few transformer blocks, and reconstructs the per-pixel band values.
* No MLP projection head: the encoder is built with ``use_projection=False`` and
  the transformer decoder replaces it. The reconstruction loss is MSE over the
  *masked* tokens only (optionally with ``norm_pix_loss`` per-token target
  normalization, the paper's default).

Contrast with :class:`pretrain.simmim.SimMIMPretrainModel`,
which keeps all tokens in the encoder (SimMIM-style in-place masking) and uses an
MLP decoder. That path is left untouched; this is a separate ``objective="mae"``
variant selected in ``scripts/pretrain.py``.

The transformer decoder is reused from :mod:`pretrain.decoders` (its
``forward(visible_tokens, ids_restore)`` already implements the canonical MAE
unshuffle: concat ``[visible, mask_tokens]`` -> ``gather(ids_restore)`` -> add
decoder ``pos_embed`` -> transformer layers -> ``output_proj``). ``random_masking``
below produces exactly the convention that decode expects (visible tokens at the
front of the shuffled order, ``ids_restore = argsort(ids_shuffle)``).
"""

import torch
import torch.nn as nn

from coffe.utils.spatial_weights import make_center_weights

from .decoders import TransformerDecoder


class MAEPretrainModel(nn.Module):
    """
    Masked-autoencoder pretraining wrapper around a ``CoFFE`` encoder.

    The encoder must be built with ``use_projection=False`` (the MAE recipe has
    no projection head). It must expose ``tokenize``, ``encoder``, ``norm``,
    ``class_agnostic_emb`` and ``pos_embed`` (a ``CoFFE`` does).

    Args:
        encoder: The CoFFE encoder to pretrain.
        hsi_channels: Number of HSI spectral bands.
        aux_channels: Number of auxiliary channels (e.g. 1 for LiDAR).
        use_aux: If True, aux bands are concatenated with HSI and reconstructed
                 jointly. If False, the model is HSI-only.
        patch_size: Spatial patch size (N = patch_size ** 2 tokens).
        embed_dim: Encoder embedding dimension.
        mask_ratio: Fraction of pixel tokens to remove before encoding (0.75).
        decoder_dim: Transformer decoder width (asymmetric: narrower than encoder).
        decoder_depth: Number of transformer decoder layers.
        decoder_heads: Number of decoder attention heads (must divide decoder_dim).
        decoder_mlp_ratio: Decoder MLP expansion ratio.
        decoder_dropout: Decoder dropout.
        norm_pix_loss: If True, normalize each token's channel vector (mean/var
                       across bands) before the MSE loss (MAE default).
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
        recon_sigma: float | None = None,
    ):
        super().__init__()

        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if decoder_dim % decoder_heads != 0:
            raise ValueError(
                f"decoder_dim ({decoder_dim}) must be divisible by decoder_heads ({decoder_heads})"
            )

        self.encoder = encoder
        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels
        self.use_aux = use_aux
        self.total_channels = hsi_channels + aux_channels if use_aux else hsi_channels
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

        # Optional center-weighted reconstruction loss (mean weight 1.0 so the
        # loss scale is preserved). Disabled by default for the plain MAE recipe.
        if recon_sigma is not None:
            rw = make_center_weights(patch_size, recon_sigma, normalize=False)
            rw = rw / rw.mean()
            self.register_buffer("_recon_center_weights", rw)

        # Encoder dim -> decoder dim. Named ``enc_to_dec`` so the evaluator's
        # fix_state_dict_keys skip-list drops it at load time.
        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim)

        # Canonical MAE transformer decoder. Named ``decoder`` so it is skipped
        # at eval. num_tokens = N (the CLS token is dropped before decoding); the
        # decoder owns its own mask_token + pos_embed, separate from the encoder.
        self.decoder = TransformerDecoder(
            embed_dim=decoder_dim,
            output_dim=self.total_channels,
            num_tokens=self.num_tokens,
            num_layers=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            dropout=decoder_dropout,
            use_cross_attention=False,
        )

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-sample random masking by shuffle (He et al. 2022).

        Visible tokens occupy the FRONT of the shuffled order, and
        ``ids_restore = argsort(ids_shuffle)`` is the inverse permutation — the
        exact convention :class:`TransformerDecoder.forward` assumes when it
        concatenates ``[visible, mask_tokens]`` and gathers by ``ids_restore``.

        Args:
            x: [B, N, D] patch tokens.
            mask_ratio: fraction of tokens to remove.

        Returns:
            x_visible: [B, len_keep, D] kept tokens.
            mask: [B, N] in original token order (1 = masked/removed, 0 = kept).
            ids_restore: [B, N] indices that restore original order.
        """
        B, N, D = x.shape
        len_keep = int(round(N * (1.0 - mask_ratio)))

        # Explicit fp32 noise (independent of x's dtype under AMP); driven by the
        # global RNG seeded via utils.seed.set_seed.
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascending; front = kept
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse permutation

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # mask: 1 = removed, 0 = kept, restored to original token order.
        mask = torch.ones(B, N, device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_visible, mask, ids_restore

    def forward(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        MAE forward pass.

        Args:
            hsi: [B, C_hsi, H, W]
            aux: [B, C_aux, H, W] (ignored when use_aux is False)

        Returns:
            loss: scalar MSE over masked tokens.
            info: dict with 'band_recon' (= loss), 'pred', 'target', 'mask'.
        """
        B, _, H, W = hsi.shape
        N = self.num_tokens
        assert H * W == N, f"Patch size mismatch: H*W={H * W}, num_tokens={N}"

        # 1. Reconstruction target in original token order: [B, N, C_total].
        # Kept as if/else (not a ternary) so both branch shapes stay documented.
        if self.use_aux:  # noqa: SIM108
            combined = torch.cat([hsi, aux], dim=1)  # [B, C_total, H, W]
        else:
            combined = hsi  # [B, C_hsi, H, W]
        target = combined.flatten(2).transpose(1, 2)  # [B, N, C_total]

        # 2. Tokenize (no CLS / pos yet). Encoder re-concatenates aux internally.
        if self.use_aux:
            patch_tokens = self.encoder.tokenize(hsi, aux)  # [B, N, D]
        else:
            patch_tokens = self.encoder.tokenize(hsi)  # [B, N, D]

        # 3. Add PATCH positional embedding (original order) BEFORE masking, so
        #    each surviving token keeps the position it would have at eval. The
        #    encoder's pos_embed is [1, 1+N, D]: [:, :1] = CLS, [:, 1:] = patches.
        patch_tokens = patch_tokens + self.encoder.pos_embed[:, 1:]

        # 4. Remove mask_ratio of tokens; encode visible only.
        x_visible, mask, ids_restore = self.random_masking(patch_tokens, self.mask_ratio)

        # 5. Prepend CLS (+ its positional slice).
        cls = self.encoder.class_agnostic_emb.expand(B, -1, -1)
        cls = cls + self.encoder.pos_embed[:, :1]
        x = torch.cat([cls, x_visible], dim=1)  # [B, 1+len_keep, D]

        # 6. Encode the visible-only sequence.
        x = self.encoder.encoder(x)
        x = self.encoder.norm(x)

        # 7. Drop CLS, project to decoder width.
        x = self.enc_to_dec(x[:, 1:])  # [B, len_keep, decoder_dim]

        # 8. Decode to all-token predictions in original order.
        pred = self.decoder(x, ids_restore)  # [B, N, C_total]

        # 9. MSE loss on masked tokens only.
        if self.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target_used = (target - mu) / torch.sqrt(var + 1e-6)
        else:
            target_used = target

        per_token = ((pred - target_used) ** 2).mean(dim=-1)  # [B, N]
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
        """Return the encoder alone — the only part kept for evaluation.

        The decoder and (where present) the projection head exist for the
        pretext task and are discarded at eval time (PAPER_CANON §2).
        """
        return self.encoder

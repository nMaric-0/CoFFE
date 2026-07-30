"""Upstream-faithful MAE adaptation wrapper for HyperSIGMA (Level-2, Houston).

Replaces the prior MFT-CPEA-style adapter. The masking now mirrors upstream
HyperSIGMA's MAE recipe — token-level masking with a learnable ``mask_token``
substituted at masked positions, default ``mask_ratio = 0.75``, and
per-patch z-scored reconstruction targets (``norm_pix_loss=True`` in
upstream parlance).

The deformable attention inside the local SpatViT_fusion_patch and
SpecViT_fusion variants requires the full token grid, so we **substitute
mask_tokens before** the encoder rather than running a visible-only forward.
The encoder still sees N=full tokens; the decoder reconstructs only the
masked-token targets (loss is zero on visible tokens by construction).

Three modes via ``adapt_mode``:

* ``spatial_only`` — train SpatViT random-init pieces (``patch_embed``,
  ``pos_embed``, deformable ``sampling_offsets``) + ``spat_mask_token`` +
  ``spat_decoder``. SpecViT and SEM are frozen.
* ``spectral_only`` — train SpecViT random-init pieces (``spat_map``,
  ``pos_embed``, ``l1``, ``sampling_offsets``) + ``spec_mask_token`` +
  ``spec_decoder``. SpatViT and SEM are frozen.
* ``joint_sem`` — everything above plus SEM + ``fused_decoder``. Loss is
  ``L_spat + L_spec + L_fused`` (uniform weights).

Hooks are attached on ``dual.spat.model.patch_embed`` and
``dual.spec.model.spat_map`` (the modules whose outputs carry the 768-d
token embedding at the position where mask substitution belongs). Masks
are sampled inside ``forward`` once per call, stashed briefly so the hooks
can read them, then cleared in a ``finally``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hypersigma.hypersigma_dual import HyperSIGMADual

logger = logging.getLogger(__name__)


_SUPPORTED_MODES = ("spatial_only", "spectral_only", "joint_sem", "sem_only")
_EPS = 1e-6


def _per_token_zscore(t: torch.Tensor) -> torch.Tensor:
    """Per-row z-score: ``(t - mean) / sqrt(var + eps)`` over the last axis."""
    mu = t.mean(dim=-1, keepdim=True)
    var = t.var(dim=-1, keepdim=True, unbiased=False)
    return (t - mu) / torch.sqrt(var + _EPS)


def _masked_token_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE averaged only over masked tokens.

    Args:
        pred, target: ``[B, N, D]``.
        mask: ``[B, N]``, 1 = masked (contributes), 0 = visible (ignored).
    """
    se = (pred - target).pow(2).mean(dim=-1)            # [B, N]
    denom = mask.sum().clamp(min=1.0)
    return (se * mask).sum() / denom


def _build_per_token_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class HyperSIGMAMaskedAdaptation(nn.Module):
    """Upstream-faithful HyperSIGMA MAE adapter with three modes."""

    SUPPORTED_MODES = _SUPPORTED_MODES

    def __init__(
        self,
        dual: HyperSIGMADual,
        adapt_mode: str = "joint_sem",
        hsi_channels: int = 144,
        patch_size: int = 11,
        mask_ratio: float = 0.75,
        spat_decoder_hidden: int = 256,
        spec_decoder_hidden: int = 256,
        fused_decoder_hidden: int = 512,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if adapt_mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"adapt_mode must be one of {_SUPPORTED_MODES}, got {adapt_mode!r}"
            )
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        self.dual = dual
        self.adapt_mode = adapt_mode
        self.hsi_channels = hsi_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # `use_*` gate the per-branch reconstruction decoders + which branch
        # weights stay trainable. `sem_only` (native-geometry SEM tuning) trains
        # ONLY the fusion path: it builds no per-branch decoder (use_spat/use_spec
        # False) but still masks both branch inputs (mask_spat/mask_spec) and keeps
        # the SEM + fused decoder (use_sem True).
        self.use_spat = adapt_mode in ("spatial_only", "joint_sem")
        self.use_spec = adapt_mode in ("spectral_only", "joint_sem")
        self.use_sem = adapt_mode in ("joint_sem", "sem_only")
        self.sem_only = adapt_mode == "sem_only"
        # Which branch inputs get token-masked (drives the hooks + mask sampling).
        self.mask_spat = self.use_spat or self.sem_only
        self.mask_spec = self.use_spec or self.sem_only

        embed_dim = dual.embed_dim

        # --- Spatial side -------------------------------------------------
        if self.use_spat:
            # Read the actual patch grid from patch_embed so this is correct for
            # both the adapted geometry (12/3 -> 16) and native (64/8 -> 64).
            self.num_spat_tokens = dual.spat.model.patch_embed.num_patches
            self.spat_in_chans = dual.spat.in_chans
            self.spat_patch_k = dual.spat.new_patch_size
            self.spat_token_pixels = self.spat_in_chans * self.spat_patch_k * self.spat_patch_k

            self.spat_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.spat_mask_token, std=0.02)

            self.spat_decoder = _build_per_token_mlp(
                in_dim=embed_dim,
                hidden_dim=spat_decoder_hidden,
                out_dim=self.spat_token_pixels,
                dropout=decoder_dropout,
            )
        else:
            # Spatial branch frozen for non-spatial modes.
            for p in dual.spat.parameters():
                p.requires_grad_(False)
            if hasattr(dual, "pca_standardize"):
                for p in dual.pca_standardize.parameters():
                    p.requires_grad_(False)

        # --- Spectral side ------------------------------------------------
        if self.use_spec:
            self.num_spec_tokens = dual.num_tokens
            # The l1 layer in SpecViT projects 768 -> 128 (see
            # SpecViT_fusion.py: self.l1 = nn.Linear(embed_dim, 128, bias=False)).
            self.spec_l1_out = 128
            self.spec_token_pixels = patch_size * patch_size  # 121 per spectral token

            self.spec_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.spec_mask_token, std=0.02)

            self.spec_decoder = _build_per_token_mlp(
                in_dim=self.spec_l1_out,
                hidden_dim=spec_decoder_hidden,
                out_dim=self.spec_token_pixels,
                dropout=decoder_dropout,
            )
        else:
            for p in dual.spec.parameters():
                p.requires_grad_(False)

        # --- SEM + fused decoder (joint_sem only) -------------------------
        if self.use_sem:
            fused_in = dual.num_stages * dual.dr_dim  # 4 * 128 = 512
            fused_out = dual.spat.in_chans * patch_size * patch_size  # 3*11*11 = 363
            self.fused_out_chans = dual.spat.in_chans
            self.fused_decoder = nn.Sequential(
                nn.Linear(fused_in, fused_decoder_hidden),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(fused_decoder_hidden, fused_out),
            )
        else:
            for p in dual.sem.parameters():
                p.requires_grad_(False)

        # --- sem_only: mask both inputs, but only the fusion path trains -------
        # Both branches were frozen above (use_spat/use_spec False). We still need
        # the mask tokens + token grids to mask the (frozen) branch inputs, and we
        # unfreeze the spectral l1 projection (768->128, random — absent from the
        # checkpoint) since the SEM consumes it via `spec_first`. Net trainable:
        # SEM + fused_decoder + l1 + the two mask tokens.
        if self.sem_only:
            self.num_spat_tokens = dual.spat.model.patch_embed.num_patches
            self.num_spec_tokens = dual.num_tokens
            self.spat_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.spat_mask_token, std=0.02)
            self.spec_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.spec_mask_token, std=0.02)
            for p in dual.spec.model.l1.parameters():
                p.requires_grad_(True)

        # Mask cache slots, populated inside forward() so hooks can read them.
        self._pending_spat_mask: Optional[torch.Tensor] = None
        self._pending_spec_mask: Optional[torch.Tensor] = None

        # Attach forward hooks for mask-token substitution.
        self._hook_handles: List = []
        if self.mask_spat:
            self._hook_handles.append(
                dual.spat.model.patch_embed.register_forward_hook(self._spat_patch_embed_hook)
            )
        if self.mask_spec:
            self._hook_handles.append(
                dual.spec.model.spat_map.register_forward_hook(self._spec_spat_map_hook)
            )

    # ------------------------------------------------------------------
    # Hook cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    def __del__(self):
        # Best-effort cleanup; module deletion can race during interpreter
        # shutdown so we swallow any exceptions.
        try:
            self.remove_hooks()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _spat_patch_embed_hook(self, module, inputs, output):
        """Substitute spat_mask_token at masked positions of SpatViT patch_embed output."""
        if self._pending_spat_mask is None:
            return output
        tokens, hw = output  # tokens: [B, N, D]
        mask = self._pending_spat_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        tokens = tokens * (1.0 - mask) + self.spat_mask_token.to(tokens.dtype) * mask
        return tokens, hw

    def _spec_spat_map_hook(self, module, inputs, output):
        """Substitute spec_mask_token at masked positions of SpecViT spat_map output."""
        if self._pending_spec_mask is None:
            return output
        tokens = output  # [B, N, D]
        mask = self._pending_spec_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        tokens = tokens * (1.0 - mask) + self.spec_mask_token.to(tokens.dtype) * mask
        return tokens

    # ------------------------------------------------------------------
    # Mask sampling and target computation
    # ------------------------------------------------------------------

    def _sample_mask(self, batch_size: int, num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        noise = torch.rand(batch_size, num_tokens, device=device)
        return (noise < self.mask_ratio).to(dtype)

    def _spat_target(self, hsi: torch.Tensor) -> torch.Tensor:
        """Per-token z-scored 3x3x3 = 27-pixel target for SpatViT.

        Reconstructs the same input view SpatViT actually sees: PCA(144 -> 3),
        then ``pca_standardize``, then reflect-pad to 12x12. Each of the 16
        non-overlapping 3x3 patches becomes one row of length 27.
        """
        with torch.no_grad():
            x = self.dual.pca_spat(hsi)
            x = self.dual.pca_standardize(x)
            pad_to = self.dual.spat.pad_to
            pad_right = pad_to - x.shape[-1]
            pad_bottom = pad_to - x.shape[-2]
            if pad_right or pad_bottom:
                x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="reflect")
            k = self.dual.spat.new_patch_size
            tgt = F.unfold(x, kernel_size=k, stride=k).transpose(1, 2)  # [B, N, k*k*C]
            return _per_token_zscore(tgt)

    def _spec_target(self, hsi: torch.Tensor) -> torch.Tensor:
        """Per-token z-scored 121-pixel target for SpecViT.

        Replays the stateless ``patch_embed -> spec_embed -> transpose`` chain
        of the upstream SpectralVisionTransformer so each of the 100 spectral
        tokens carries a 121-dim spatial signature ready for MSE.
        """
        with torch.no_grad():
            spec_model = self.dual.spec.model
            x = spec_model.patch_embed(hsi)            # [B, 121, 144]
            x = spec_model.spec_embed(x)               # [B, 121, 100]
            x = x.transpose(1, 2)                      # [B, 100, 121]
            return _per_token_zscore(x)

    def _fused_target(self, hsi: torch.Tensor) -> torch.Tensor:
        """Per-channel z-scored 11x11 PCA-standardized cube for the fused decoder."""
        with torch.no_grad():
            x = self.dual.pca_spat(hsi)
            x = self.dual.pca_standardize(x)           # [B, 3, 11, 11]
            mu = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            return (x - mu) / torch.sqrt(var + _EPS)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hsi: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = hsi.shape
        assert C == self.hsi_channels and H == W == self.patch_size, (
            f"expected [B, {self.hsi_channels}, {self.patch_size}, {self.patch_size}], got {tuple(hsi.shape)}"
        )

        device = hsi.device
        dtype = hsi.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        loss_spat = zero
        loss_spec = zero
        loss_fused = zero
        out: Dict[str, torch.Tensor] = {}

        if self.mask_spat:
            self._pending_spat_mask = self._sample_mask(B, self.num_spat_tokens, device, dtype)
        if self.mask_spec:
            self._pending_spec_mask = self._sample_mask(B, self.num_spec_tokens, device, dtype)

        try:
            if self.adapt_mode == "sem_only":
                # Encoder frozen; mask both inputs, fuse, reconstruct the original
                # 11x11 PCA cube. Trains SEM + fused_decoder + l1 + mask tokens.
                dual_out = self.dual(hsi)
                pred_fused_flat = self.fused_decoder(dual_out["fused"])
                pred_fused = pred_fused_flat.view(
                    B, self.fused_out_chans, self.patch_size, self.patch_size
                )
                tgt_fused = self._fused_target(hsi)
                loss_fused = F.mse_loss(pred_fused, tgt_fused)
                out.update(pred_fused=pred_fused, target_fused=tgt_fused)

            elif self.adapt_mode == "spatial_only":
                x = self.dual.pca_spat(hsi)
                x = self.dual.pca_standardize(x)
                spat_features = self.dual.spat(x)
                f = spat_features[-1]                                   # [B, 768, 4, 4]
                tokens = f.flatten(2).transpose(1, 2)                   # [B, 16, 768]
                pred_spat = self.spat_decoder(tokens)                   # [B, 16, 27]
                tgt_spat = self._spat_target(hsi)                       # [B, 16, 27]
                loss_spat = _masked_token_mse(pred_spat, tgt_spat, self._pending_spat_mask)
                out.update(
                    pred_spat=pred_spat, target_spat=tgt_spat, mask_spat=self._pending_spat_mask
                )

            elif self.adapt_mode == "spectral_only":
                spec_features = self.dual.spec(hsi)
                f = spec_features[0]                                    # [B, 100, 128] post-l1
                pred_spec = self.spec_decoder(f)                        # [B, 100, 121]
                tgt_spec = self._spec_target(hsi)
                loss_spec = _masked_token_mse(pred_spec, tgt_spec, self._pending_spec_mask)
                out.update(
                    pred_spec=pred_spec, target_spec=tgt_spec, mask_spec=self._pending_spec_mask
                )

            else:  # joint_sem
                dual_out = self.dual(hsi)
                # Spatial
                f_spat = dual_out["spat_features"][-1]
                tokens_spat = f_spat.flatten(2).transpose(1, 2)
                pred_spat = self.spat_decoder(tokens_spat)
                tgt_spat = self._spat_target(hsi)
                loss_spat = _masked_token_mse(pred_spat, tgt_spat, self._pending_spat_mask)

                # Spectral
                f_spec = dual_out["spec_first"]                         # [B, 100, 128]
                pred_spec = self.spec_decoder(f_spec)
                tgt_spec = self._spec_target(hsi)
                loss_spec = _masked_token_mse(pred_spec, tgt_spec, self._pending_spec_mask)

                # Fused
                pred_fused_flat = self.fused_decoder(dual_out["fused"]) # [B, 3*11*11]
                pred_fused = pred_fused_flat.view(B, self.fused_out_chans, self.patch_size, self.patch_size)
                tgt_fused = self._fused_target(hsi)
                loss_fused = F.mse_loss(pred_fused, tgt_fused)

                out.update(
                    pred_spat=pred_spat, target_spat=tgt_spat, mask_spat=self._pending_spat_mask,
                    pred_spec=pred_spec, target_spec=tgt_spec, mask_spec=self._pending_spec_mask,
                    pred_fused=pred_fused, target_fused=tgt_fused,
                )
        finally:
            self._pending_spat_mask = None
            self._pending_spec_mask = None

        loss = loss_spat + loss_spec + loss_fused
        out["loss"] = loss
        out["loss_spat"] = loss_spat.detach() if loss_spat.requires_grad else loss_spat
        out["loss_spec"] = loss_spec.detach() if loss_spec.requires_grad else loss_spec
        out["loss_fused"] = loss_fused.detach() if loss_fused.requires_grad else loss_fused
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def parameter_counts(self) -> Dict[str, int]:
        """Return a per-component breakdown of trainable parameters."""
        counts: Dict[str, int] = {}

        def _train(module_or_param) -> int:
            if isinstance(module_or_param, nn.Module):
                return sum(p.numel() for p in module_or_param.parameters() if p.requires_grad)
            return module_or_param.numel() if module_or_param.requires_grad else 0

        if self.use_spat:
            counts["spat_branch_trainable"] = _train(self.dual.spat)
            counts["spat_mask_token"] = _train(self.spat_mask_token)
            counts["spat_decoder"] = _train(self.spat_decoder)
        if self.use_spec:
            counts["spec_branch_trainable"] = _train(self.dual.spec)
            counts["spec_mask_token"] = _train(self.spec_mask_token)
            counts["spec_decoder"] = _train(self.spec_decoder)
        if self.use_sem:
            counts["sem_trainable"] = _train(self.dual.sem)
            counts["fused_decoder"] = _train(self.fused_decoder)

        counts["total_trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["total_frozen"] = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return counts

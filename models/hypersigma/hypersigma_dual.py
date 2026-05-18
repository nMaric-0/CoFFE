"""Combined HyperSIGMA dual-branch module.

Replicates the canonical ``ss_fusion_cls.SSFusionFramework`` pipeline:
PCA-reduced spatial input -> SpatViT branch (4 FPN feature stages) +
raw-band spectral input -> SpecViT branch (100 tokens, ``l1``-projected
first stage) -> SEM fusion -> ``[B, 512]`` feature.

The classifier head from upstream is replaced with a feature tap — we
return the 512-d SEM output (plus the per-branch features for
ablations) and let the eval wrapper handle prototype matching.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .preprocessing import PCAPreprocessor, load_pca
from .sem import SEM
from .spat_vit_branch import SpatViTBranch
from .spec_vit_branch import SpecViTBranch

logger = logging.getLogger(__name__)


class HyperSIGMADual(nn.Module):
    """Spatial+Spectral HyperSIGMA encoder with SEM fusion.

    Args:
        pca_spat_path:  Path to a pickled ``sklearn.decomposition.PCA``
                        fit to the dataset's HSI bands (144 -> 3 for
                        Houston).
        spat_ckpt:      Path to the SpatViT-B ``spat-vit-base.pth``
                        checkpoint (``None`` for random init).
        spec_ckpt:      Path to the SpecViT-B ``spec-vit-base.pth``
                        checkpoint (``None`` for random init).
        hsi_channels:   Number of input HSI bands.
        spat_patch_k:   SpatViT ``patch_size`` (3 or 1).
        freeze_body:    If True, freeze SpatViT/SpecViT transformer
                        bodies and leave only ``patch_embed``,
                        ``pos_embeds``, ``spat_map``, ``l1``, and the
                        SEM trainable.
    """

    def __init__(
        self,
        pca_spat_path: str,
        spat_ckpt: Optional[str],
        spec_ckpt: Optional[str],
        hsi_channels: int = 144,
        spat_patch_k: int = 3,
        freeze_body: bool = True,
        embed_dim: int = 768,
        num_tokens: int = 100,
        dr_dim: int = 128,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        self.hsi_channels = hsi_channels
        self.spat_patch_k = spat_patch_k
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.dr_dim = dr_dim
        self.num_stages = num_stages

        pca = load_pca(pca_spat_path)
        if pca.components_.shape[1] != hsi_channels:
            raise ValueError(
                f"PCA fit on {pca.components_.shape[1]} bands, model expects {hsi_channels}"
            )
        self.pca_spat = PCAPreprocessor(pca)

        pad_to = 12 if spat_patch_k == 3 else 11
        self.spat = SpatViTBranch(
            pretrained_path=spat_ckpt,
            in_chans=self.pca_spat.out_channels,
            new_patch_size=spat_patch_k,
            pad_to=pad_to,
            embed_dim=embed_dim,
            freeze_body=freeze_body,
        )
        self.spec = SpecViTBranch(
            pretrained_path=spec_ckpt,
            in_chans=hsi_channels,
            img_size=11,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            freeze_body=freeze_body,
        )

        self.sem = SEM(
            spat_dim=embed_dim,
            num_tokens=num_tokens,
            dr_dim=dr_dim,
            num_stages=num_stages,
        )

        # Buffer for the spec pool (parameterless) so it lives on the
        # right device.
        self.register_module("spec_pool", nn.AdaptiveAvgPool1d(1))

    # ------------------------------------------------------------------
    # Sanity / introspection helpers
    # ------------------------------------------------------------------

    def parameter_counts(self) -> Dict[str, int]:
        """Return (trainable, frozen) parameter counts for the whole module."""
        trainable = 0
        frozen = 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def log_sanity(self) -> None:
        spat = self.spat
        spec = self.spec
        sem = self.sem
        counts = self.parameter_counts()
        logger.info(
            "[HyperSIGMA] Spatial PCA cumulative explained variance: %d -> %d : %.4f",
            self.hsi_channels, self.pca_spat.out_channels,
            self.pca_spat.cumulative_explained_variance,
        )
        logger.info(
            "[HyperSIGMA] SpatViT branch (SpatViT_fusion_patch): cls_token=None, "
            "patch_embed.proj.weight.shape=%s, pos_embed source=%s, pos_embed.shape=%s, "
            "fpn(k=%d) ops=%s, feature stage HxW=%dx%d",
            tuple(spat.model.patch_embed.proj.weight.shape),
            spat.pos_embed_source,
            tuple(spat.model.pos_embed.shape),
            spat.new_patch_size,
            [type(getattr(spat.model, f"fpn{i}")).__name__ for i in range(1, 5)],
            spat.pad_to // spat.new_patch_size,
            spat.pad_to // spat.new_patch_size,
        )
        logger.info(
            "[HyperSIGMA] SpecViT branch (SpecViT_fusion): cls_token=None, "
            "spec_embed=AdaptiveAvgPool1d(out=%d), spat_map.weight.shape=%s, "
            "pos_embed source=%s, pos_embed.shape=%s, in_chans=%d (raw, no spectral PCA)",
            spec.num_tokens,
            tuple(spec.model.spat_map.weight.shape),
            spec.pos_embed_source,
            tuple(spec.model.pos_embed.shape),
            spec.in_chans,
        )
        logger.info(
            "[HyperSIGMA] SEM: %d DR convs (Conv2d(%d,%d,1) each); "
            "%d fc_spec MLPs (Linear(%d,%d)+ReLU+Linear(%d,%d)+Sigmoid); output dim=%d",
            sem.num_stages, self.embed_dim, sem.dr_dim,
            sem.num_stages, sem.num_tokens, sem.dr_dim, sem.dr_dim, sem.dr_dim,
            sem.num_stages * sem.dr_dim,
        )
        logger.info(
            "[HyperSIGMA] Parameter counts: trainable=%d, frozen=%d, total=%d",
            counts["trainable"], counts["frozen"], counts["total"],
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hsi: torch.Tensor) -> Dict[str, object]:
        spat_in = self.pca_spat(hsi)                       # [B, 3, H, W]
        spat_features = self.spat(spat_in)                 # 4 x [B, 768, Hp, Wp]
        spec_features = self.spec(hsi)                     # 4 x [B, 100, *]
        # features[0] has had `l1` applied -> [B, 100, 128]; the rest stay 768.
        spec_first = spec_features[0]                      # [B, 100, 128]
        spec_pooled = self.spec_pool(spec_first).flatten(1)  # [B, 100]
        fused = self.sem(spat_features, spec_pooled)       # [B, 512]
        return {
            "fused": fused,
            "spat_features": spat_features,
            "spec_first": spec_first,
            "spec_features": spec_features,
        }

"""Four-stage gated spatial-spectral fusion (SEM).

Faithful port of ``ss_fusion_cls.SSFusionFramework``'s fusion block,
with the downstream classifier head removed. Output is the canonical
``(B, 4 * dr_dim) = (B, 512)`` feature used for prototype evaluation
and as the MAE reconstruction source during Houston Level-2 adaptation.

All parameters here are randomly initialized (no upstream checkpoint
ships the downstream SEM weights) and trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEM(nn.Module):
    """Spatial-Spectral Enhancement Module (4 stages).

    Args:
        spat_dim:   Channel dim of each spatial feature stage (e.g. 768).
        num_tokens: Number of spectral tokens produced by SpecViT (100).
        dr_dim:     Dimension reduction target per stage (128).
        num_stages: Number of FPN stages to fuse (4 for ViT-B).
    """

    def __init__(
        self,
        spat_dim: int = 768,
        num_tokens: int = 100,
        dr_dim: int = 128,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.dr_dim = dr_dim
        self.num_tokens = num_tokens

        self.dr = nn.ModuleList(
            [nn.Conv2d(spat_dim, dr_dim, kernel_size=1, bias=False) for _ in range(num_stages)]
        )
        self.fc_spec = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_tokens, dr_dim, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(dr_dim, dr_dim, bias=False),
                    nn.Sigmoid(),
                )
                for _ in range(num_stages)
            ]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        spat_features: list[torch.Tensor],
        spec_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spat_features: list of ``num_stages`` tensors, each ``[B, spat_dim, Hp, Wp]``.
            spec_pooled:   ``[B, num_tokens]`` — global-pooled spectral feature.

        Returns:
            ``[B, num_stages * dr_dim]`` fused feature (e.g. ``[B, 512]``).
        """
        assert len(spat_features) == self.num_stages, (
            f"SEM expects {self.num_stages} stages, got {len(spat_features)}"
        )
        assert spec_pooled.shape[-1] == self.num_tokens, (
            f"SEM expects spec_pooled with {self.num_tokens} tokens, got {spec_pooled.shape[-1]}"
        )

        pooled_stages: list[torch.Tensor] = []
        for i in range(self.num_stages):
            dr_i = self.dr[i](spat_features[i])  # [B, dr_dim, Hp, Wp]
            w_i = self.fc_spec[i](spec_pooled)  # [B, dr_dim]
            fused_i = (1.0 + w_i[:, :, None, None]) * dr_i  # [B, dr_dim, Hp, Wp]
            pooled_i = F.adaptive_avg_pool2d(fused_i, 1).flatten(1)  # [B, dr_dim]
            pooled_stages.append(pooled_i)

        return torch.cat(pooled_stages, dim=1)  # [B, num_stages * dr_dim]

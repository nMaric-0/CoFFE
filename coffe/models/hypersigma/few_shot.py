"""Eval-shaped wrapper around ``HyperSIGMADual``.

Mirrors the public surface of ``CoFFE`` so the existing
``scripts/evaluate.py`` episode loop can drive it. The wrapper:

* freezes every parameter in the dual encoder at construction time and
  switches to ``eval()`` mode,
* exposes ``forward_features(hsi, aux)`` returning a ``(patch, cls,
  aux_dup)`` triple so the upstream eval loop unpacks correctly. The
  downstream ``.mean(dim=1)`` over the patch axis is a no-op because we
  pack the single feature vector as ``[B, 1, D]``,
* selects which feature to expose via ``mode``:
    - ``"fused"`` (default, headline): SEM 512-d output, L2-normalized
    - ``"spat_pool"``: adaptive_avg_pool2d the last SpatViT FPN stage to
      ``[B, 768]``, L2-normalized
    - ``"spec_pool"``: mean over the 100 SpecViT tokens of the last
      stage (post-norm, 768-d), L2-normalized,
* reuses ``compute_prototypes`` / ``_forward_mean_features`` /
  ``_forward_mean_distances`` from ``CoFFE`` verbatim (we don't
  recompute the prototype math).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypersigma_dual import HyperSIGMADual


class HyperSIGMAFewShot(nn.Module):
    """Prototype-eval wrapper over a frozen HyperSIGMADual."""

    SUPPORTED_MODES = ("fused", "spat_pool", "spec_pool")

    def __init__(
        self,
        dual: HyperSIGMADual,
        mode: str = "fused",
        distance_metric: str = "euclidean",
        temperature: float = 10.0,
        prototype_mode: str = "mean_features",
        pool_sigma: float | None = None,
    ) -> None:
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode={mode!r} not in {self.SUPPORTED_MODES}")
        if distance_metric not in ("cosine", "euclidean"):
            raise ValueError(f"distance_metric={distance_metric!r}")

        self.dual = dual
        self.mode = mode
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.prototype_mode = prototype_mode
        self.pool_sigma = pool_sigma  # unused; kept for API parity

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract(self, out: dict) -> torch.Tensor:
        if self.mode == "fused":
            return F.normalize(out["fused"], p=2, dim=-1)
        if self.mode == "spat_pool":
            last = out["spat_features"][-1]  # [B, 768, Hp, Wp]
            pooled = F.adaptive_avg_pool2d(last, 1).flatten(1)
            return F.normalize(pooled, p=2, dim=-1)
        if self.mode == "spec_pool":
            # Use the deepest spectral stage (post-norm, [B, 100, 768])
            # rather than features[0] (which is the l1-projected [B, 100, 128]).
            last = out["spec_features"][-1]
            pooled = last.mean(dim=1)  # [B, 768]
            return F.normalize(pooled, p=2, dim=-1)
        raise ValueError(self.mode)

    def forward_features(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a patch batch into the frozen eval feature.

        Returns the ``(patch_emb, cls_emb, aux_emb)`` triple that the shared
        evaluator expects; for HyperSIGMA all three carry the same L2-normalised
        vector — spatial 768-d (``spat_pool``), spectral 768-d (``spec_pool``)
        or fused SEM 512-d (``fused``), the three feature columns of
        PAPER_CANON Table 3. ``aux`` is accepted and ignored: this route is
        HSI-only.
        """
        # Skip the unused branch when the eval mode only needs one side.
        # `mode="fused"` still requires both (SEM consumes spat + spec).
        # Saves ~half the eval wallclock for `spat_pool` / `spec_pool` runs.
        if self.mode == "spat_pool":
            spat_in = self.dual.pca_spat(hsi)
            spat_in = self.dual.pca_standardize(spat_in)
            spat_features = self.dual.spat(spat_in)
            last = spat_features[-1]
            pooled = F.adaptive_avg_pool2d(last, 1).flatten(1)
            feats = F.normalize(pooled, p=2, dim=-1)
        elif self.mode == "spec_pool":
            spec_features = self.dual.spec(hsi)
            last = spec_features[-1]
            feats = F.normalize(last.mean(dim=1), p=2, dim=-1)
        else:  # "fused"
            out = self.dual(hsi)
            feats = self._extract(out)
        patch = feats.unsqueeze(1)  # [B, 1, D]
        return patch, feats, feats

    def eval_patch_embeddings(
        self,
        patch_emb: torch.Tensor,
        cls_emb: torch.Tensor,
        cls_token_weight: float | None = None,
        renormalize: bool = True,
    ) -> torch.Tensor:
        """Return ``patch_emb`` unchanged — there is no class token to fold in.

        CoFFE mixes the class-agnostic token into the patch embedding with
        weight λ; HyperSIGMA has no such token, so λ is inert on this route
        (PAPER_CANON §8 D3). The signature matches CoFFE's only so the one
        evaluator can drive both.
        """
        # No class-token folding in HyperSIGMA — patch tokens pass through.
        return patch_emb

    # ------------------------------------------------------------------
    # Prototype math (copy of CoFFE internals; no shared state)
    # ------------------------------------------------------------------

    def compute_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Class means of the support features — the NCM prototypes of §4.

        With K=5 support samples per class this is the mean of five feature
        vectors per class. Duplicated from CoFFE rather than shared, so the two
        routes cannot drift into each other's state.
        """
        num_classes = labels.max().item() + 1
        prototypes = torch.zeros(num_classes, features.shape[-1], device=features.device)
        counts = torch.zeros(num_classes, device=features.device)
        prototypes.scatter_add_(0, labels.unsqueeze(-1).expand_as(features), features)
        counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
        prototypes = prototypes / counts.unsqueeze(-1).clamp(min=1)
        return prototypes

    def _forward_mean_features(
        self,
        s_features: torch.Tensor,
        support_labels: torch.Tensor,
        q_features: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = self.compute_prototypes(s_features, support_labels)
        if self.distance_metric == "cosine":
            return torch.matmul(q_features, prototypes.T) * self.temperature
        dists = torch.cdist(q_features, prototypes, p=2)
        return -dists.pow(2)

    def _forward_mean_distances(
        self,
        s_features: torch.Tensor,
        support_labels: torch.Tensor,
        q_features: torch.Tensor,
    ) -> torch.Tensor:
        num_classes = support_labels.max().item() + 1
        num_queries = q_features.shape[0]
        device = q_features.device

        if self.distance_metric == "cosine":
            all_sim = torch.matmul(q_features, s_features.T) * self.temperature
        else:
            all_dist = torch.cdist(q_features, s_features, p=2)
            all_sim = -all_dist.pow(2)

        logits = torch.zeros(num_queries, num_classes, device=device)
        for c in range(num_classes):
            mask = support_labels == c
            logits[:, c] = all_sim[:, mask].mean(dim=1)
        return logits

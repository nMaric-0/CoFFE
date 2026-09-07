"""Label-free linear compressions of a frozen eval feature.

Motivation. The paper compares a 128-d in-domain encoder (CoFFE, PAPER_CANON
§2) against a foundation model whose eval features are 768-d (spatial /
spectral) or 512-d (fused SEM) — Table 3's three feature columns. A reader can
reasonably ask whether the comparison is confounded by that width: with K=5
supports, a nearest-class-mean prototype in 768-d is estimated from five
vectors in a space where distances concentrate. This module supplies the maps
needed to answer it, by compressing the frozen features to ``dim`` *before*
prototypes are formed.

Every reducer here is **label-free and deterministic**, which is what keeps the
protocol intact: the encoder stays frozen (§4), and no builder in this module
accepts labels — a structural guarantee, asserted in
``tests/unit/test_dim_reduction.py``.

The kinds, and what each one answers:

``identity``
    No compression. The control; its OA must reproduce the published cell.
``pca``
    Top-``dim`` principal subspace of the cached features, fit unsupervised.
    In the least-squares sense this is the *best* linear map into ``dim``
    dimensions, so it is deliberately generous to the foundation model: if OA
    does not move, "the comparison is confounded by feature width" is dead.
    The fit corpus is the caller's choice — the whole labelled pool
    (transductive, still label-free) or the MFT train split only (inductive).
``gaussian_rp``
    Johnson-Lindenstrauss random projection, ``R ~ N(0, 1)^(D x dim) / sqrt(dim)``,
    which preserves squared distances in expectation and fits nothing at all.
    It isolates the cost of the *dimension count itself*; average over seeds.
``stage_block`` / ``stage_mean``
    Structural maps for the fused SEM feature only, which is
    ``num_stages x dr_dim = 4 x 128`` by construction
    (``coffe/models/hypersigma/sem.py``): take one stage's block, or average
    the four. A 512->128 compression the architecture already implies, with no
    fitting.

What is deliberately absent: anything supervised (a learned head would break
the frozen-encoder protocol) and per-episode PCA on the supports (Houston's
episodes have 75 support vectors, so the fit has rank < 128).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Every reducer kind this module can build.
KINDS = ("identity", "pca", "gaussian_rp", "stage_block", "stage_mean")

#: Results-JSON key under which a reduced-feature evaluation records its map.
FEATURE_REDUCTION_KEY = "feature_reduction"


def describe_feature_reduction(results: dict) -> str | None:
    """Short label if ``results`` is a feature-width ablation, else ``None``.

    **Every reader that aggregates ``experiments/*/evaluations/*/results.json``
    must consult this.** An ablation run shares the published cell's
    ``model_type``, ``mode``, geometry and adapted checkpoint, and differs only
    in the width of the feature its prototypes were built from — so a reader
    that classifies by those fields will classify it *as* the published cell
    and can let it stand in for one. Its control is the published config
    itself, so nothing but this key distinguishes the tree.

    Consumers: ``scripts/compile_results.py``,
    ``scripts/reports/aggregate_experiment_results.py``,
    ``scripts/reports/build_experiment_metadata.py``.
    """
    spec = results.get(FEATURE_REDUCTION_KEY) if isinstance(results, dict) else None
    if spec is None:
        return None
    if isinstance(spec, dict):
        return f"{spec.get('kind', 'unknown')}, dim={spec.get('dim')}"
    return str(spec)


@dataclass(frozen=True)
class Reducer:
    """A fitted linear map ``x -> (x - mean) @ matrix``, plus its provenance.

    ``matrix`` is ``[D, dim]`` and ``mean`` is ``[D]`` or ``None`` (no
    centering). ``identity`` carries neither and returns its input unchanged,
    so the control path is exactly the published one rather than a projection
    that happens to be the identity.
    """

    kind: str
    dim: int
    label: str
    spec: dict[str, Any] = field(default_factory=dict)
    matrix: torch.Tensor | None = None
    mean: torch.Tensor | None = None

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the map to a ``[n, D]`` feature matrix."""
        if self.matrix is None:
            return features
        x = features if self.mean is None else features - self.mean.to(features)
        return x @ self.matrix.to(features)

    def to(self, device: str | torch.device) -> Reducer:
        """Move the map's tensors to ``device`` (cheap; they are ``[D, dim]``)."""
        return Reducer(
            kind=self.kind,
            dim=self.dim,
            label=self.label,
            spec=self.spec,
            matrix=None if self.matrix is None else self.matrix.to(device),
            mean=None if self.mean is None else self.mean.to(device),
        )


def identity_reducer(feature_dim: int) -> Reducer:
    """The no-op control at the encoder's native width."""
    return Reducer(
        kind="identity",
        dim=feature_dim,
        label=f"identity_d{feature_dim}",
        spec={"kind": "identity", "dim": feature_dim},
    )


def pca_reducer(
    fit_features: torch.Tensor,
    dim: int,
    *,
    corpus: str,
) -> Reducer:
    """Fit an unsupervised PCA of ``fit_features`` down to ``dim``.

    Args:
        fit_features: ``[n_fit, D]`` cached features — **no labels**.
        dim: target width; must not exceed ``min(n_fit, D)``.
        corpus: name of the fit corpus, recorded in ``spec`` and the label
            (e.g. ``"pool"`` for all labelled features, ``"train"`` for the
            MFT train split).

    ``svd_solver="full"`` and a fixed ``random_state`` make the basis
    reproducible; the fit runs in float64 and the map is stored as float32.
    """
    from sklearn.decomposition import PCA

    n_fit, feature_dim = fit_features.shape
    if dim > min(n_fit, feature_dim):
        raise ValueError(
            f"PCA to {dim} dims needs min(n_fit, D) >= {dim}, got n_fit={n_fit}, D={feature_dim}"
        )

    x = fit_features.detach().cpu().numpy().astype(np.float64)
    pca = PCA(n_components=dim, svd_solver="full", random_state=0).fit(x)
    kept = float(pca.explained_variance_ratio_.sum())
    logger.info(
        "[reduce] pca_%s d=%d fit on %d features: %.2f%% of variance kept",
        corpus,
        dim,
        n_fit,
        kept * 100,
    )
    return Reducer(
        kind="pca",
        dim=dim,
        label=f"pca_{corpus}_d{dim}",
        spec={
            "kind": "pca",
            "dim": dim,
            "fit_corpus": corpus,
            "n_fit": int(n_fit),
            "explained_variance_ratio": kept,
        },
        matrix=torch.from_numpy(pca.components_.T.astype(np.float32)),
        mean=torch.from_numpy(pca.mean_.astype(np.float32)),
    )


def pca_family(
    fit_features: torch.Tensor,
    dims: tuple[int, ...] | list[int],
    *,
    corpus: str,
) -> list[Reducer]:
    """One PCA fit, sliced to every requested width.

    ``svd_solver="full"`` computes the full SVD and truncates it, so the
    ``k``-component basis *is* the first ``k`` components of the ``k_max``
    fit: slicing is bit-identical to fitting each width separately (asserted in
    ``tests/unit/test_dim_reduction.py``) and costs one SVD instead of
    ``len(dims)``. On MUUFL's 53,687 x 768 feature matrix that is the
    difference between ~3 minutes and ~30 seconds.
    """
    from sklearn.decomposition import PCA

    widths = sorted({int(d) for d in dims})
    if not widths:
        return []
    n_fit, feature_dim = fit_features.shape
    if widths[-1] > min(n_fit, feature_dim):
        raise ValueError(
            f"PCA to {widths[-1]} dims needs min(n_fit, D) >= {widths[-1]}, "
            f"got n_fit={n_fit}, D={feature_dim}"
        )

    x = fit_features.detach().cpu().numpy().astype(np.float64)
    pca = PCA(n_components=widths[-1], svd_solver="full", random_state=0).fit(x)
    mean = torch.from_numpy(pca.mean_.astype(np.float32))
    logger.info("[reduce] pca_%s fit once on %d features, sliced to %s", corpus, n_fit, widths)

    reducers = []
    for dim in widths:
        kept = float(pca.explained_variance_ratio_[:dim].sum())
        reducers.append(
            Reducer(
                kind="pca",
                dim=dim,
                label=f"pca_{corpus}_d{dim}",
                spec={
                    "kind": "pca",
                    "dim": dim,
                    "fit_corpus": corpus,
                    "n_fit": int(n_fit),
                    "explained_variance_ratio": kept,
                },
                matrix=torch.from_numpy(pca.components_[:dim].T.astype(np.float32)),
                mean=mean,
            )
        )
    return reducers


def gaussian_rp_reducer(feature_dim: int, dim: int, *, seed: int) -> Reducer:
    """A Johnson-Lindenstrauss Gaussian random projection to ``dim``.

    ``R[i, j] ~ N(0, 1) / sqrt(dim)``, drawn from ``np.random.default_rng(seed)``
    so the same seed gives the same map on any machine. No centering and no
    fitting: this measures what the dimension count costs on its own, and
    should be reported as a mean over several seeds.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((feature_dim, dim)) / np.sqrt(dim)
    return Reducer(
        kind="gaussian_rp",
        dim=dim,
        label=f"gaussian_rp_d{dim}_s{seed}",
        spec={"kind": "gaussian_rp", "dim": dim, "seed": int(seed)},
        matrix=torch.from_numpy(matrix.astype(np.float32)),
    )


def stage_block_reducer(feature_dim: int, *, stage: int, num_stages: int = 4) -> Reducer:
    """Select one SEM stage's ``dr_dim`` block out of the fused feature.

    The fused feature is ``cat([pooled_stage_0, ..., pooled_stage_{S-1}])``
    (``coffe/models/hypersigma/sem.py``), so stage ``i`` occupies columns
    ``[i * dr_dim, (i + 1) * dr_dim)``.
    """
    dr_dim, remainder = divmod(feature_dim, num_stages)
    if remainder:
        raise ValueError(f"feature_dim={feature_dim} is not {num_stages} equal stage blocks")
    if not 0 <= stage < num_stages:
        raise ValueError(f"stage={stage} outside 0..{num_stages - 1}")

    matrix = torch.zeros(feature_dim, dr_dim)
    matrix[stage * dr_dim : (stage + 1) * dr_dim] = torch.eye(dr_dim)
    return Reducer(
        kind="stage_block",
        dim=dr_dim,
        label=f"stage{stage}_d{dr_dim}",
        spec={"kind": "stage_block", "dim": dr_dim, "stage": int(stage), "num_stages": num_stages},
        matrix=matrix,
    )


def stage_mean_reducer(feature_dim: int, *, num_stages: int = 4) -> Reducer:
    """Average the SEM stage blocks into one ``dr_dim`` feature."""
    dr_dim, remainder = divmod(feature_dim, num_stages)
    if remainder:
        raise ValueError(f"feature_dim={feature_dim} is not {num_stages} equal stage blocks")

    matrix = torch.eye(dr_dim).repeat(num_stages, 1) / num_stages
    return Reducer(
        kind="stage_mean",
        dim=dr_dim,
        label=f"stage_mean_d{dr_dim}",
        spec={"kind": "stage_mean", "dim": dr_dim, "num_stages": num_stages},
        matrix=matrix,
    )


def build_reducer(
    kind: str,
    *,
    feature_dim: int,
    dim: int | None = None,
    fit_features: torch.Tensor | None = None,
    corpus: str = "pool",
    seed: int | None = None,
    stage: int | None = None,
    num_stages: int = 4,
) -> Reducer:
    """Dispatch to one of the builders above by ``kind``.

    Note the absent argument: no reducer in this module can see labels.
    """
    if kind == "identity":
        return identity_reducer(feature_dim)
    if kind == "pca":
        if fit_features is None or dim is None:
            raise ValueError("pca needs fit_features and dim")
        return pca_reducer(fit_features, dim, corpus=corpus)
    if kind == "gaussian_rp":
        if dim is None or seed is None:
            raise ValueError("gaussian_rp needs dim and seed")
        return gaussian_rp_reducer(feature_dim, dim, seed=seed)
    if kind == "stage_block":
        if stage is None:
            raise ValueError("stage_block needs stage")
        return stage_block_reducer(feature_dim, stage=stage, num_stages=num_stages)
    if kind == "stage_mean":
        return stage_mean_reducer(feature_dim, num_stages=num_stages)
    raise ValueError(f"unknown reducer kind {kind!r}; expected one of {KINDS}")

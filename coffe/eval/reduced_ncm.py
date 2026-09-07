"""Nearest-class-mean episodes over a cached, optionally compressed feature.

This is PAPER_CANON §4's protocol — N-way, K=5, 100 queries per class,
Euclidean nearest class mean — run over the feature matrix produced by
:mod:`coffe.eval.feature_cache` instead of re-encoding patches per episode.
Two properties make it a valid stand-in for the live loop rather than a second
protocol:

* **Same episodes.** The plan is drawn from the same
  :class:`~coffe.data.samplers.patched_episode_sampler.PatchedEpisodeSampler`
  through :meth:`sample_episode_indices`, which consumes the sampler's RNG in
  the same order as ``sample_episode``. At a given seed the episode stream is
  identical to the published run's, so a compressed variant and its control are
  *paired* episode by episode — hence :func:`paired_delta` rather than
  comparing two confidence intervals.
* **Same classifier.** The logits come from ``_compute_logits`` and the metrics
  from ``compute_metrics_from_confusion_matrix``, both imported from the
  evaluators rather than reimplemented, so there is exactly one NCM in the
  repository. (``_compute_logits`` is private to
  :mod:`coffe.eval.hypersigma`; importing it is deliberate — duplicating the
  prototype math is the thing worth avoiding.)

Three metric blocks are accumulated in one pass over the episodes, mirroring
how the HyperSIGMA evaluator accumulates cosine and Euclidean together:

``euclidean``
    The paper's metric on the reduced feature as the map returns it.
``euclidean_l2``
    The same after re-normalising the reduced feature to unit length, which is
    the geometry the 768-d/512-d features already had (``_extract`` L2-normalises
    them) and which projection destroys. For ``identity`` the two blocks
    coincide up to float noise — a built-in consistency check.
``cosine``
    Recorded for parity with the published JSONs. Scale-invariant per vector,
    so it is by construction identical for the raw and re-normalised features
    and is therefore computed once.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from coffe.data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from coffe.eval.episodic import compute_metrics_from_confusion_matrix
from coffe.eval.hypersigma import _compute_logits

logger = logging.getLogger(__name__)

#: The metric blocks :func:`run_episodes` accumulates in one pass.
METRIC_BLOCKS = ("euclidean", "euclidean_l2", "cosine")


@dataclass(frozen=True)
class EpisodePlan:
    """The dataset indices of every episode, drawn once and reused.

    ``support_indices`` is ``[E, N*K]`` and ``query_indices`` ``[E, N*Q]``,
    both ordered class-major so the episode labels are the fixed patterns in
    :attr:`support_labels` / :attr:`query_labels` — exactly the order in which
    ``sample_episode`` stacks its tensors.
    """

    support_indices: np.ndarray
    query_indices: np.ndarray
    original_classes: np.ndarray
    n_way: int
    k_shot: int
    k_query: int
    seed: int

    @property
    def num_episodes(self) -> int:
        return int(self.support_indices.shape[0])

    @property
    def support_labels(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_way), self.k_shot)

    @property
    def query_labels(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_way), self.k_query)


def build_episode_plan(
    sampler: PatchedEpisodeSampler,
    num_episodes: int,
    *,
    seed: int,
) -> EpisodePlan:
    """Draw ``num_episodes`` episodes as indices, touching no patch data.

    ``seed`` is recorded on the plan for the artifact; it must be the seed
    ``sampler`` was constructed with, since that is what fixes the stream.
    """
    support, query, classes = [], [], []
    for _ in range(num_episodes):
        selected_classes, per_class_indices = sampler.sample_episode_indices()
        support.append(
            [idx for per_class in per_class_indices for idx in per_class[: sampler.k_shot]]
        )
        query.append(
            [idx for per_class in per_class_indices for idx in per_class[sampler.k_shot :]]
        )
        classes.append(np.asarray(selected_classes))

    plan = EpisodePlan(
        support_indices=np.asarray(support, dtype=np.int64),
        query_indices=np.asarray(query, dtype=np.int64),
        original_classes=np.asarray(classes),
        n_way=sampler.n_way,
        k_shot=sampler.k_shot,
        k_query=sampler.k_query,
        seed=int(seed),
    )
    logger.info(
        "[plan] %d episodes, %d-way, %d-shot, %d queries/class",
        plan.num_episodes,
        plan.n_way,
        plan.k_shot,
        plan.k_query,
    )
    return plan


def episode_ci(values: np.ndarray | list[float]) -> tuple[float, float, float]:
    """``(mean, std, 95% CI half-width)`` over episodes.

    Mirrors the ``_ci`` helper inside ``coffe.eval.hypersigma.evaluate`` — the
    same population std and Student-t half-width — so the numbers here are
    directly comparable to the published Table 3 cells. That the control run
    reproduces a published cell's CI as well as its OA is the check that this
    is in fact the same formula.
    """
    from scipy.stats import t as t_dist

    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    ci = float(t_dist.ppf(0.975, df=max(1, len(arr) - 1)) * std / max(1, np.sqrt(len(arr))))
    return mean, std, ci


@torch.no_grad()
def run_episodes(
    features: torch.Tensor,
    plan: EpisodePlan,
    *,
    num_total_classes: int,
    temperature: float = 10.0,
    device: str = "cpu",
    blocks: tuple[str, ...] = METRIC_BLOCKS,
) -> dict[str, dict[str, Any]]:
    """Run the plan's episodes over ``features`` and return one blob per block."""
    unknown = set(blocks) - set(METRIC_BLOCKS)
    if unknown:
        raise ValueError(f"unknown metric block(s) {sorted(unknown)}; expected {METRIC_BLOCKS}")

    z_raw = features.to(device).float()
    z_l2 = F.normalize(z_raw, p=2, dim=-1) if "euclidean_l2" in blocks else None

    n_way = plan.n_way
    support_labels = torch.as_tensor(plan.support_labels, device=device, dtype=torch.long)
    query_labels_t = torch.as_tensor(plan.query_labels, device=device, dtype=torch.long)

    acc: dict[str, dict[str, Any]] = {
        block: {
            "episode_oas": [],
            "episode_aas": [],
            "episode_kappas": [],
            "class_results": defaultdict(lambda: {"correct": 0, "total": 0}),
            "class_episode_accs": defaultdict(list),
            "global_conf_matrix": np.zeros(
                (num_total_classes + 1, num_total_classes + 1), dtype=np.int64
            ),
        }
        for block in blocks
    }

    support_idx_all = torch.as_tensor(plan.support_indices, device=device)
    query_idx_all = torch.as_tensor(plan.query_indices, device=device)

    for episode in range(plan.num_episodes):
        sup_idx = support_idx_all[episode]
        qry_idx = query_idx_all[episode]
        original_classes = plan.original_classes[episode]

        for block in blocks:
            source = z_l2 if block == "euclidean_l2" else z_raw
            metric = "cosine" if block == "cosine" else "euclidean"
            s_feat = source.index_select(0, sup_idx)
            q_feat = source.index_select(0, qry_idx)

            logits = _compute_logits(
                s_feat,
                support_labels,
                q_feat,
                metric=metric,
                temperature=temperature,
            )
            preds = logits.argmax(dim=1)

            # Same counts as the live loop's `for t, p in zip(...): conf[t, p] += 1`.
            conf = (
                torch.bincount(query_labels_t * n_way + preds, minlength=n_way * n_way)
                .reshape(n_way, n_way)
                .cpu()
                .numpy()
            )
            metrics = compute_metrics_from_confusion_matrix(conf)

            blk = acc[block]
            blk["episode_oas"].append(metrics["OA"])
            blk["episode_aas"].append(metrics["AA"])
            blk["episode_kappas"].append(metrics["Kappa"])
            for way_idx, orig in enumerate(original_classes):
                total = int(conf[way_idx, :].sum())
                correct = int(conf[way_idx, way_idx])
                blk["class_results"][int(orig)]["correct"] += correct
                blk["class_results"][int(orig)]["total"] += total
                if total > 0:
                    blk["class_episode_accs"][int(orig)].append(correct / total * 100)
            for i, oi in enumerate(original_classes):
                for j, oj in enumerate(original_classes):
                    blk["global_conf_matrix"][int(oi), int(oj)] += conf[i, j]

        if (episode + 1) % 500 == 0:
            logger.info(
                "[episodes] %d/%d (%s OA so far %.2f)",
                episode + 1,
                plan.num_episodes,
                blocks[0],
                float(np.mean(acc[blocks[0]]["episode_oas"])),
            )

    return acc


def summarize(block: dict[str, Any]) -> dict[str, Any]:
    """Reduce one accumulated block to the published results-JSON shape."""
    summary: dict[str, Any] = {}
    for key in ("OA", "AA", "Kappa"):
        mean, std, ci = episode_ci(block[f"episode_{key.lower()}s"])
        summary[key] = {"mean": mean, "std": std, "ci_95": ci}

    per_class = {}
    for orig, data in block["class_results"].items():
        accs = block["class_episode_accs"].get(orig, [])
        if accs:
            mean, std, ci = episode_ci(accs)
        else:
            pooled = data["correct"] / data["total"] * 100 if data["total"] else 0.0
            mean, std, ci = pooled, 0.0, 0.0
        per_class[orig] = {
            "accuracy": float(mean),
            "pooled_accuracy": float(
                data["correct"] / data["total"] * 100 if data["total"] else 0.0
            ),
            "std": float(std),
            "ci_95": float(ci),
            "total_samples": int(data["total"]),
            "correct_samples": int(data["correct"]),
        }

    summary["per_class"] = per_class
    summary["num_episodes"] = len(block["episode_oas"])
    return summary


def paired_delta(control_oas: np.ndarray, variant_oas: np.ndarray) -> dict[str, Any]:
    """Compare a variant to its control on the *same* episodes.

    Both runs saw the identical episode stream, so the per-episode OA
    differences are paired: the mean difference and its CI are far tighter than
    the two runs' own CIs would suggest, and a Wilcoxon signed-rank test says
    whether a small shift is real. Returns OA percentage points.
    """
    from scipy.stats import wilcoxon

    control = np.asarray(control_oas, dtype=float)
    variant = np.asarray(variant_oas, dtype=float)
    if control.shape != variant.shape:
        raise ValueError(f"unpaired episode counts: {control.shape} vs {variant.shape}")

    diff = variant - control
    mean, std, ci = episode_ci(diff)
    if np.allclose(diff, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(diff)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "delta_oa": mean,
        "delta_std": std,
        "delta_ci_95": ci,
        "episodes_improved": int((diff > 0).sum()),
        "episodes_worsened": int((diff < 0).sum()),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "n_episodes": int(diff.size),
    }

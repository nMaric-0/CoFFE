"""Few-shot evaluation for HyperSIGMA on the same episodes as CoFFE.

Mirrors ``scripts/evaluate.py`` so result files are
directly comparable. Differences:

* Encoder is ``HyperSIGMAFewShot`` (a frozen wrapper around
  ``HyperSIGMADual``), not ``CoFFE``.
* The eval loop accumulates *both* a cosine and a Euclidean confusion
  matrix from the same per-batch features so the JSON has parallel
  ``cosine`` and ``euclidean`` result blocks.
* CLI flags add ``--mode {fused, spat_pool, spec_pool}``,
  ``--spat-patch-k {1, 3}``, ``--pca-spat-path``, and
  ``--adapted-checkpoint`` (use the literal string ``none`` to skip
  loading adapted weights — that's the "unadapted" ablation).

Usage:
    python scripts/evaluate_hypersigma.py \
        --pca-spat-path checkpoints/hypersigma/pca_houston_3band.pkl \
        --spat-ckpt checkpoints/hypersigma/spat-vit-base.pth \
        --spec-ckpt checkpoints/hypersigma/spec-vit-base.pth \
        --adapted-checkpoint checkpoints/hypersigma_adapted/houston_k3/checkpoint.pth \
        --mode fused
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from coffe.data.datasets import DATASET_REGISTRY, get_spec
from coffe.data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from coffe.eval.episodic import (
    CLASS_NAMES,
    compute_metrics_from_confusion_matrix,
    generate_plots,
)
from coffe.models.hypersigma import HyperSIGMADual, HyperSIGMAFewShot
from coffe.utils.seed import set_seed

# Derived from the central registry (coffe/data/datasets/registry.py) so band
# and class counts cannot drift from the adapt pipeline.
DATASETS = {name: spec.patched_cls for name, spec in DATASET_REGISTRY.items()}

DATASET_SPECS = {
    name: {"hsi_channels": spec.hsi_channels, "num_classes": spec.num_classes}
    for name, spec in DATASET_REGISTRY.items()
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------


# Which dual-branch components each eval mode actually consumes. Building only
# these avoids loading an unused ViT-B (~half the encoder) into GPU memory.
_MODE_BRANCHES = {
    "fused": {"build_spat": True, "build_spec": True, "build_sem": True},
    "spat_pool": {"build_spat": True, "build_spec": False, "build_sem": False},
    "spec_pool": {"build_spat": False, "build_spec": True, "build_sem": False},
}


def _build_dual(
    dataset_name: str,
    pca_spat_path: str,
    spat_ckpt: str,
    spec_ckpt: str,
    spat_patch_k: int,
    embed_dim: int = 768,
    num_tokens: int = 100,
    dr_dim: int = 128,
    num_stages: int = 4,
    pca_stats_path: str | None = None,
    mode: str = "fused",
    spat_resample_to: int | None = None,
    native_geometry: bool = False,
    input_fit: str = "upscale",
    pad_anchor: str = "center",
    interp_mode: str = "bicubic",
    native_pca_spat_path: str | None = None,
) -> HyperSIGMADual:
    specs = DATASET_SPECS[dataset_name]
    branches = _MODE_BRANCHES.get(mode, _MODE_BRANCHES["fused"])
    # native_geometry composes with any mode: spat_pool / spec_pool (single-branch
    # unadapted ablation) or fused (the SEM-tuning ablation, with an adapted ckpt).
    dual = HyperSIGMADual(
        pca_spat_path=pca_spat_path,
        spat_ckpt=spat_ckpt,
        spec_ckpt=spec_ckpt,
        hsi_channels=specs["hsi_channels"],
        spat_patch_k=spat_patch_k,
        freeze_body=True,
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        dr_dim=dr_dim,
        num_stages=num_stages,
        pca_stats_path=pca_stats_path,
        spat_resample_to=spat_resample_to,
        native_geometry=native_geometry,
        input_fit=input_fit,
        pad_anchor=pad_anchor,
        interp_mode=interp_mode,
        native_pca_spat_path=native_pca_spat_path,
        **branches,
    )
    dual.log_sanity()
    return dual


def _load_adapted_checkpoint(model: HyperSIGMAFewShot, ckpt_path: str) -> None:
    """Apply an adapted-state checkpoint to the dual encoder.

    The adapted checkpoint is the state_dict of a
    ``HyperSIGMAMaskedAdaptation`` — keys are prefixed with ``dual.``,
    plus ``input_masking.*`` and ``decoder.*``. We keep only the
    ``dual.*`` keys.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    dual_state = {}
    for k, v in state.items():
        if k.startswith("dual."):
            dual_state[k[len("dual.") :]] = v
    if not dual_state:
        raise RuntimeError(f"No 'dual.*' keys found in adapted checkpoint at {ckpt_path}")

    missing, unexpected = model.dual.load_state_dict(dual_state, strict=False)
    logger.info(
        "[HyperSIGMA] Loaded adapted dual state: missing=%d, unexpected=%d",
        len(missing),
        len(unexpected),
    )
    if missing:
        logger.warning("Missing keys (first 10): %s", missing[:10])
    if unexpected:
        logger.warning("Unexpected keys (first 10): %s", unexpected[:10])


def load_model(
    dataset_name: str,
    pca_spat_path: str,
    spat_ckpt: str,
    spec_ckpt: str,
    spat_patch_k: int,
    adapted_checkpoint: str | None,
    mode: str,
    temperature: float,
    prototype_mode: str,
    distance_metric: str,
    device: str,
    pca_stats_path: str | None = None,
    spat_resample_to: int | None = None,
    native_geometry: bool = False,
    input_fit: str = "upscale",
    pad_anchor: str = "center",
    interp_mode: str = "bicubic",
    native_pca_spat_path: str | None = None,
) -> HyperSIGMAFewShot:
    """Build the frozen HyperSIGMA evaluator for one Table 3 configuration.

    ``mode`` selects the feature column — ``spat_pool`` (spatial 768-d),
    ``spec_pool`` (spectral 768-d) or ``fused`` (SEM 512-d);
    ``native_geometry`` + ``input_fit`` select the 64x64 backbone-native
    regime (pad or upscale) over the 11x11 patch-native one; and
    ``adapted_checkpoint``, when given, loads the label-free adaptation on
    top. Returns the model in eval mode on ``device`` (PAPER_CANON §5).
    """
    has_adapted = bool(
        adapted_checkpoint and adapted_checkpoint.lower() not in ("none", "null", "random")
    )
    # Native geometry can be evaluated unadapted (Ablation 1) OR with a
    # native-trained adapted checkpoint (Ablation 2, SEM tuning). A checkpoint
    # trained at the *small adapted* geometry is shape-incompatible with the
    # native encoder — `_load_adapted_checkpoint` loads strict=False and logs the
    # resulting missing/unexpected keys, which is the signal that you mixed
    # geometries (expect missing≈0 / unexpected≈0 for a matched native checkpoint).
    dual = _build_dual(
        dataset_name=dataset_name,
        pca_spat_path=pca_spat_path,
        spat_ckpt=spat_ckpt,
        spec_ckpt=spec_ckpt,
        spat_patch_k=spat_patch_k,
        pca_stats_path=pca_stats_path,
        mode=mode,
        spat_resample_to=spat_resample_to,
        native_geometry=native_geometry,
        input_fit=input_fit,
        pad_anchor=pad_anchor,
        interp_mode=interp_mode,
        native_pca_spat_path=native_pca_spat_path,
    )
    model = HyperSIGMAFewShot(
        dual=dual,
        mode=mode,
        distance_metric=distance_metric,
        temperature=temperature,
        prototype_mode=prototype_mode,
    )

    if has_adapted:
        _load_adapted_checkpoint(model, adapted_checkpoint)
    elif native_geometry:
        logger.info(
            "[HyperSIGMA] Running NATIVE-GEOMETRY ablation (mode=%s, input_fit=%s, unadapted)",
            mode,
            input_fit,
        )
    else:
        logger.info("[HyperSIGMA] Running UNADAPTED ablation (no Houston adaptation applied)")

    model = model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Eval loop
# ----------------------------------------------------------------------


def _compute_logits(
    s_features: torch.Tensor,
    support_labels: torch.Tensor,
    q_features: torch.Tensor,
    *,
    metric: str,
    temperature: float,
) -> torch.Tensor:
    """Prototype matching with explicit metric so we can run both in parallel."""
    if metric == "cosine":
        s_norm = F.normalize(s_features, p=2, dim=-1)
        q_norm = F.normalize(q_features, p=2, dim=-1)
    else:
        s_norm = s_features
        q_norm = q_features

    num_classes = support_labels.max().item() + 1
    prototypes = torch.zeros(num_classes, s_norm.shape[-1], device=s_norm.device)
    counts = torch.zeros(num_classes, device=s_norm.device)
    prototypes.scatter_add_(0, support_labels.unsqueeze(-1).expand_as(s_norm), s_norm)
    counts.scatter_add_(0, support_labels, torch.ones_like(support_labels, dtype=s_norm.dtype))
    prototypes = prototypes / counts.unsqueeze(-1).clamp(min=1)

    if metric == "cosine":
        return torch.matmul(q_norm, prototypes.T) * temperature
    dists = torch.cdist(q_norm, prototypes, p=2)
    return -dists.pow(2)


@torch.no_grad()
def evaluate(
    model: HyperSIGMAFewShot,
    sampler: PatchedEpisodeSampler,
    device: str,
    *,
    num_episodes: int,
    n_way: int,
    num_total_classes: int,
    temperature: float,
    num_example_episodes: int,
    max_tsne_samples: int,
) -> dict:
    """Run the prototype eval and return a results blob with parallel
    cosine / euclidean confusion matrices."""
    model.eval()

    blocks = {}
    for metric in ("cosine", "euclidean"):
        blocks[metric] = {
            "episode_oas": [],
            "episode_aas": [],
            "episode_kappas": [],
            "class_results": defaultdict(lambda: {"correct": 0, "total": 0}),
            "class_episode_accs": defaultdict(list),
            "global_conf_matrix": np.zeros(
                (num_total_classes + 1, num_total_classes + 1),
                dtype=np.int64,
            ),
            "pairwise_confusion": defaultdict(lambda: defaultdict(int)),
            "pairwise_totals": defaultdict(lambda: defaultdict(int)),
        }

    example_episodes_data = []
    aggregated_features = defaultdict(list)

    primary_metric = model.distance_metric

    # Opt-in per-section timing breakdown (set HYPERSIGMA_EVAL_PROFILE=1).
    # When off, the `if profile` branches are skipped and the loop is unchanged.
    profile = os.environ.get("HYPERSIGMA_EVAL_PROFILE") == "1"
    profile_cuda = profile and isinstance(device, str) and device.startswith("cuda")
    PROFILE_WARMUP = 3  # skip first episodes (CUDA/cuDNN warmup inflates them)

    def _stamp() -> float:
        # Wait for async CUDA kernels so GPU timings are real, not just launches.
        if profile_cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    timings: dict = defaultdict(float)
    timed_episodes = 0
    last_end = None

    pbar = tqdm(sampler, total=min(num_episodes, len(sampler)), desc="Evaluating")
    completed = 0
    for episode in pbar:
        if completed >= num_episodes:
            break
        sec = {}
        if profile:
            t_top = _stamp()
            if last_end is not None:
                sec["data_wait"] = t_top - last_end

        support_hsi = episode["support_hsi"].to(device)
        support_aux = episode["support_aux"].to(device)
        support_labels = episode["support_labels"].to(device)
        query_hsi = episode["query_hsi"].to(device)
        query_aux = episode["query_aux"].to(device)
        query_labels = episode["query_labels"].to(device)
        original_classes = episode["original_classes"].tolist()
        if profile:
            t_h2d = _stamp()
            sec["h2d"] = t_h2d - t_top

        s_patch, s_cls, _ = model.forward_features(support_hsi, support_aux)
        if profile:
            t_fs = _stamp()
            sec["fwd_support"] = t_fs - t_h2d
        q_patch, q_cls, _ = model.forward_features(query_hsi, query_aux)
        if profile:
            t_fq = _stamp()
            sec["fwd_query"] = t_fq - t_fs
        s_feat = model.eval_patch_embeddings(s_patch, s_cls).mean(dim=1)
        q_feat = model.eval_patch_embeddings(q_patch, q_cls).mean(dim=1)
        if profile:
            t_ad = _stamp()
            sec["adapt"] = t_ad - t_fq

        for metric in ("cosine", "euclidean"):
            logits = _compute_logits(
                s_feat,
                support_labels,
                q_feat,
                metric=metric,
                temperature=temperature,
            )
            preds = logits.argmax(dim=1)
            query_cpu = query_labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()

            conf = np.zeros((n_way, n_way), dtype=np.int64)
            for t, p in zip(query_cpu, preds_cpu):
                conf[t, p] += 1
            m = compute_metrics_from_confusion_matrix(conf)
            blk = blocks[metric]
            blk["episode_oas"].append(m["OA"])
            blk["episode_aas"].append(m["AA"])
            blk["episode_kappas"].append(m["Kappa"])

            for way_idx, orig in enumerate(original_classes):
                total = conf[way_idx, :].sum()
                correct = conf[way_idx, way_idx]
                blk["class_results"][orig]["correct"] += correct
                blk["class_results"][orig]["total"] += total
                if total > 0:
                    blk["class_episode_accs"][orig].append(correct / total * 100)
            for i, oi in enumerate(original_classes):
                for j, oj in enumerate(original_classes):
                    blk["global_conf_matrix"][oi, oj] += conf[i, j]
                    blk["pairwise_confusion"][oi][oj] += conf[i, j]
                    blk["pairwise_totals"][oi][oj] += conf[i, :].sum()

            # Feature snapshots and aggregated features only need the
            # primary metric (these drive the plots).
            if metric == primary_metric:
                if completed < num_example_episodes:
                    s_norm = F.normalize(s_feat, p=2, dim=-1) if metric == "cosine" else s_feat
                    q_norm = F.normalize(q_feat, p=2, dim=-1) if metric == "cosine" else q_feat
                    prototypes = torch.zeros(
                        support_labels.max().item() + 1,
                        s_norm.shape[-1],
                        device=s_norm.device,
                    )
                    counts = torch.zeros_like(prototypes[:, 0])
                    prototypes.scatter_add_(
                        0,
                        support_labels.unsqueeze(-1).expand_as(s_norm),
                        s_norm,
                    )
                    counts.scatter_add_(
                        0,
                        support_labels,
                        torch.ones_like(support_labels, dtype=s_norm.dtype),
                    )
                    prototypes = prototypes / counts.unsqueeze(-1).clamp(min=1)
                    example_episodes_data.append(
                        {
                            "s_features": s_norm.cpu().numpy(),
                            "q_features": q_norm.cpu().numpy(),
                            "prototypes": prototypes.cpu().numpy(),
                            "s_labels": support_labels.cpu().numpy(),
                            "q_labels": query_cpu,
                            "q_preds": preds_cpu,
                            "original_classes": original_classes,
                        }
                    )
                q_feats_np = (
                    (F.normalize(q_feat, p=2, dim=-1) if metric == "cosine" else q_feat)
                    .cpu()
                    .numpy()
                )
                for way_idx, orig in enumerate(original_classes):
                    mask = query_cpu == way_idx
                    feats = q_feats_np[mask]
                    cur = len(aggregated_features[orig])
                    if max_tsne_samples > 0 and cur >= max_tsne_samples:
                        continue
                    if max_tsne_samples > 0:
                        feats = feats[: max_tsne_samples - cur]
                    aggregated_features[orig].append(feats)

        if profile:
            t_post = _stamp()
            sec["post"] = t_post - t_ad
            last_end = t_post
            if completed >= PROFILE_WARMUP:
                for k, v in sec.items():
                    timings[k] += v
                timed_episodes += 1
                if timed_episodes:
                    pbar.set_postfix(
                        {
                            "fwd_s(ms)": f"{timings['fwd_support'] / timed_episodes * 1e3:.0f}",
                            "fwd_q(ms)": f"{timings['fwd_query'] / timed_episodes * 1e3:.0f}",
                            "data(ms)": f"{timings['data_wait'] / timed_episodes * 1e3:.0f}",
                        }
                    )
        completed += 1

    if profile and timed_episodes:
        order = ("data_wait", "h2d", "fwd_support", "fwd_query", "adapt", "post")
        means_ms = {k: timings[k] / timed_episodes * 1e3 for k in order}
        total_ms = sum(means_ms.values())
        logger.info(
            "[HyperSIGMA][profile] mean ms/episode over %d episodes (warmup %d skipped):",
            timed_episodes,
            PROFILE_WARMUP,
        )
        for k in order:
            pct = (means_ms[k] / total_ms * 100) if total_ms else 0.0
            logger.info("    %-12s %8.1f ms  (%4.1f%%)", k, means_ms[k], pct)
        logger.info(
            "    %-12s %8.1f ms  -> %.2f it/s",
            "TOTAL",
            total_ms,
            (1e3 / total_ms) if total_ms else 0.0,
        )

    # Concatenate aggregated features
    for cls in aggregated_features:
        aggregated_features[cls] = np.concatenate(aggregated_features[cls], axis=0)

    def _ci(values):
        from scipy.stats import t as t_dist

        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        ci = float(t_dist.ppf(0.975, df=max(1, len(arr) - 1)) * std / max(1, np.sqrt(len(arr))))
        return mean, std, ci

    summary = {}
    for metric, blk in blocks.items():
        oa_m, oa_s, oa_ci = _ci(blk["episode_oas"])
        aa_m, aa_s, aa_ci = _ci(blk["episode_aas"])
        k_m, k_s, k_ci = _ci(blk["episode_kappas"])
        per_class = {}
        for orig, data in blk["class_results"].items():
            accs = blk["class_episode_accs"].get(orig, [])
            if accs:
                m, s, ci = _ci(accs)
            else:
                pooled = data["correct"] / data["total"] * 100 if data["total"] else 0.0
                m, s, ci = pooled, 0.0, 0.0
            per_class[orig] = {
                "accuracy": float(m),
                "pooled_accuracy": float(
                    data["correct"] / data["total"] * 100 if data["total"] else 0.0
                ),
                "std": float(s),
                "ci_95": float(ci),
                "total_samples": int(data["total"]),
                "correct_samples": int(data["correct"]),
            }
        summary[metric] = {
            "OA": {"mean": oa_m, "std": oa_s, "ci_95": oa_ci},
            "AA": {"mean": aa_m, "std": aa_s, "ci_95": aa_ci},
            "Kappa": {"mean": k_m, "std": k_s, "ci_95": k_ci},
            "per_class": per_class,
            "num_episodes": len(blk["episode_oas"]),
            "global_conf_matrix": blk["global_conf_matrix"],
            "episode_oas": np.asarray(blk["episode_oas"]),
            "episode_aas": np.asarray(blk["episode_aas"]),
            "episode_kappas": np.asarray(blk["episode_kappas"]),
            "class_episode_accs": dict(blk["class_episode_accs"]),
            "pairwise_confusion": {k: dict(v) for k, v in blk["pairwise_confusion"].items()},
            "pairwise_totals": {k: dict(v) for k, v in blk["pairwise_totals"].items()},
        }

    summary["example_episodes_data"] = example_episodes_data
    summary["aggregated_features"] = dict(aggregated_features)
    return summary


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def _to_json_friendly(block: dict) -> dict:
    """Strip numpy arrays / non-serializable bits before json.dump."""
    out = {}
    for k, v in block.items():
        if k in {
            "global_conf_matrix",
            "episode_oas",
            "episode_aas",
            "episode_kappas",
        }:
            continue
        if k in {"class_episode_accs", "pairwise_confusion", "pairwise_totals"}:
            continue
        out[k] = v
    return out


def _write_results_json(
    output_path: Path,
    results: dict,
    *,
    dataset_name: str,
    args,
    seeds,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cosine = _to_json_friendly(results["cosine"])
    euclidean = _to_json_friendly(results["euclidean"])
    primary = results[args.distance_metric]
    payload = {
        "model_type": "HyperSIGMADual",
        "mode": args.mode,
        "spat_patch_k": args.spat_patch_k,
        "spat_resample_to": getattr(args, "spat_resample_to", None),
        "native_geometry": getattr(args, "native_geometry", False),
        "input_fit": getattr(args, "input_fit", "upscale"),
        "pad_anchor": getattr(args, "pad_anchor", "center"),
        "interp_mode": getattr(args, "interp_mode", "bicubic"),
        "native_pca_spat_path": getattr(args, "native_pca_spat_path", None),
        "adapted_checkpoint": args.adapted_checkpoint,
        "pca_spat_path": args.pca_spat_path,
        "pca_stats_path": getattr(args, "pca_stats_path", None),
        "spat_ckpt": args.spat_ckpt,
        "spec_ckpt": args.spec_ckpt,
        "dataset": dataset_name,
        "split": args.split,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "k_query": args.k_query,
        "seeds": seeds,
        "num_episodes": primary["num_episodes"],
        "distance_metric": args.distance_metric,
        "temperature": args.temperature,
        "prototype_mode": args.prototype_mode,
        "cosine": cosine,
        "euclidean": euclidean,
        # Promote the primary metric's headline OA/AA/Kappa to the top
        # level so existing tooling that looks for results["OA"] still
        # works.
        "OA": primary["OA"],
        "AA": primary["AA"],
        "Kappa": primary["Kappa"],
        "per_class": {str(k): v for k, v in primary["per_class"].items()},
        "class_names": CLASS_NAMES.get(dataset_name, []),
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


# ----------------------------------------------------------------------
# Plot adapter
# ----------------------------------------------------------------------


def _shape_for_plots(block: dict) -> dict:
    """Reshape a metric block into the dict ``generate_plots`` expects."""
    return {
        "OA": block["OA"],
        "AA": block["AA"],
        "Kappa": block["Kappa"],
        "per_class": block["per_class"],
        "num_episodes": block["num_episodes"],
        "global_conf_matrix": block["global_conf_matrix"],
        "episode_oas": block["episode_oas"],
        "episode_aas": block["episode_aas"],
        "episode_kappas": block["episode_kappas"],
        "class_episode_accs": block["class_episode_accs"],
        "pairwise_confusion": block["pairwise_confusion"],
        "pairwise_totals": block["pairwise_totals"],
        "example_episodes_data": [],
        "aggregated_features": {},
    }


# ----------------------------------------------------------------------
# Main / programmatic entry-point
# ----------------------------------------------------------------------


def _resolve_dataset_paths(args, dataset_name: str) -> None:
    """Fill in dataset-coupled paths from the registry convention.

    Makes ``--dataset <name>`` enough to evaluate: the spatial PCA and its
    output-stats are strictly tied to the dataset, so when not given they
    default to ``checkpoints/hypersigma/pca_<name>_3band[.|_stats.]pkl``.

    ``--adapted-checkpoint``:
      * omitted/None  -> unadapted ablation (no adapted weights loaded)
      * "auto"        -> the convention path
                         ``checkpoints/hypersigma_adapted/<name>_k<k>/checkpoint.pth``
      * any other str -> used verbatim ("none"/"null"/"random" still skip)
    """
    spec = get_spec(dataset_name)
    if not getattr(args, "pca_spat_path", None):
        args.pca_spat_path = spec.pca_spat_path()
    if not getattr(args, "pca_stats_path", None):
        conv = spec.pca_stats_path()
        # Only standardize if the stats file actually exists; otherwise
        # fall back to raw PCA output (None), matching prior behavior.
        args.pca_stats_path = conv if Path(conv).exists() else None
    # Native-geometry spatial front-end: the pretrained SpatViT patch_embed
    # needs 100 channels. Use the 100-band PCA pickle if it exists (Houston,
    # 144->100); otherwise the dual spectrally resamples raw bands -> 100
    # (Trento 63, MUUFL 64). Leave None to trigger resampling.
    if getattr(args, "native_geometry", False) and not getattr(args, "native_pca_spat_path", None):
        conv100 = spec.pca_spat_path().replace(f"_{spec.spat_components}band.pkl", "_100band.pkl")
        args.native_pca_spat_path = conv100 if Path(conv100).exists() else None
    if getattr(args, "adapted_checkpoint", None) and args.adapted_checkpoint.lower() == "auto":
        args.adapted_checkpoint = str(
            Path(spec.adapt_ckpt_dir(args.spat_patch_k)) / "checkpoint.pth"
        )


def _resolve_device(args) -> str:
    """Resolve the eval device, honoring ``--device``/``--gpu``.

    ``--cpu`` or absent CUDA forces ``"cpu"`` (warning if a cuda device was
    explicitly requested). Otherwise use ``args.device`` (e.g. ``"cuda:1"``),
    falling back to ``"cuda"`` (= cuda:0).
    """
    requested = getattr(args, "gpu", None)
    requested = f"cuda:{requested}" if requested is not None else getattr(args, "device", None)

    if args.cpu or not torch.cuda.is_available():
        if requested and str(requested).startswith("cuda"):
            logger.warning(
                "Requested device %r but %s -> falling back to CPU.",
                requested,
                "--cpu set" if args.cpu else "CUDA unavailable",
            )
        return "cpu"
    return requested or "cuda"


def main(args: argparse.Namespace) -> dict | None:
    """Evaluate a HyperSIGMA configuration under the same few-shot protocol.

    The loop accumulates cosine *and* Euclidean confusion matrices in one
    pass; ``--distance-metric`` only selects which block is recorded as the
    run's primary metric. Paper Table 3 quotes the Euclidean block throughout;
    PAPER_CANON §8 D18 records the one cell whose ``eval_config.json`` says
    ``cosine`` even though its published value is the Euclidean sub-block.
    """
    device = _resolve_device(args)
    logger.info("Using device: %s", device)

    seeds = args.seeds if args.seeds else [args.seed]
    dataset_name = args.dataset.lower()
    _resolve_dataset_paths(args, dataset_name)
    logger.info(
        "Resolved paths: pca_spat=%s pca_stats=%s adapted=%s",
        args.pca_spat_path,
        args.pca_stats_path,
        args.adapted_checkpoint,
    )
    DatasetClass = DATASETS[dataset_name]
    dataset = DatasetClass(
        data_root=args.data_root,
        patch_size=args.patch_size,
        split=args.split,
        normalize=True,
    )
    logger.info("Loaded %s (%s): %d samples", dataset_name, args.split, len(dataset))

    model = load_model(
        dataset_name=dataset_name,
        pca_spat_path=args.pca_spat_path,
        spat_ckpt=args.spat_ckpt,
        spec_ckpt=args.spec_ckpt,
        spat_patch_k=args.spat_patch_k,
        adapted_checkpoint=args.adapted_checkpoint,
        mode=args.mode,
        temperature=args.temperature,
        prototype_mode=args.prototype_mode,
        distance_metric=args.distance_metric,
        device=device,
        pca_stats_path=getattr(args, "pca_stats_path", None),
        spat_resample_to=getattr(args, "spat_resample_to", None),
        native_geometry=getattr(args, "native_geometry", False),
        input_fit=getattr(args, "input_fit", "upscale"),
        pad_anchor=getattr(args, "pad_anchor", "center"),
        interp_mode=getattr(args, "interp_mode", "bicubic"),
        native_pca_spat_path=getattr(args, "native_pca_spat_path", None),
    )

    num_total = DATASET_SPECS[dataset_name]["num_classes"]

    last_results = None
    all_summaries = []
    for seed in seeds:
        set_seed(seed, deterministic=True)
        sampler = PatchedEpisodeSampler(
            dataset=dataset,
            n_way=args.n_way,
            k_shot=args.k_shot,
            k_query=args.k_query,
            num_episodes=args.num_episodes,
            seed=seed,
        )
        args.n_way = sampler.n_way
        results = evaluate(
            model,
            sampler,
            device,
            num_episodes=args.num_episodes,
            n_way=args.n_way,
            num_total_classes=num_total,
            temperature=args.temperature,
            num_example_episodes=args.num_example_episodes,
            max_tsne_samples=args.max_tsne_samples,
        )
        last_results = results
        all_summaries.append(
            {
                "seed": seed,
                "cosine": {k: results["cosine"][k] for k in ("OA", "AA", "Kappa")},
                "euclidean": {k: results["euclidean"][k] for k in ("OA", "AA", "Kappa")},
            }
        )

        primary = results[args.distance_metric]
        logger.info(
            "[seed %d] %s OA=%.2f+-%.2f AA=%.2f+-%.2f Kappa=%.2f+-%.2f (%d eps)",
            seed,
            args.distance_metric,
            primary["OA"]["mean"],
            primary["OA"]["ci_95"],
            primary["AA"]["mean"],
            primary["AA"]["ci_95"],
            primary["Kappa"]["mean"],
            primary["Kappa"]["ci_95"],
            primary["num_episodes"],
        )

    if args.output:
        _write_results_json(
            Path(args.output),
            last_results,
            dataset_name=dataset_name,
            args=args,
            seeds=seeds,
        )

    if not args.no_plots and args.output_dir is not None:
        plots_dir = Path(args.output_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Pull in example_episodes_data + aggregated_features from the
        # full results so the plot suite has feature-space inputs.
        shaped = _shape_for_plots(last_results[args.distance_metric])
        shaped["example_episodes_data"] = last_results["example_episodes_data"]
        shaped["aggregated_features"] = last_results["aggregated_features"]
        generate_plots(
            shaped,
            dataset_name,
            args.n_way,
            args.k_shot,
            plots_dir,
            args.distance_metric,
            args.prototype_mode,
            args.max_tsne_samples,
        )

    return last_results


_DEFAULT_ARGS = {
    "n_way": None,  # N-way: every class in the scene (PAPER_CANON §4)
    "k_shot": 5,  # §4
    # Table 3's protocol (phase-8 gate). This entry point serves Table 3, which
    # used 2000 episodes at k_query 100 - one cell excepted, §8 D18 - and every
    # paper eval passed split "all". These were 30 / 600 / "test" before the
    # gate, matching no published cell; no paper number depends on them because
    # every paper run passes them explicitly.
    "k_query": 100,
    "num_episodes": 2000,
    "data_root": "./data/raw",
    "split": "all",
    "patch_size": 11,
    "distance_metric": "euclidean",
    "temperature": 10.0,
    "prototype_mode": "mean_features",
    "seed": 42,
    "seeds": None,
    "cpu": False,
    "device": "cuda",
    "gpu": None,
    "output": None,
    "output_dir": None,
    "no_plots": False,
    "num_example_episodes": 3,
    "max_tsne_samples": 0,
    "spat_patch_k": 3,
    # None -> derived from `dataset` via the registry convention in main().
    "pca_spat_path": None,
    "pca_stats_path": None,
    "spat_ckpt": "checkpoints/hypersigma/spat-vit-base.pth",
    "spec_ckpt": "checkpoints/hypersigma/spec-vit-base.pth",
    "adapted_checkpoint": None,
    "mode": "fused",
    # PCA-100 variant: fixed spatial channel width; sub-target datasets resample up.
    "spat_resample_to": None,
    # Native-geometry single-branch ablation (default off -> existing behavior).
    "native_geometry": False,
    "input_fit": "upscale",
    "pad_anchor": "center",
    "interp_mode": "bicubic",
    "native_pca_spat_path": None,
}


def run_evaluation(dataset: str, **overrides: Any) -> dict | None:
    """Programmatic entry-point used by notebooks."""
    cfg = {**_DEFAULT_ARGS, "dataset": dataset, **overrides}
    args = argparse.Namespace(**cfg)
    return main(args)

#!/usr/bin/env python
"""Re-run the best HyperSIGMA cells with their eval features compressed to 128-d.

The question. PAPER_CANON Table 3's HyperSIGMA features are 768-d (spatial,
spectral) or 512-d (fused SEM); CoFFE's is 128-d (§2). Table 2/3's margins are
therefore measured across a 4-6x difference in feature width, and with K=5
supports a nearest-class-mean prototype in 768-d is estimated from five vectors
in a space where distances concentrate. This driver compresses the frozen
features to a target width *before* the prototypes are formed and re-runs the
identical protocol, so the margin can be re-read at matched capacity.

What it runs, per scene (the best published cell of each):

    houston  11x11 patch-native, PCA-100, joint+SEM adaptation, fused 512-d,  OA 67.48
    trento   64x64 backbone-native upscale, frozen, spat_pool 768-d,          OA 91.07
    muufl    64x64 backbone-native pad,     frozen, spat_pool 768-d,          OA 54.80

Method. The encoder is frozen, so each labelled patch has exactly one feature
vector: it is encoded once into a cache (``coffe.eval.feature_cache``) and the
episodes then index that cache (``coffe.eval.reduced_ncm``). Reducers come from
``coffe.eval.dim_reduction`` — unsupervised PCA (fit on the whole labelled pool
or on the MFT train split alone), Johnson-Lindenstrauss random projections over
five seeds, and, for the fused feature, the SEM's own stage blocks. None of
them sees a label.

Trust. Episodes are drawn from the same sampler at the same seed as the
published runs, so every variant is *paired* with the control episode by
episode, and the control is the published configuration at native width: if it
does not reproduce that cell's OA to ``--tolerance`` the run is marked FAILED,
because then the cache or the loop is wrong. ``tests/integration/
test_reduced_ncm_equivalence.py`` separately pins the cached loop against the
live one exactly.

This adds an ablation; it changes no published number. Results land under
``experiments/<--experiment>/`` (git-ignored) and the aggregate in
``results/hypersigma_dim_sweep.json``.

Usage:
    python scripts/experiments/run_hypersigma_dim_sweep.py --scenes all --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coffe.data.datasets import DATASET_REGISTRY
from coffe.data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from coffe.eval.dim_reduction import build_reducer, pca_family
from coffe.eval.episodic import CLASS_NAMES
from coffe.eval.feature_cache import encode_dataset, load_features, save_features
from coffe.eval.hypersigma import _resolve_dataset_paths, load_model
from coffe.eval.reduced_ncm import (
    METRIC_BLOCKS,
    build_episode_plan,
    paired_delta,
    run_episodes,
    summarize,
)
from coffe.runners.experiments import ExperimentLogger
from coffe.utils.seed import set_seed

logger = logging.getLogger("dim_sweep")

#: Protocol constants held at the published Table 3 values (PAPER_CANON §4, §8 D18).
K_SHOT = 5
K_QUERY = 100
NUM_EPISODES = 2000
SPLIT = "all"
PATCH_SIZE = 11
TEMPERATURE = 10.0
PROTOTYPE_MODE = "mean_features"
SEED = 42

#: Target widths. Filtered per cell to those below its native feature width.
DEFAULT_DIMS = (16, 32, 64, 128, 256, 512)
#: PAPER_CANON §4's seed list, reused for the random projections.
DEFAULT_RP_SEEDS = (42, 123, 456, 789, 1011)
#: Blocks carried into the aggregate's curves. `euclidean_l2` is the primary: it keeps
#: the unit-norm geometry the native features have (`_extract` L2-normalises them), which
#: a projection otherwise destroys. `euclidean` is the same metric on the raw projected
#: coordinates.
CURVE_BLOCKS = ("euclidean_l2", "euclidean")

#: The width CoFFE's eval feature has (PAPER_CANON §2) — the headline point.
COFFE_DIM = 128
#: CoFFE SimMIM token / HSI+LiDAR, Table 2, for context in the aggregate.
COFFE_PUBLISHED_OA = {"houston": 75.30, "trento": 94.19, "muufl": 67.83}


@dataclass(frozen=True)
class Cell:
    """One published Table 3 configuration, plus the number it must reproduce."""

    scene: str
    eval_name: str
    description: str
    feature_dim: int
    #: The paper's Table 3 value, for display (2 dp).
    published_oa: float
    published_ci: float
    #: The frozen run's own mean at full precision, read from `source`/results.json.
    #: The control is gated on *this*: gating on the rounded paper value would fold
    #: up to 0.005 pp of rounding into the drift the gate reports.
    reference_oa: float
    source: str
    mode: str
    native_geometry: bool = False
    input_fit: str = "upscale"
    spat_patch_k: int | None = 3
    pca_spat_path: str | None = None
    pca_stats_path: str | None = None
    adapted_checkpoint: str | None = None
    num_stages: int = 4
    #: Structural SEM maps only make sense for the fused feature.
    stage_maps: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


CELLS: dict[str, Cell] = {
    "houston": Cell(
        scene="houston",
        eval_name="houston_patchnative_joint_sem_fused512",
        description="11x11 patch-native, PCA-100, joint+SEM adaptation, fused SEM feature",
        feature_dim=512,
        published_oa=67.48,
        published_ci=0.11,
        reference_oa=67.478900,
        source=(
            "experiments/hypersigma_houston_pca100_joint_sem_run1/evaluations/"
            "hypersigma_houston_pca100_C15way_5shot_adapted_joint_sem_run2"
        ),
        mode="fused",
        native_geometry=False,
        spat_patch_k=3,
        pca_spat_path="checkpoints/hypersigma/pca_houston_100band.pkl",
        pca_stats_path="checkpoints/hypersigma/pca_houston_100band_stats.pkl",
        adapted_checkpoint=(
            "experiments/hypersigma_adapt_houston_pca100_joint_sem_run1/"
            "checkpoints/checkpoint_final.pth"
        ),
        stage_maps=True,
    ),
    "trento": Cell(
        scene="trento",
        eval_name="trento_backbonenative_upscale_frozen_spat768",
        description="64x64 backbone-native (upscale), frozen encoder, pooled SpatViT feature",
        feature_dim=768,
        published_oa=91.07,
        published_ci=0.09,
        reference_oa=91.072333,
        source="experiments/hypersigma_native_ablation_run1/evaluations/native_trento_spatial_upscale",
        mode="spat_pool",
        native_geometry=True,
        input_fit="upscale",
        spat_patch_k=None,
        adapted_checkpoint="none",
    ),
    "muufl": Cell(
        scene="muufl",
        eval_name="muufl_backbonenative_pad_frozen_spat768",
        description="64x64 backbone-native (pad), frozen encoder, pooled SpatViT feature",
        feature_dim=768,
        published_oa=54.80,
        published_ci=0.12,
        reference_oa=54.796682,
        source="experiments/hypersigma_native_ablation_run1/evaluations/native_muufl_spatial_pad",
        mode="spat_pool",
        native_geometry=True,
        input_fit="pad",
        spat_patch_k=None,
        adapted_checkpoint="none",
    ),
}


# ----------------------------------------------------------------------
# Encoder + cache
# ----------------------------------------------------------------------


def _resolved_paths(cell: Cell) -> SimpleNamespace:
    """Apply the evaluator's own path conventions to this cell's config."""
    args = SimpleNamespace(
        pca_spat_path=cell.pca_spat_path,
        pca_stats_path=cell.pca_stats_path,
        native_geometry=cell.native_geometry,
        native_pca_spat_path=None,
        adapted_checkpoint=cell.adapted_checkpoint,
        spat_patch_k=cell.spat_patch_k,
        spat_ckpt="checkpoints/hypersigma/spat-vit-base.pth",
        spec_ckpt="checkpoints/hypersigma/spec-vit-base.pth",
    )
    _resolve_dataset_paths(args, cell.scene)
    return args


def _cache_meta(cell: Cell, paths: SimpleNamespace, dataset) -> dict[str, Any]:
    return {
        "scene": cell.scene,
        "mode": cell.mode,
        "native_geometry": cell.native_geometry,
        "input_fit": cell.input_fit,
        "spat_patch_k": cell.spat_patch_k,
        "adapted_checkpoint": paths.adapted_checkpoint,
        "pca_spat_path": paths.pca_spat_path,
        "pca_stats_path": paths.pca_stats_path,
        "native_pca_spat_path": paths.native_pca_spat_path,
        "spat_ckpt": paths.spat_ckpt,
        "spec_ckpt": paths.spec_ckpt,
        "split": SPLIT,
        "patch_size": PATCH_SIZE,
        "num_samples": len(dataset),
        "feature_dim": cell.feature_dim,
    }


def build_cache(
    cell: Cell,
    dataset,
    *,
    device: str,
    cache_dir: Path,
    batch_size: int,
    refresh: bool = False,
) -> torch.Tensor:
    """Encode the scene once (or reuse a matching cache) and return ``[n, D]``."""
    paths = _resolved_paths(cell)
    expect = _cache_meta(cell, paths, dataset)
    cache_path = cache_dir / f"{cell.eval_name}"

    if not refresh:
        hit = load_features(cache_path, expect=expect)
        if hit is not None:
            return hit[0]

    logger.info("[%s] building encoder: %s", cell.scene, cell.description)
    model = load_model(
        dataset_name=cell.scene,
        pca_spat_path=paths.pca_spat_path,
        spat_ckpt=paths.spat_ckpt,
        spec_ckpt=paths.spec_ckpt,
        spat_patch_k=cell.spat_patch_k,
        adapted_checkpoint=paths.adapted_checkpoint,
        mode=cell.mode,
        temperature=TEMPERATURE,
        prototype_mode=PROTOTYPE_MODE,
        distance_metric="euclidean",
        device=device,
        pca_stats_path=paths.pca_stats_path,
        spat_resample_to=None,
        native_geometry=cell.native_geometry,
        input_fit=cell.input_fit,
        pad_anchor="center",
        interp_mode="bicubic",
        native_pca_spat_path=paths.native_pca_spat_path,
    )
    features = encode_dataset(model, dataset, device=device, batch_size=batch_size)
    if features.shape[1] != cell.feature_dim:
        raise RuntimeError(
            f"{cell.scene}: encoder produced {features.shape[1]}-d features, "
            f"expected {cell.feature_dim} (wrong mode or geometry?)"
        )
    save_features(cache_path, features, expect)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return features


# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------


def plan_variants(
    cell: Cell,
    features: torch.Tensor,
    *,
    dims: tuple[int, ...],
    rp_seeds: tuple[int, ...],
    n_train: int,
) -> list[Any]:
    """Build every reducer for this cell: PCA (two corpora), RPs, stage maps."""
    widths = tuple(d for d in dims if d < cell.feature_dim)
    reducers = []

    # One SVD per corpus, sliced to each width (see `pca_family`).
    reducers.extend(pca_family(features, widths, corpus="pool"))

    # `split="all"` concatenates Tr then Te (coffe/data/datasets/patched.py), so the
    # train split is the cache's leading rows; `run_cell` asserts that by labels.
    train_widths = tuple(d for d in widths if d <= n_train)
    if len(train_widths) < len(widths):
        logger.warning(
            "[%s] pca_train capped at d<=%d: the MFT train split has only %d features",
            cell.scene,
            max(train_widths, default=0),
            n_train,
        )
    reducers.extend(pca_family(features[:n_train], train_widths, corpus="train"))
    for dim in widths:
        for seed in rp_seeds:
            reducers.append(
                build_reducer("gaussian_rp", feature_dim=cell.feature_dim, dim=dim, seed=seed)
            )
    if cell.stage_maps:
        for stage in range(cell.num_stages):
            reducers.append(
                build_reducer(
                    "stage_block",
                    feature_dim=cell.feature_dim,
                    stage=stage,
                    num_stages=cell.num_stages,
                )
            )
        reducers.append(
            build_reducer("stage_mean", feature_dim=cell.feature_dim, num_stages=cell.num_stages)
        )
    return reducers


# ----------------------------------------------------------------------
# One cell
# ----------------------------------------------------------------------


def run_cell(
    cell: Cell,
    *,
    device: str,
    data_root: str,
    experiment: str,
    cache_dir: Path,
    dims: tuple[int, ...],
    rp_seeds: tuple[int, ...],
    num_episodes: int,
    batch_size: int,
    tolerance: float,
    refresh_cache: bool,
    control_only: bool,
    overwrite: bool,
) -> dict[str, Any]:
    spec = DATASET_REGISTRY[cell.scene]
    dataset_all = spec.patched_cls(
        data_root=data_root, patch_size=PATCH_SIZE, split=SPLIT, normalize=True
    )
    dataset_train = spec.patched_cls(
        data_root=data_root, patch_size=PATCH_SIZE, split="train", normalize=True
    )
    n_train = len(dataset_train)
    if not torch.equal(dataset_train.labels, dataset_all.labels[:n_train]):
        raise RuntimeError(
            f"{cell.scene}: split='all' does not lead with the train split, so the "
            "pca_train fit corpus would be the wrong rows of the cache"
        )
    del dataset_train
    logger.info(
        "[%s] %d labelled patches (%d in the MFT train split)",
        cell.scene,
        len(dataset_all),
        n_train,
    )

    features = build_cache(
        cell,
        dataset_all,
        device=device,
        cache_dir=cache_dir,
        batch_size=batch_size,
        refresh=refresh_cache,
    )

    # The published episode stream: same sampler, same seed, same constants.
    set_seed(SEED, deterministic=True)
    sampler = PatchedEpisodeSampler(
        dataset=dataset_all,
        n_way=None,
        k_shot=K_SHOT,
        k_query=K_QUERY,
        num_episodes=num_episodes,
        seed=SEED,
    )
    plan = build_episode_plan(sampler, num_episodes, seed=SEED)

    control_reducer = build_reducer("identity", feature_dim=cell.feature_dim)
    variants = (
        []
        if control_only
        else plan_variants(cell, features, dims=dims, rp_seeds=rp_seeds, n_train=n_train)
    )

    def _evaluate(reducer) -> tuple[dict[str, Any], dict[str, list[float]]]:
        reduced = reducer.to(device)(features.to(device))
        blocks = run_episodes(
            reduced,
            plan,
            num_total_classes=spec.num_classes,
            temperature=TEMPERATURE,
            device=device,
        )
        summaries = {name: summarize(block) for name, block in blocks.items()}
        oas = {name: block["episode_oas"] for name, block in blocks.items()}
        return summaries, oas

    started = time.perf_counter()
    control_summaries, control_oas = _evaluate(control_reducer)
    control_oa = control_summaries["euclidean"]["OA"]["mean"]
    delta = control_oa - cell.reference_oa
    passed = abs(delta) <= tolerance
    logger.info(
        "[%s] CONTROL euclidean OA %.4f vs frozen run %.4f (drift %+.5f pp; "
        "paper prints %.2f +- %.2f) -> %s",
        cell.scene,
        control_oa,
        cell.reference_oa,
        delta,
        cell.published_oa,
        cell.published_ci,
        "PASS" if passed else "FAIL",
    )

    variant_records = []
    for index, reducer in enumerate(variants, start=1):
        summaries, oas = _evaluate(reducer)
        record = {
            "label": reducer.label,
            "kind": reducer.kind,
            "dim": reducer.dim,
            "spec": reducer.spec,
            "blocks": {
                name: {key: summaries[name][key] for key in ("OA", "AA", "Kappa")}
                for name in summaries
            },
            "paired_vs_control": {
                name: paired_delta(control_oas[name], oas[name]) for name in summaries
            },
        }
        # Per-class detail only at the headline width, to keep the artifact readable.
        if reducer.dim == COFFE_DIM:
            record["per_class_euclidean_l2"] = summaries["euclidean_l2"]["per_class"]
        variant_records.append(record)
        logger.info(
            "[%s] %2d/%2d %-24s eucl %.2f (%+.2f pp) | eucl_l2 %.2f (%+.2f pp) | cos %.2f",
            cell.scene,
            index,
            len(variants),
            reducer.label,
            summaries["euclidean"]["OA"]["mean"],
            record["paired_vs_control"]["euclidean"]["delta_oa"],
            summaries["euclidean_l2"]["OA"]["mean"],
            record["paired_vs_control"]["euclidean_l2"]["delta_oa"],
            summaries["cosine"]["OA"]["mean"],
        )

    wall = time.perf_counter() - started

    # --- artifacts -----------------------------------------------------
    root = Path("experiments") / experiment
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# HyperSIGMA feature-width ablation\n\n"
            "Feature-cached re-runs of the best published HyperSIGMA cell per scene with the\n"
            "frozen eval feature linearly compressed before the prototypes are formed.\n"
            "Driver: `scripts/experiments/run_hypersigma_dim_sweep.py`.\n"
            "Every `results.json` here carries a `feature_reduction` key; the aggregators\n"
            "skip those, so nothing in this tree can stand in for a Table 3 cell.\n"
        )

    exp_logger = ExperimentLogger(experiments_root="experiments", repo_root=PROJECT_ROOT)
    eval_config = {
        "dataset": cell.scene,
        "description": cell.description,
        "mode": cell.mode,
        "native_geometry": cell.native_geometry,
        "input_fit": cell.input_fit,
        "spat_patch_k": cell.spat_patch_k,
        "adapted_checkpoint": _resolved_paths(cell).adapted_checkpoint,
        "distance_metric": "euclidean",
        "k_shot": K_SHOT,
        "k_query": K_QUERY,
        "num_episodes": num_episodes,
        "split": SPLIT,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "prototype_mode": PROTOTYPE_MODE,
        "feature_reduction": {
            "control": control_reducer.spec,
            "dims": list(dims),
            "rp_seeds": list(rp_seeds),
            "variants": len(variants),
        },
        "published_reference": {
            "OA": cell.published_oa,
            "ci_95": cell.published_ci,
            "frozen_run_OA": cell.reference_oa,
            "source": cell.source,
        },
    }
    eval_run = exp_logger.start_eval(experiment, cell.eval_name, eval_config, overwrite=overwrite)

    control_payload = {
        "model_type": "HyperSIGMADual",
        # Marks this tree as an ablation, and is what the aggregators skip on.
        "feature_reduction": control_reducer.spec,
        "dataset": cell.scene,
        "mode": cell.mode,
        "native_geometry": cell.native_geometry,
        "input_fit": cell.input_fit,
        "spat_patch_k": cell.spat_patch_k,
        "adapted_checkpoint": eval_config["adapted_checkpoint"],
        "split": SPLIT,
        "n_way": plan.n_way,
        "k_shot": K_SHOT,
        "k_query": K_QUERY,
        "seeds": [SEED],
        "num_episodes": control_summaries["euclidean"]["num_episodes"],
        "distance_metric": "euclidean",
        "temperature": TEMPERATURE,
        "prototype_mode": PROTOTYPE_MODE,
        **{name: control_summaries[name] for name in METRIC_BLOCKS},
        "OA": control_summaries["euclidean"]["OA"],
        "AA": control_summaries["euclidean"]["AA"],
        "Kappa": control_summaries["euclidean"]["Kappa"],
        "per_class": {str(k): v for k, v in control_summaries["euclidean"]["per_class"].items()},
        "class_names": CLASS_NAMES.get(cell.scene, []),
        "published_reference": eval_config["published_reference"],
        "control_check": {
            "reproduced_oa": control_oa,
            "reference_oa": cell.reference_oa,
            "reference_source": cell.source,
            "delta_oa": delta,
            "tolerance_pp": tolerance,
            "passed": bool(passed),
        },
    }
    eval_run.results_path.write_text(json.dumps(control_payload, indent=2, default=str))

    sweep = {
        "cell": {
            "scene": cell.scene,
            "eval_name": cell.eval_name,
            "description": cell.description,
            "feature_dim": cell.feature_dim,
            "mode": cell.mode,
            "native_geometry": cell.native_geometry,
            "input_fit": cell.input_fit,
            "adapted_checkpoint": eval_config["adapted_checkpoint"],
            "source": cell.source,
        },
        "protocol": {
            "k_shot": K_SHOT,
            "k_query": K_QUERY,
            "num_episodes": num_episodes,
            "n_way": plan.n_way,
            "split": SPLIT,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "labelled_patches": len(dataset_all),
            "train_split_patches": n_train,
        },
        "control": {
            "label": control_reducer.label,
            "dim": cell.feature_dim,
            "blocks": {
                name: {key: control_summaries[name][key] for key in ("OA", "AA", "Kappa")}
                for name in control_summaries
            },
            "published": {
                "OA": cell.published_oa,
                "ci_95": cell.published_ci,
                "frozen_run_OA": cell.reference_oa,
            },
            "check": control_payload["control_check"],
            "per_class_euclidean": control_summaries["euclidean"]["per_class"],
        },
        "coffe_published_oa": COFFE_PUBLISHED_OA.get(cell.scene),
        "variants": variant_records,
        "wallclock_seconds": round(wall, 1),
    }
    (eval_run.root / "sweep.json").write_text(json.dumps(sweep, indent=2, default=str))
    eval_run.finalize(
        results=control_summaries["euclidean"],
        control_check=control_payload["control_check"],
        variants=len(variant_records),
    )
    if not passed:
        eval_run.mark_failed(
            f"control OA {control_oa:.4f} differs from the frozen run's "
            f"{cell.reference_oa:.4f} by {delta:+.5f} pp (> {tolerance} pp)"
        )
    logger.info("[%s] wrote %s (%.1f s)", cell.scene, eval_run.root, wall)
    return sweep


# ----------------------------------------------------------------------
# Aggregate
# ----------------------------------------------------------------------


def write_aggregate(sweeps: list[dict[str, Any]], out_path: Path) -> Path:
    """Collapse the per-cell sweeps into one release-ready summary JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenes = {}
    for sweep in sweeps:
        scene = sweep["cell"]["scene"]
        rows: dict[str, dict[str, Any]] = {}
        for variant in sweep["variants"]:
            kind = variant["kind"]
            if kind == "pca":
                # The fit corpus is the axis; widths of one corpus form one curve.
                group = "pca_" + variant["spec"].get("fit_corpus", "")
            elif kind == "stage_block":
                # Averaging the four SEM stages would hide which one carries the signal.
                group = f"stage{variant['spec']['stage']}"
            else:
                # gaussian_rp collapses over its seeds, which is the point of the seeds.
                group = kind
            rows.setdefault(group, {})
            key = str(variant["dim"])
            entry = rows[group].setdefault(
                key, {block: {"oa": [], "delta": [], "p": []} for block in CURVE_BLOCKS}
            )
            for block in CURVE_BLOCKS:
                entry[block]["oa"].append(variant["blocks"][block]["OA"]["mean"])
                entry[block]["delta"].append(variant["paired_vs_control"][block]["delta_oa"])
                entry[block]["p"].append(variant["paired_vs_control"][block]["p_value"])
            if "explained_variance_ratio" in variant["spec"]:
                entry["explained_variance_ratio"] = variant["spec"]["explained_variance_ratio"]

        curves = {}
        for group, by_dim in rows.items():
            curves[group] = {
                dim: {
                    block: {
                        # `n_runs` > 1 only for gaussian_rp, where it is the seed count:
                        # `oa_std` is then the spread over projections, not over episodes.
                        "oa_mean": float(np.mean(vals[block]["oa"])),
                        "oa_std": float(np.std(vals[block]["oa"])),
                        "delta_oa_mean": float(np.mean(vals[block]["delta"])),
                        "max_p_value": float(np.max(vals[block]["p"])),
                        "n_runs": len(vals[block]["oa"]),
                    }
                    for block in CURVE_BLOCKS
                }
                | (
                    {"explained_variance_ratio": vals["explained_variance_ratio"]}
                    if "explained_variance_ratio" in vals
                    else {}
                )
                for dim, vals in sorted(by_dim.items(), key=lambda kv: int(kv[0]))
            }

        scenes[scene] = {
            "cell": sweep["cell"],
            "protocol": sweep["protocol"],
            "control": {
                "dim": sweep["control"]["dim"],
                "oa_euclidean": sweep["control"]["blocks"]["euclidean"]["OA"],
                "oa_euclidean_l2": sweep["control"]["blocks"]["euclidean_l2"]["OA"],
                "published": sweep["control"]["published"],
                "check": sweep["control"]["check"],
            },
            "coffe_published_oa": sweep["coffe_published_oa"],
            "curves": curves,
        }

    payload = {
        "what": (
            "HyperSIGMA's best published cell per scene, re-evaluated under PAPER_CANON §4's "
            "protocol with the frozen eval feature linearly compressed to a target width "
            "before the nearest-class-mean prototypes are formed."
        ),
        "metric_blocks": {
            "euclidean": "paper metric on the reduced feature as returned",
            "euclidean_l2": "reduced feature re-normalised to unit length (primary)",
            "cosine": "recorded for parity; scale-invariant, so identical for both",
        },
        "protocol": {
            "k_shot": K_SHOT,
            "k_query": K_QUERY,
            "num_episodes": NUM_EPISODES,
            "split": SPLIT,
            "seed": SEED,
            "paired": "variants share the control's episode stream; deltas are paired",
        },
        "coffe_reference": {
            "feature_dim": COFFE_DIM,
            "table_2_simmim_token_hsi_lidar_oa": COFFE_PUBLISHED_OA,
        },
        "scenes": scenes,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote aggregate -> %s", out_path)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _guard_out(out: Path, force: bool) -> None:
    """Refuse to clobber a committed artifact without an explicit ``--force``.

    ``results/hypersigma_dim_sweep.json`` is tracked, so re-running this driver
    would overwrite it in place. Checked before any work is done, so a mistaken
    invocation costs nothing — the same guard, for the same reason, as
    ``scripts/compile_results.py`` and the three other report builders the
    phase-6 gate fixed (`results/README.md`).
    """
    if out.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite {out}\n"
            "It is a committed artifact. Pass --force to regenerate it, or "
            "--aggregate-out PATH to write elsewhere (--aggregate-out none to skip)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenes", default="all", help="comma-separated scenes, or 'all'")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="./data/raw")
    parser.add_argument("--experiment", default="hypersigma_dimreduce_run1")
    parser.add_argument("--cache-dir", default=None, help="default: experiments/<exp>/features")
    parser.add_argument("--dims", default=",".join(str(d) for d in DEFAULT_DIMS))
    parser.add_argument("--rp-seeds", default=",".join(str(s) for s in DEFAULT_RP_SEEDS))
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--batch-size", type=int, default=256, help="patches per encoder pass")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="max |OA - published| the control may drift, in pp",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--control-only", action="store_true", help="gate the cache, run no variants"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild the aggregate from the per-cell sweep.json files; runs nothing",
    )
    parser.add_argument(
        "--aggregate-out", default="results/hypersigma_dim_sweep.json", help="'none' to skip"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite --aggregate-out if it exists (it is a committed artifact)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    aggregate_out = None if args.aggregate_out.lower() == "none" else Path(args.aggregate_out)
    if aggregate_out is not None:
        _guard_out(aggregate_out, args.force)

    scenes = list(CELLS) if args.scenes == "all" else [s.strip() for s in args.scenes.split(",")]
    unknown = set(scenes) - set(CELLS)
    if unknown:
        parser.error(f"unknown scene(s) {sorted(unknown)}; known: {sorted(CELLS)}")

    if args.aggregate_only:
        root = Path("experiments") / args.experiment / "evaluations"
        wanted = {CELLS[scene].eval_name for scene in scenes}
        found = [
            json.loads(path.read_text())
            for path in sorted(root.glob("*/sweep.json"))
            if path.parent.name in wanted
        ]
        if not found:
            parser.error(f"no sweep.json under {root} for scenes {scenes}")
        if aggregate_out is None:
            parser.error("--aggregate-only with --aggregate-out none would write nothing")
        write_aggregate(found, aggregate_out)
        return

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable -> falling back to CPU (this will be slow)")
        device = "cpu"

    cache_dir = Path(args.cache_dir or Path("experiments") / args.experiment / "features")
    dims = tuple(int(d) for d in args.dims.split(",") if d)
    rp_seeds = tuple(int(s) for s in args.rp_seeds.split(",") if s)

    sweeps = []
    for scene in scenes:
        sweeps.append(
            run_cell(
                CELLS[scene],
                device=device,
                data_root=args.data_root,
                experiment=args.experiment,
                cache_dir=cache_dir,
                dims=dims,
                rp_seeds=rp_seeds,
                num_episodes=args.num_episodes,
                batch_size=args.batch_size,
                tolerance=args.tolerance,
                refresh_cache=args.refresh_cache,
                control_only=args.control_only,
                overwrite=args.overwrite,
            )
        )

    if aggregate_out is not None and sweeps:
        write_aggregate(sweeps, aggregate_out)


if __name__ == "__main__":
    main()

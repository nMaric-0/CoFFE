"""Single source of truth for the multi-seed statistical-significance experiment.

A "cell" is one (group, dataset, variant) configuration of a model. Each cell is
trained under 5 seeds, fresh to 700 epochs with an identical schedule, then its
epoch-700 checkpoint is evaluated with one fixed protocol and reported as
mean +/- std +/- 95% CI (t, df=4) across seeds.

Groups (model families):
  enhanced      CoFFE, with LiDAR; variants spatial/spectral/both  (v1, done)
  hsi_only      CoFFE with use_aux=false; variants spatial/spectral/both
  enhanced_mae  CoFFE objective=mae; variants lidar/hsi_only
  mft_mae       faithful mft_original baseline, MAE objective
  mft_spatial   faithful mft_original baseline, spatial-masking objective

Post-paper ablation groups (opt-in via --groups; not in GROUP_ORDER):
  aux_token_simmim  CoFFE-AuxToken, SimMIM token, Houston only
  aux_token_mae     CoFFE-AuxToken, MAE, Houston only

For `enhanced` and `hsi_only` the per-variant mask ratios are CLONED at runtime
from a canonical seed-42 run's pretrain overrides. The other groups use a
self-contained base config, so no per-variant override is needed.

Used by: sig_pretrain_worker.py, sig_eval_worker.py,
         run_significance_experiment.py, aggregate_significance.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

# ---------------------------------------------------------------------------
# Experiment knobs
# ---------------------------------------------------------------------------

#: Five independent seeds per cell -> mean +/- std +/- 95% CI (t, df=4).
SEEDS: list[int] = [42, 123, 456, 789, 1011]

#: Every run trains fresh to this many epochs with an identical schedule.
EPOCHS: int = 700

#: Forced so checkpoint_epoch_700.pth is always written (700 % 100 == 0).
SAVE_INTERVAL: int = 100

#: Eval the epoch-700 checkpoint.
EVAL_EPOCH: int = EPOCHS

#: Eval subdir name created under each experiment.
EVAL_NAME: str = "sig_eval_epoch700"

#: GPUs the global scheduler may use, max MAX_PARALLEL_PER_GPU runs on each.
GPUS: list[str] = ["cuda:0", "cuda:2", "cuda:3"]
MAX_PARALLEL_PER_GPU: int = 2

DATASETS: list[str] = ["houston", "trento", "muufl"]

#: Front-load the slowest dataset so the bottleneck starts earliest.
DATASET_COST_ORDER: list[str] = ["muufl", "trento", "houston"]

#: Canonical seed-42 dirs for the v1 `enhanced` group (with LiDAR).
_ENHANCED_CANONICAL: dict[tuple, str] = {
    ("houston", "spatial"): "houston_enhanced_spatial_run1",
    ("houston", "spectral"): "houston_enhanced_spectral_run2",
    ("houston", "both"): "houston_enhanced_spec_spat_combined",
    ("trento", "spatial"): "trento_enhanced_spat_run1",
    ("trento", "spectral"): "trento_enhanced_spectral_run1",
    ("trento", "both"): "trento_enhanced_spectral_spatial_run1",
    ("muufl", "spatial"): "muufl_enhanced_spatial_run1",
    ("muufl", "spectral"): "muufl_enhanced_spectral_run1",
    ("muufl", "both"): "muufl_enhanced_spectral_spatial_run2",
}

#: enhanced/hsi_only variant -> suffix used in the *_no_lidar canonical dir names.
_NO_LIDAR_SUFFIX = {"spatial": "spatial", "spectral": "spectral", "both": "spectral_spatial"}


# ---------------------------------------------------------------------------
# Group registry
# ---------------------------------------------------------------------------


class Group:
    def __init__(
        self,
        name: str,
        variants: list[str],
        base_config: Callable[[str, str], str],
        experiment_name: Callable[[str, str, int], str],
        *,
        clone_mask: bool = False,
        canonical_dir: Callable[[str, str], str] | None = None,
        datasets: list[str] | None = None,
    ):
        self.name = name
        self.variants = variants
        self.base_config = base_config
        self._experiment_name = experiment_name
        self.clone_mask = clone_mask
        self._canonical_dir = canonical_dir
        #: Scenes this group covers. The five paper groups cover all three;
        #: single-scene ablation groups narrow it (their configs exist for that
        #: scene only, and `cells()` must not ask for the others).
        self.datasets = list(datasets) if datasets is not None else list(DATASETS)

    def experiment_name(self, dataset: str, variant: str, seed: int) -> str:
        return self._experiment_name(dataset, variant, seed)

    def canonical_dir(self, dataset: str, variant: str) -> str | None:
        return self._canonical_dir(dataset, variant) if self._canonical_dir else None


GROUPS: dict[str, Group] = {
    "enhanced": Group(
        "enhanced",
        ["spatial", "spectral", "both"],
        base_config=lambda ds, v: f"configs/coffe/{ds}_simmim.yaml",
        experiment_name=lambda ds, v, s: f"{ds}_enhanced_{v}_seed{s}",
        clone_mask=True,
        canonical_dir=lambda ds, v: _ENHANCED_CANONICAL[(ds, v)],
    ),
    "hsi_only": Group(
        "hsi_only",
        ["spatial", "spectral", "both"],
        base_config=lambda ds, v: f"configs/coffe/{ds}_simmim_hsi.yaml",
        experiment_name=lambda ds, v, s: f"{ds}_hsi_only_{v}_seed{s}",
        clone_mask=True,
        canonical_dir=lambda ds, v: f"{ds}_enhanced_{_NO_LIDAR_SUFFIX[v]}_no_lidar",
    ),
    "enhanced_mae": Group(
        "enhanced_mae",
        ["lidar", "hsi_only"],
        base_config=lambda ds, v: (
            f"configs/coffe/{ds}_mae.yaml" if v == "lidar" else f"configs/coffe/{ds}_mae_hsi.yaml"
        ),
        experiment_name=lambda ds, v, s: f"{ds}_enhanced_mae_{v}_seed{s}",
        clone_mask=False,
    ),
    "mft_mae": Group(
        "mft_mae",
        ["mae"],
        base_config=lambda ds, v: f"configs/mft/{ds}_mae.yaml",
        experiment_name=lambda ds, v, s: f"mft_original_{ds}_mae_seed{s}",
        clone_mask=False,
    ),
    "mft_spatial": Group(
        "mft_spatial",
        ["spatial"],
        base_config=lambda ds, v: f"configs/mft/{ds}_simmim_token.yaml",
        experiment_name=lambda ds, v, s: f"mft_original_{ds}_spatial_seed{s}",
        clone_mask=False,
    ),
    # --- post-paper ablation: LiDAR as one separate encoder token ---
    # docs/ablations/AUX_TOKEN_ABLATION.md. Houston only, and deliberately NOT
    # in GROUP_ORDER: `--groups aux_token_*` opts in, so a bare run of the
    # significance experiment still reproduces exactly the paper's five groups.
    "aux_token_simmim": Group(
        "aux_token_simmim",
        ["simmim_token"],
        base_config=lambda ds, v: f"configs/ablation/{ds}_aux_token_simmim_token.yaml",
        experiment_name=lambda ds, v, s: f"{ds}_aux_token_simmim_token_seed{s}",
        clone_mask=False,
        datasets=["houston"],
    ),
    "aux_token_mae": Group(
        "aux_token_mae",
        ["mae"],
        base_config=lambda ds, v: f"configs/ablation/{ds}_aux_token_mae.yaml",
        experiment_name=lambda ds, v, s: f"{ds}_aux_token_mae_seed{s}",
        clone_mask=False,
        datasets=["houston"],
    ),
}

#: Default order groups are processed / reported in.
GROUP_ORDER: list[str] = ["enhanced", "hsi_only", "enhanced_mae", "mft_mae", "mft_spatial"]


def cells(groups: list[str] | None = None):
    """Yield (group, dataset, variant) for the selected groups (all by default)."""
    for gname in groups or GROUP_ORDER:
        g = GROUPS[gname]
        for dataset in g.datasets:
            for variant in g.variants:
                yield gname, dataset, variant


def runs(groups: list[str] | None = None, seeds: list[int] | None = None):
    """Yield (group, dataset, variant, seed, experiment_name) for all selected runs."""
    seeds = seeds or SEEDS
    for gname, dataset, variant in cells(groups):
        g = GROUPS[gname]
        for seed in seeds:
            yield gname, dataset, variant, seed, g.experiment_name(dataset, variant, seed)


def canonical_pretrain_overrides(group: str, dataset: str, variant: str) -> dict[str, Any]:
    """Return the cloned ``pretrain`` override block for a cell.

    For clone_mask groups (enhanced, hsi_only) this reads the canonical seed-42
    run's ``overrides.pretrain`` (mask ratios etc.) so the new seeds use the same
    recipe. For self-contained groups it returns ``{}`` (the base config is
    authoritative).
    """
    g = GROUPS[group]
    if not g.clone_mask:
        return {}
    canon = g.canonical_dir(dataset, variant)
    meta_path = EXPERIMENTS_ROOT / canon / "pretrain_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Canonical metadata not found for ({group}, {dataset}, {variant}): {meta_path}"
        )
    with meta_path.open() as f:
        meta = json.load(f)
    pretrain = (meta.get("overrides", {}) or {}).get("pretrain", {})
    if not pretrain:
        raise ValueError(
            f"No pretrain overrides in {meta_path} for ({group}, {dataset}, {variant})"
        )
    return dict(pretrain)


def eval_params(dataset: str) -> dict[str, Any]:
    """Shared eval protocol (matches the existing enhanced/MFT evals).

    ``n_way`` is omitted so the evaluator defaults to all classes for the
    dataset (Houston 15 / Trento 6 / MUUFL 11). Architecture (model name,
    attention_type, mlp_dim, use_aux) is pulled from the run's frozen
    pretrain_config.yaml by coffe.runners.eval_runner, so this is
    family-agnostic. ``use_projection`` is the exception: it is set here,
    because the pretrain configs keep the head on (it is a pretraining part)
    and every paper eval discarded it (PAPER_CANON §4).
    """
    return {
        "dataset": dataset,
        "k_shot": 5,
        "k_query": 100,
        "num_episodes": 1000,
        "distance_metric": "euclidean",
        "split": "all",
        "use_projection": False,
        "temperature": 10.0,
        "prototype_mode": "mean_features",
        "seed": 42,
        # Skip per-eval plot generation (confusion matrix / per-class / episode
        # KDE / example episodes). Plotting is CPU-bound and dominates wall time
        # (~54 min vs ~1 min for the actual GPU eval), while the significance
        # report only consumes results.json metrics — which are written BEFORE
        # the plotting gate, so the numbers are identical.
        "no_plots": True,
    }

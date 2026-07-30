"""Single source of truth for the BEST-CONFIG rerun of the main experiment.

Re-runs the main significance experiment with the architecture trio that won the
Houston-spatial combination study applied to every cell:

    BEST: dropout=0, embed_dim=256, num_layers=4   (model overrides)

The trio is masking-agnostic, so each cell keeps its NATIVE masking recipe
(cloned from the v1/v2 canonical runs) and only the model architecture changes.

Cells = {enhanced (HSI+LiDAR), hsi_only} x {spatial, spectral, both} x
{houston, trento, muufl} = 18 cells x 5 seeds = 90 runs. Trained to 1000 epochs
(save_interval 200), evaluated at epoch 1000. Reported as mean +/- std +/- 95% CI.

Reuses scripts.sig_significance_config for the group definitions, canonical mask-
ratio cloning, base configs, and eval protocol. Standalone otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import sig_significance_config as sig  # noqa: E402

EXPERIMENTS_ROOT = sig.EXPERIMENTS_ROOT

#: Reuse the two modality groups from the significance harness.
GROUPS_USED: List[str] = ["enhanced", "hsi_only"]   # enhanced == HSI+LiDAR
DATASETS: List[str] = sig.DATASETS
VARIANTS: List[str] = ["spatial", "spectral", "both"]

SEEDS: List[int] = [42, 123, 456, 789, 1011]
EPOCHS: int = 1000
SAVE_INTERVAL: int = 200
EVAL_EPOCH: int = 1000
EVAL_NAME: str = "best_eval_epoch1000"

#: Per-GPU process caps. cuda:0 capped at 1 (shared), others 2. Use the
#: orchestrator's --gpus flag to temporarily exclude a GPU without editing this.
GPU_SLOTS: Dict[str, int] = {"cuda:0": 1, "cuda:1": 2, "cuda:2": 2, "cuda:3": 2}
GPUS: List[str] = list(GPU_SLOTS)
TOTAL_SLOTS: int = sum(GPU_SLOTS.values())

#: The winning architecture config from the combination study (model section).
BEST_MODEL_OVERRIDES: Dict[str, Any] = {
    "dropout": 0.0,
    "embed_dim": 256,
    "num_layers": 4,
}


def experiment_name(group: str, dataset: str, variant: str, seed: int) -> str:
    # e.g. houston_enhanced_spatial_bestcfg_seed42 / houston_hsi_only_both_bestcfg_seed42
    return f"{dataset}_{group}_{variant}_bestcfg_seed{seed}"


def base_config(group: str, dataset: str, variant: str) -> str:
    return sig.GROUPS[group].base_config(dataset, variant)


def cells(groups: Optional[List[str]] = None):
    for group in (groups or GROUPS_USED):
        for dataset in DATASETS:
            for variant in VARIANTS:
                yield group, dataset, variant


def runs(groups: Optional[List[str]] = None, seeds: Optional[List[int]] = None):
    for group, dataset, variant in cells(groups):
        for seed in (seeds or SEEDS):
            yield group, dataset, variant, seed, experiment_name(group, dataset, variant, seed)


def build_overrides(group: str, dataset: str, variant: str, seed: int, device: str) -> Dict[str, Any]:
    """Native masking recipe (cloned from canonical) + forced schedule + the
    best-config model trio + seed/device."""
    pretrain = sig.canonical_pretrain_overrides(group, dataset, variant)  # mask ratios, batch, warmup
    pretrain["epochs"] = EPOCHS
    pretrain["save_interval"] = SAVE_INTERVAL
    return {
        "pretrain": pretrain,
        "model": dict(BEST_MODEL_OVERRIDES),
        "hardware": {"seed": seed, "device": device},
    }


def eval_params(dataset: str) -> Dict[str, Any]:
    return sig.eval_params(dataset)  # euclidean, k5/q100/1000ep, use_projection False, no_plots True

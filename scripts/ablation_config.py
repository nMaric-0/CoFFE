"""Single source of truth for the OFAT hyperparameter ablation.

Ablates the main model's Houston / enhanced / SPATIAL / HSI+LiDAR recipe one
factor at a time. Each variant changes ONE hyperparameter vs a shared baseline
(cloned from the v1 Houston-spatial run), trained for 3 seeds to 1000 epochs
(checkpoints every 200 -> 200/400/600/800/1000), and evaluated at epoch 800 and
1000. Reported as mean +/- std +/- 95% CI (t, df=2) per variant per epoch.

Standalone — independent of sig_significance_config.py so the live significance
run is never touched.

Used by: ablation_pretrain_worker.py, ablation_eval_worker.py,
         run_ablation.py, aggregate_ablation.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
DATASET = "houston"
BASE_CONFIG = "configs/pretrain/houston_pretrain_enhanced.yaml"

#: Canonical run whose pretrain overrides define the baseline recipe
#: (band=0, spatial=0.75, batch=128, warmup=100) — matches the v1 Houston spatial.
CANONICAL_BASELINE_DIR = "houston_enhanced_spatial_run1"

SEEDS: List[int] = [42, 123, 456]
EPOCHS: int = 1000
SAVE_INTERVAL: int = 200               # -> checkpoints at 200/400/600/800/1000
EVAL_EPOCHS: List[int] = [800, 1000]

GPUS: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
MAX_PARALLEL_PER_GPU: int = 2

EXP_PREFIX = "ablation"


def eval_name_for_epoch(epoch: int) -> str:
    return f"abl_eval_epoch{epoch}"


# ---------------------------------------------------------------------------
# Variant grid — (slug, partial nested override merged onto the baseline).
# Exactly one field differs from baseline per variant. Section keys are
# "pretrain" or "model" so they land in the right config block.
# ---------------------------------------------------------------------------
VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    ("baseline", {}),

    # --- reconstruction loss (new recon_loss axis; huber==smooth_l1 beta=1, omitted) ---
    ("loss_l1", {"pretrain": {"recon_loss": "l1"}}),
    ("loss_smooth_l1", {"pretrain": {"recon_loss": "smooth_l1"}}),

    # --- Gaussian centering of the recon loss (baseline sigma=1.0) ---
    ("center_off", {"pretrain": {"recon_center_sigma": None}}),
    ("center_s0p5", {"pretrain": {"recon_center_sigma": 0.5}}),
    ("center_s2", {"pretrain": {"recon_center_sigma": 2.0}}),
    ("center_s3", {"pretrain": {"recon_center_sigma": 3.0}}),

    # --- spatial masking amount (baseline 0.75) ---
    ("mask0p50", {"pretrain": {"spatial_mask_ratio": 0.50}}),
    ("mask0p60", {"pretrain": {"spatial_mask_ratio": 0.60}}),
    ("mask0p85", {"pretrain": {"spatial_mask_ratio": 0.85}}),
    ("mask0p90", {"pretrain": {"spatial_mask_ratio": 0.90}}),

    # --- optimization (baseline lr 1.5e-5, wd 0.05, warmup 100, batch 128) ---
    ("lr5e5", {"pretrain": {"lr": 5.0e-5}}),
    ("lr1e4", {"pretrain": {"lr": 1.0e-4}}),
    ("wd0", {"pretrain": {"weight_decay": 0.0}}),
    ("wd0p1", {"pretrain": {"weight_decay": 0.1}}),
    ("warmup50", {"pretrain": {"warmup_epochs": 50}}),
    ("warmup200", {"pretrain": {"warmup_epochs": 200}}),
    ("batch64", {"pretrain": {"batch_size": 64}}),
    ("batch256", {"pretrain": {"batch_size": 256}}),

    # --- architecture (baseline layers2/heads2/embed128/lambda0.5/dropout0.1/decoder256/proj1) ---
    ("layers4", {"model": {"num_layers": 4}}),
    ("layers6", {"model": {"num_layers": 6}}),
    ("heads4", {"model": {"num_heads": 4}}),
    ("heads8", {"model": {"num_heads": 8}}),
    ("embed64", {"model": {"embed_dim": 64}}),
    ("embed256", {"model": {"embed_dim": 256}}),
    ("lambda1", {"model": {"lambda_factor": 1.0}}),
    ("lambda2", {"model": {"lambda_factor": 2.0}}),
    ("dropout0", {"model": {"dropout": 0.0}}),
    ("dropout0p2", {"model": {"dropout": 0.2}}),
    ("decoder128", {"pretrain": {"decoder_hidden_dim": 128}}),
    ("decoder512", {"pretrain": {"decoder_hidden_dim": 512}}),
    ("proj2", {"model": {"proj_num_layers": 2}}),
]

SLUGS: List[str] = [s for s, _ in VARIANTS]
_VARIANT_MAP: Dict[str, Dict[str, Any]] = {s: o for s, o in VARIANTS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def baseline_pretrain_overrides() -> Dict[str, Any]:
    """Clone the canonical Houston-spatial run's pretrain overrides."""
    meta_path = EXPERIMENTS_ROOT / CANONICAL_BASELINE_DIR / "pretrain_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Canonical baseline metadata not found: {meta_path}")
    with meta_path.open() as f:
        meta = json.load(f)
    pretrain = (meta.get("overrides", {}) or {}).get("pretrain", {})
    if not pretrain:
        raise ValueError(f"No pretrain overrides in {meta_path}")
    return dict(pretrain)


def variant_override(slug: str) -> Dict[str, Any]:
    """Public accessor: deep copy of a single variant's partial override
    (used by the combination study to compose multiple factors)."""
    return copy.deepcopy(_VARIANT_MAP[slug])


def experiment_name(slug: str, seed: int) -> str:
    return f"{EXP_PREFIX}_{slug}_seed{seed}"


def build_overrides(slug: str, seed: int, device: str) -> Dict[str, Any]:
    """Full override dict for one (slug, seed, device): baseline + variant +
    forced epochs/save_interval/seed/device."""
    if slug not in _VARIANT_MAP:
        raise KeyError(f"unknown slug {slug!r}")
    overrides: Dict[str, Any] = {"pretrain": baseline_pretrain_overrides()}
    overrides = _deep_merge(overrides, copy.deepcopy(_VARIANT_MAP[slug]))
    overrides["pretrain"]["epochs"] = EPOCHS
    overrides["pretrain"]["save_interval"] = SAVE_INTERVAL
    overrides = _deep_merge(overrides, {"hardware": {"seed": seed, "device": device}})
    return overrides


def runs(slugs: Optional[List[str]] = None, seeds: Optional[List[int]] = None):
    """Yield (slug, seed, experiment_name)."""
    for slug in (slugs or SLUGS):
        for seed in (seeds or SEEDS):
            yield slug, seed, experiment_name(slug, seed)


def eval_params() -> Dict[str, Any]:
    """Same protocol as the significance eval (Houston, all classes, no plots)."""
    return {
        "dataset": DATASET,
        "k_shot": 5,
        "k_query": 100,
        "num_episodes": 1000,
        "distance_metric": "euclidean",
        "split": "all",
        "use_projection": False,
        "temperature": 10.0,
        "prototype_mode": "mean_features",
        "seed": 42,
        "no_plots": True,
    }

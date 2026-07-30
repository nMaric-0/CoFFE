"""Single source of truth for the COMBINATION study.

Builds combinations of the best-performing single-factor ablations (the OFAT
winners) on the Houston / enhanced / spatial / HSI+LiDAR baseline. Each combo
applies several winning factors at once; trained 3 seeds to 1000 epochs and
evaluated at epoch 800 and 1000, same protocol as the OFAT ablation.

Pool (the 6 strong OFAT winners, each the better value of its parameter):
    decoder512, dropout0, mask0p90, heads8, layers4, embed256

For a 6-factor pool the requested tiers collapse to subset sizes 2..6 (deduped):
    anchor  : sizes {6, 5}  -> all-best (1) + leave-one-out (6)        = 7 combos
    pairs   : size  {2}                                                = 15
    triples : size  {3}     (== leave-three-out)                       = 20
    l2o     : size  {4}     (leave-two-out)                            = 15
Total 57 combos x 3 seeds = 171 runs. (singletons are the OFAT runs; the empty
set is the baseline — both already trained.)

Reuses ablation_config for the baseline recipe and the per-factor overrides, so
a combo == baseline + each member factor's single-field change.

Standalone harness; GPUs cuda:2,3 only.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import ablation_config as abl  # noqa: E402

EXPERIMENTS_ROOT = abl.EXPERIMENTS_ROOT
BASE_CONFIG = abl.BASE_CONFIG
SEEDS = abl.SEEDS                 # [42, 123, 456]
EPOCHS = abl.EPOCHS               # 1000
SAVE_INTERVAL = abl.SAVE_INTERVAL # 200
EVAL_EPOCHS = abl.EVAL_EPOCHS     # [800, 1000]
eval_name_for_epoch = abl.eval_name_for_epoch
eval_params = abl.eval_params

#: Per-GPU concurrency caps (cuda:0 is shared -> only 1 process there).
GPU_SLOTS: Dict[str, int] = {"cuda:0": 1, "cuda:2": 2, "cuda:3": 2}
GPUS: List[str] = list(GPU_SLOTS)
TOTAL_SLOTS: int = sum(GPU_SLOTS.values())

#: Pool of OFAT-winning factors (slugs valid in ablation_config.VARIANTS),
#: in canonical order used for naming.
POOL: List[str] = ["decoder512", "dropout0", "mask0p90", "heads8", "layers4", "embed256"]

#: Short tag per factor for compact experiment-dir names.
TAGS: Dict[str, str] = {
    "decoder512": "dec512",
    "dropout0": "drop0",
    "mask0p90": "m90",
    "heads8": "h8",
    "layers4": "L4",
    "embed256": "e256",
}

#: Stage -> set of subset sizes (informative-first). L3O==triples and LOO==size5
#: so this covers every size-2..6 subset exactly once across stages.
STAGES: Dict[str, List[int]] = {
    "anchor": [6, 5],   # all-best + leave-one-out
    "pairs": [2],
    "triples": [3],     # also == leave-three-out
    "l2o": [4],         # leave-two-out
}
STAGE_ORDER: List[str] = ["anchor", "pairs", "triples", "l2o"]


def _canonical(factors) -> Tuple[str, ...]:
    """Order a set of factors by POOL order (stable naming/identity)."""
    fset = set(factors)
    return tuple(f for f in POOL if f in fset)


def combo_name(factors) -> str:
    tags = "-".join(TAGS[f] for f in _canonical(factors))
    return f"combo_{tags}"


def experiment_name(factors, seed: int) -> str:
    return f"{combo_name(factors)}_seed{seed}"


def stage_combos(stage: str) -> List[Tuple[str, ...]]:
    """Deduped canonical factor-tuples for one stage."""
    sizes = STAGES[stage]
    seen, out = set(), []
    for k in sizes:
        for c in combinations(POOL, k):
            cc = _canonical(c)
            if cc not in seen:
                seen.add(cc)
                out.append(cc)
    return out


def all_combos(stages: Optional[List[str]] = None) -> List[Tuple[str, ...]]:
    seen, out = set(), []
    for st in (stages or STAGE_ORDER):
        for cc in stage_combos(st):
            if cc not in seen:
                seen.add(cc)
                out.append(cc)
    return out


def runs(stages: Optional[List[str]] = None, seeds: Optional[List[int]] = None):
    """Yield (factors_tuple, seed, experiment_name)."""
    for factors in all_combos(stages):
        for seed in (seeds or SEEDS):
            yield factors, seed, experiment_name(factors, seed)


def build_overrides(factors, seed: int, device: str) -> Dict[str, Any]:
    """baseline + each member factor's single-field override + forced
    epochs/save_interval/seed/device."""
    overrides: Dict[str, Any] = {"pretrain": abl.baseline_pretrain_overrides()}
    for f in _canonical(factors):
        overrides = abl._deep_merge(overrides, abl.variant_override(f))
    overrides["pretrain"]["epochs"] = EPOCHS
    overrides["pretrain"]["save_interval"] = SAVE_INTERVAL
    overrides = abl._deep_merge(overrides, {"hardware": {"seed": seed, "device": device}})
    return overrides

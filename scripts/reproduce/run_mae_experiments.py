#!/usr/bin/env python
"""Full MAE-pretraining experiment matrix, run sequentially.

Matrix:  {houston, trento, muufl}  ×  {lidar, no_lidar}   (6 cells)

Original-MAE recipe (He et al. 2022): remove 75% of the pixel tokens, encode the
visible 25% (+ CLS) only, and a transformer decoder reconstructs the masked
tokens. The MLP projection head is removed (use_projection=false). Each cell
pretrains the CoFFE encoder with the per-dataset MAE config
(lidar -> *_mae.yaml with use_aux=true, no_lidar -> *_mae_hsi.yaml with
use_aux=false), then runs the paper's few-shot evaluation (Euclidean
nearest-class-mean on the frozen encoder) on the same dataset. Both stages go through the standard runners (coffe.runners.pretrain_runner /
coffe.runners.eval_runner), so each lands in experiments/<name>/ and the evaluator
auto-loads the architecture — including use_aux and use_projection=false — from
the saved pretrain_config.yaml.

Experiment names:
    <dataset>_mae_lidar      and      <dataset>_mae_no_lidar

Run on physical GPU 2:
    bash scripts/reproduce/run_mae_experiments.sh
or equivalently:
    CUDA_VISIBLE_DEVICES=2 python scripts/reproduce/run_mae_experiments.py

Useful flags:
    --datasets houston trento     # subset of datasets
    --lidar                       # only the HSI+LiDAR cells
    --no-lidar                    # only the HSI-only cells   (default: both)
    --epochs   1500               # force epoch count on every cell (0 = keep
                                  #   each config's own value: 3000 Houston /
                                  #   1500 Trento+MUUFL)
    --overwrite                   # reuse/clobber existing experiment dirs
    --skip-eval                   # pretrain only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coffe.runners.eval_runner import run_evaluation
from coffe.runners.pretrain_runner import run_pretrain

# Physical GPU is pinned by CUDA_VISIBLE_DEVICES=2 (see the .sh wrapper), which
# remaps it to logical index 0. Keep this as cuda:0; do not hard-code cuda:2.
DEVICE = "cuda:0"

# dataset -> {lidar setting: MAE config path}
CONFIGS = {
    "houston": {
        "lidar": "configs/coffe/houston_mae.yaml",
        "no_lidar": "configs/coffe/houston_mae_hsi.yaml",
    },
    "trento": {
        "lidar": "configs/coffe/trento_mae.yaml",
        "no_lidar": "configs/coffe/trento_mae_hsi.yaml",
    },
    "muufl": {
        "lidar": "configs/coffe/muufl_mae.yaml",
        "no_lidar": "configs/coffe/muufl_mae_hsi.yaml",
    },
}

# lidar setting -> (experiment-name suffix, human label)
LIDAR_SETTINGS = {
    "lidar": ("lidar", "HSI+LiDAR (use_aux=true)"),
    "no_lidar": ("no_lidar", "HSI-only (use_aux=false)"),
}

# Evaluation params matching the current result runs (run_hsi_only_experiments.py).
# use_aux is NOT set here: it auto-loads from the experiment's pretrain_config.yaml.
# Evaluated checkpoint epoch, per dataset (PAPER_CANON §8 D17).
#
# Passed EXPLICITLY. Until the phase-7 gate this call omitted `epoch`, which
# made `coffe.runners.eval_runner.find_checkpoint` fall through to a
# lexicographic filename sort — and that sort is what actually selected the
# paper's mid-schedule checkpoints ("950" > "1500" as strings). The sort is
# numeric now, so omitting `epoch` would evaluate the FINAL checkpoint and no
# longer reproduce the published cell.
#
# Houston saves every 50 epochs -> 950; Trento/MUUFL every 25 -> 975. Both are
# reachable from the committed configs' `save_interval`.
EVAL_EPOCH = {"houston": 950, "trento": 975, "muufl": 975}

EVAL_PARAMS = dict(
    split="all",
    k_shot=5,
    k_query=100,
    num_episodes=1000,
    distance_metric="euclidean",
    temperature=10.0,
    prototype_mode="mean_features",
    pool_sigma=None,
    use_projection=False,  # discard the projection head at eval (MAE has none)
    seed=42,
    num_example_episodes=1,
    max_tsne_samples=100,
    device=DEVICE,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--datasets", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--lidar", action="store_true", help="Only the HSI+LiDAR cells.")
    ap.add_argument(
        "--no-lidar", dest="no_lidar", action="store_true", help="Only the HSI-only cells."
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Epoch count applied to every cell. 0 (default) keeps "
        "each config's own value (3000 Houston / 1500 others).",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Reuse/overwrite an existing experiment directory."
    )
    ap.add_argument("--skip-eval", action="store_true", help="Pretrain only; skip evaluation.")
    args = ap.parse_args()

    # Default: both settings. --lidar / --no-lidar narrow it; passing both = both.
    if args.lidar and not args.no_lidar:
        settings = ["lidar"]
    elif args.no_lidar and not args.lidar:
        settings = ["no_lidar"]
    else:
        settings = ["lidar", "no_lidar"]

    cells = [(ds, s) for ds in args.datasets for s in settings]
    print(
        f"Planned {len(cells)} cells on {DEVICE} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}):"
    )
    for ds, s in cells:
        print(f"  - {ds}_mae_{LIDAR_SETTINGS[s][0]}")
    print()

    summary = []
    for i, (ds, setting) in enumerate(cells, 1):
        suffix, label = LIDAR_SETTINGS[setting]
        exp_name = f"{ds}_mae_{suffix}"
        eval_name = f"{exp_name}_eval"
        header = f"[{i}/{len(cells)}] {exp_name}  ({label})"
        print("=" * 80)
        print(header)
        print("=" * 80)

        overrides = {"hardware": {"device": DEVICE}}
        if args.epochs > 0:
            overrides["pretrain"] = {"epochs": args.epochs}

        t0 = time.time()
        try:
            run_pretrain(
                name=exp_name,
                description=(
                    f"MAE pretraining (mask_ratio=0.75, transformer decoder, "
                    f"use_projection=false) of CoFFE on {ds}; {label}."
                ),
                config=CONFIGS[ds][setting],
                overrides=overrides,
                overwrite=args.overwrite,
            )
            pretrain_secs = time.time() - t0

            eval_done = False
            if not args.skip_eval:
                run_evaluation(
                    experiment_name=exp_name,
                    eval_name=eval_name,
                    epoch=EVAL_EPOCH[ds],
                    eval_params={"dataset": ds, **EVAL_PARAMS},
                    overwrite=args.overwrite,
                )
                eval_done = True

            summary.append((exp_name, "OK", time.time() - t0))
            print(
                f"--> {exp_name}: pretrain {pretrain_secs / 60:.1f} min, "
                f"eval {'done' if eval_done else 'skipped'}."
            )
        except Exception as e:  # keep the matrix going if one cell fails
            summary.append((exp_name, f"FAILED: {e}", time.time() - t0))
            print(f"!!! {exp_name} FAILED after {(time.time() - t0) / 60:.1f} min:")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    for name, status, secs in summary:
        print(f"  {name:<32} {status:<14} ({secs / 60:.1f} min)")
    failures = [s for s in summary if not s[1].startswith("OK")]
    print(f"\n{len(summary) - len(failures)}/{len(summary)} cells OK, {len(failures)} failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

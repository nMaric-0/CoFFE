#!/usr/bin/env python
"""Original-MFT baseline ("Spatial" masking+loss): pretrain -> few-shot eval.

Matrix:  {houston, trento, muufl}  ×  {multimodal HSI+LiDAR}   (3 cells)

Companion to scripts/reproduce/run_mft_original_mae_experiments.py. Same faithful original
MFT encoder, but pretrained with the team's "Spatial version" masking+loss
(SimMIM-style in-place spatial token masking at 75%, 2-layer MLP decoder
reconstructing the full HSI+LiDAR per-pixel vector, center-weighted MSE on masked
entries — see coffe/pretrain/mft_spatial_mae.py) instead of standard MAE. The training
schedule and evaluation are identical to the MAE variant, so the two ablations
differ ONLY in the pretraining objective.

Both stages go through the standard runners (lib.pretrain_runner /
lib.eval_runner); each lands in experiments/<name>/ and the evaluator auto-loads
the architecture (model.name=mft_original, attention_type, use_projection=false)
from the saved pretrain_config.yaml.

Experiment names:
    mft_original_<dataset>_spatial

Run on physical GPU 2:
    bash scripts/reproduce/run_mft_original_spatial_experiments.sh
or equivalently:
    CUDA_VISIBLE_DEVICES=2 python scripts/reproduce/run_mft_original_spatial_experiments.py

Useful flags:
    --datasets houston trento     # subset of datasets
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

# dataset -> multimodal "Spatial" config path
CONFIGS = {
    "houston": "configs/mft/houston_simmim_token.yaml",
    "trento": "configs/mft/trento_simmim_token.yaml",
    "muufl": "configs/mft/muufl_simmim_token.yaml",
}

# Evaluation params — identical to the MAE variant and to how the Spatial
# CoFFE runs were evaluated (euclidean, k5/q100/1000 episodes, pool_sigma=None,
# use_projection=False). Architecture auto-loads from pretrain_config.yaml.
EVAL_PARAMS = dict(
    split="all",
    k_shot=5,
    k_query=100,
    num_episodes=1000,
    distance_metric="euclidean",
    temperature=10.0,
    prototype_mode="mean_features",
    pool_sigma=None,
    use_projection=False,
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
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Appended to each experiment name (e.g. '_faithful') so "
        "a re-run lands in NEW experiment folders instead of "
        "overwriting the existing mft_original_<ds>_spatial dirs.",
    )
    args = ap.parse_args()

    cells = list(args.datasets)
    print(
        f"Planned {len(cells)} cells on {DEVICE} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}):"
    )
    for ds in cells:
        print(f"  - mft_original_{ds}_spatial")
    print()

    summary = []
    for i, ds in enumerate(cells, 1):
        exp_name = f"mft_original_{ds}_spatial{args.suffix}"
        eval_name = f"{exp_name}_eval"
        print("=" * 80)
        print(f"[{i}/{len(cells)}] {exp_name}  (original MFT, Spatial masking, HSI+LiDAR)")
        print("=" * 80)

        overrides = {"hardware": {"device": DEVICE}}
        if args.epochs > 0:
            overrides["pretrain"] = {"epochs": args.epochs}

        t0 = time.time()
        try:
            run_pretrain(
                name=exp_name,
                description=(
                    f"'Spatial' masking+loss pretraining (spatial_mask_ratio=0.75, "
                    f"MLP decoder, center-weighted MSE, no projection head) of the "
                    f"original MFT (channel tokenization, mCrossPA) on {ds}; HSI+LiDAR."
                ),
                config=CONFIGS[ds],
                overrides=overrides,
                overwrite=args.overwrite,
            )
            pretrain_secs = time.time() - t0

            eval_done = False
            if not args.skip_eval:
                run_evaluation(
                    experiment_name=exp_name,
                    eval_name=eval_name,
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
        print(f"  {name:<34} {status:<14} ({secs / 60:.1f} min)")
    failures = [s for s in summary if not s[1].startswith("OK")]
    print(f"\n{len(summary) - len(failures)}/{len(summary)} cells OK, {len(failures)} failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Full HSI-only (LiDAR removed) experiment matrix, run sequentially.

Matrix:  {houston, trento, muufl}  ×  {spectral-only, spatial-only, combined}
Masking: spectral (band_mask_ratio) = 0.85, spatial (spatial_mask_ratio) = 0.75
         (matching the ratios used in the current HSI+LiDAR results).

Each cell pretrains the CoFFE encoder with use_aux=false (the
aux/LiDAR bands are never used) using the per-dataset *_simmim_hsi.yaml config,
then runs few-shot cosine/euclidean evaluation on the same dataset. Both stages
go through the standard experiment runners (lib.pretrain_runner /
lib.eval_runner), so each lands in experiments/<name>/ and the evaluator
auto-loads the architecture — including use_aux=false — from the saved
pretrain_config.yaml.

Experiment names mirror the current (HSI+LiDAR) runs with "_no_lidar" appended:
    <dataset>_enhanced_<regime>_no_lidar

Run on physical GPU 2:
    bash scripts/reproduce/run_hsi_only_experiments.sh
or equivalently:
    CUDA_VISIBLE_DEVICES=2 python scripts/reproduce/run_hsi_only_experiments.py

Useful flags:
    --datasets houston trento     # subset of datasets
    --regimes  spectral combined  # subset of masking regimes
    --epochs   1500               # override epoch count for every cell
    --overwrite                   # reuse/clobber existing experiment dirs
    --skip-eval                   # pretrain only
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coffe.runners.pretrain_runner import run_pretrain
from coffe.runners.eval_runner import run_evaluation

# Physical GPU is pinned by CUDA_VISIBLE_DEVICES=2 (see the .sh wrapper), which
# remaps it to logical index 0. Keep this as cuda:0; do not hard-code cuda:2.
DEVICE = "cuda:0"

# Masking ratios matching the current results.
SPECTRAL_RATIO = 0.85
SPATIAL_RATIO = 0.75

# regime key -> (band_mask_ratio, spatial_mask_ratio, human label)
REGIMES = {
    "spectral":         (SPECTRAL_RATIO, 0.0,           "spectral-only"),
    "spatial":          (0.0,            SPATIAL_RATIO,  "spatial-only"),
    "spectral_spatial": (SPECTRAL_RATIO, SPATIAL_RATIO, "combined spectral+spatial"),
}

# dataset -> HSI-only pretrain config
CONFIGS = {
    "houston": "configs/coffe/houston_simmim_hsi.yaml",
    "trento":  "configs/coffe/trento_simmim_hsi.yaml",
    "muufl":   "configs/coffe/muufl_simmim_hsi.yaml",
}

# Evaluation params matching the current HSI+LiDAR result runs. use_aux is NOT
# set here: it auto-loads (as false) from the experiment's pretrain_config.yaml.
EVAL_PARAMS = dict(
    split="all",
    k_shot=5,
    k_query=100,
    num_episodes=1000,
    distance_metric="euclidean",
    temperature=10.0,
    prototype_mode="mean_features",
    pool_sigma=None,
    use_projection=False,   # discard the projection head at eval time
    seed=42,
    num_example_episodes=1,
    max_tsne_samples=100,
    device=DEVICE,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES),
                    choices=list(REGIMES))
    ap.add_argument("--epochs", type=int, default=1500,
                    help="Epoch count applied to every cell (matches current "
                         "result runs). Use 0 to keep each config's own value.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reuse/overwrite an existing experiment directory.")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Pretrain only; skip evaluation.")
    args = ap.parse_args()

    cells = [(ds, rg) for ds in args.datasets for rg in args.regimes]
    print(f"Planned {len(cells)} cells on {DEVICE} "
          f"(CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'unset')}):")
    for ds, rg in cells:
        print(f"  - {ds}_enhanced_{rg}_no_lidar")
    print()

    summary = []
    for i, (ds, regime) in enumerate(cells, 1):
        band, spatial, label = REGIMES[regime]
        exp_name = f"{ds}_enhanced_{regime}_no_lidar"
        eval_name = f"{exp_name}_eval"
        header = f"[{i}/{len(cells)}] {exp_name}  ({label}: band={band}, spatial={spatial})"
        print("=" * 80)
        print(header)
        print("=" * 80)

        overrides = {
            "pretrain": {
                "band_mask_ratio": band,
                "spatial_mask_ratio": spatial,
            },
            "hardware": {"device": DEVICE},
        }
        if args.epochs > 0:
            overrides["pretrain"]["epochs"] = args.epochs

        t0 = time.time()
        try:
            run_pretrain(
                name=exp_name,
                description=(
                    f"HSI-only (LiDAR removed, use_aux=false) CoFFE on "
                    f"{ds}; {label} masking (band={band}, spatial={spatial}). "
                    f"Ablation mirroring {ds}_enhanced_{regime} with LiDAR removed."
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
            print(f"--> {exp_name}: pretrain {pretrain_secs/60:.1f} min, "
                  f"eval {'done' if eval_done else 'skipped'}.")
        except Exception as e:  # keep the matrix going if one cell fails
            summary.append((exp_name, f"FAILED: {e}", time.time() - t0))
            print(f"!!! {exp_name} FAILED after {(time.time()-t0)/60:.1f} min:")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    for name, status, secs in summary:
        print(f"  {name:<45} {status:<12} ({secs/60:.1f} min)")
    failures = [s for s in summary if not s[1].startswith("OK")]
    print(f"\n{len(summary) - len(failures)}/{len(summary)} cells OK, "
          f"{len(failures)} failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Re-run all six MFT "faithful" few-shot evaluations against the epoch-1500
# checkpoint instead of the epoch-950 one that the original runs used.
#
# Why a re-run is needed: lib.eval_runner.find_checkpoint() picks the "latest"
# checkpoint by lexicographically sorting checkpoint_epoch_*.pth. With training
# carried out to 1500 epochs, "checkpoint_epoch_950" sorts AFTER
# "checkpoint_epoch_1500" as a string, so epoch=None silently selected 950.
# This script pins epoch=1500 explicitly. The checkpoints already exist; only
# evaluation is rerun (no pretraining).
#
# Results land in NEW eval folders suffixed "_ep1500" so the existing
# epoch-950 evaluations are preserved for comparison:
#     experiments/mft_original_<ds>_<variant>_faithful/evaluations/
#         mft_original_<ds>_<variant>_faithful_eval_ep1500/
#
# GPU: physical GPU 2 is pinned via CUDA_VISIBLE_DEVICES (remapped to logical
# cuda:0, which the Python driver targets). Change the value to retarget.
#
#   bash scripts/rerun_mft_faithful_eval_ep1500.sh                 # all 6
#   bash scripts/rerun_mft_faithful_eval_ep1500.sh muufl_spatial   # subset
#   DATASETS="muufl_spatial houston_mae" bash scripts/rerun_mft_faithful_eval_ep1500.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
EPOCH="${EPOCH:-1500}"

# Cells to run: <dataset>_<variant>. Override via positional args or $DATASETS.
ALL_CELLS=(
  houston_mae   trento_mae   muufl_mae
  houston_spatial trento_spatial muufl_spatial
)
if [ "$#" -gt 0 ]; then
  CELLS=("$@")
elif [ -n "${DATASETS:-}" ]; then
  # shellcheck disable=SC2206
  CELLS=(${DATASETS})
else
  CELLS=("${ALL_CELLS[@]}")
fi

echo "Re-running ${#CELLS[@]} MFT faithful eval(s) at epoch ${EPOCH} on "
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (-> logical cuda:0)."
printf '  - mft_original_%s_faithful\n' "${CELLS[@]}"
echo

CELLS_STR="${CELLS[*]}" EPOCH="${EPOCH}" exec python - <<'PY'
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(Path.cwd()))

from lib.eval_runner import run_evaluation

DEVICE = "cuda:0"  # physical GPU pinned by CUDA_VISIBLE_DEVICES in the wrapper
EPOCH = int(os.environ["EPOCH"])
CELLS = os.environ["CELLS_STR"].split()

# Identical to scripts/run_mft_original_{mae,spatial}_experiments.py. The model
# architecture (name=mft_original, attention_type, use_projection=false)
# auto-loads from each experiment's saved pretrain_config.yaml.
EVAL_PARAMS = dict(
    split="all",
    k_shot=5,
    k_query=100,
    num_episodes=1000,
    distance_metric="euclidean",
    temperature=10.0,
    prototype_mode="mean_features",
    pool_sigma=None,
    use_projection=False,   # original MFT has no projection head
    seed=42,
    num_example_episodes=1,
    max_tsne_samples=100,
    device=DEVICE,
)

summary = []
for i, cell in enumerate(CELLS, 1):
    try:
        dataset, variant = cell.split("_", 1)
    except ValueError:
        print(f"!!! bad cell '{cell}' (expected <dataset>_<variant>); skipping")
        summary.append((cell, "BAD_CELL", 0.0))
        continue

    exp_name = f"mft_original_{dataset}_{variant}_faithful"
    eval_name = f"{exp_name}_eval_ep{EPOCH}"
    print("=" * 80)
    print(f"[{i}/{len(CELLS)}] {exp_name}  ->  eval @ epoch {EPOCH}")
    print("=" * 80)

    t0 = time.time()
    try:
        run_evaluation(
            experiment_name=exp_name,
            eval_name=eval_name,
            epoch=EPOCH,
            eval_params={"dataset": dataset, **EVAL_PARAMS},
            overwrite=True,  # reuse the _ep1500 eval dir on a repeat run
        )
        summary.append((exp_name, "OK", time.time() - t0))
        print(f"--> {exp_name}: eval done in {(time.time()-t0)/60:.1f} min.")
    except Exception as e:
        summary.append((exp_name, f"FAILED: {e}", time.time() - t0))
        print(f"!!! {exp_name} FAILED after {(time.time()-t0)/60:.1f} min:")
        traceback.print_exc()

print("\n" + "=" * 80)
print(f"RUN SUMMARY (epoch {EPOCH})")
print("=" * 80)
for name, status, secs in summary:
    print(f"  {name:<40} {status:<16} ({secs/60:.1f} min)")
failures = [s for s in summary if not s[1].startswith("OK")]
print(f"\n{len(summary) - len(failures)}/{len(summary)} cells OK, {len(failures)} failed.")
sys.exit(1 if failures else 0)
PY

#!/usr/bin/env bash
# Run the full MAE-pretraining experiment matrix sequentially on physical GPU 2.
# CUDA_VISIBLE_DEVICES=2 pins that GPU and remaps it to logical cuda:0, which is
# what the Python driver targets. Change the value below to retarget a free GPU.
#
#   bash scripts/reproduce/run_mae_experiments.sh                       # full 6-cell matrix
#   bash scripts/reproduce/run_mae_experiments.sh --datasets trento     # pass-through args
#   bash scripts/reproduce/run_mae_experiments.sh --no-lidar            # HSI-only cells only
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=2

exec python scripts/reproduce/run_mae_experiments.py "$@"

#!/usr/bin/env bash
# Run the full HSI-only (LiDAR removed) experiment matrix sequentially on
# physical GPU 2. CUDA_VISIBLE_DEVICES=2 pins that GPU and remaps it to logical
# cuda:0, which is what the Python driver targets.
#
#   bash scripts/run_hsi_only_experiments.sh                  # full matrix
#   bash scripts/run_hsi_only_experiments.sh --datasets trento  # pass-through args
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=2

exec python scripts/run_hsi_only_experiments.py "$@"

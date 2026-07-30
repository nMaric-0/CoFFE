#!/usr/bin/env bash
# Run the original-MFT "Spatial" masking+loss matrix (pretrain -> few-shot eval)
# sequentially on physical GPU 2. CUDA_VISIBLE_DEVICES=2 pins that GPU and remaps
# it to logical cuda:0, which is what the Python driver targets. Change the value
# below to retarget a free GPU.
#
#   bash scripts/run_mft_original_spatial_experiments.sh                    # all 3 datasets
#   bash scripts/run_mft_original_spatial_experiments.sh --datasets trento  # pass-through args
#   bash scripts/run_mft_original_spatial_experiments.sh --skip-eval        # pretrain only
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=2

exec python scripts/run_mft_original_spatial_experiments.py "$@"

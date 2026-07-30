#!/usr/bin/env bash
# HyperSIGMA spatial-branch-only adaptation + evaluation, 100-band variant,
# for all three datasets (houston, trento, muufl).
#
# Spatial branch is fed 100 channels (vs the 3-band headline) while keeping the
# adapted 11x11 / patch-3 geometry. Only the SpatViT pieces are adapted
# (adapt_mode=spatial_only) and evaluation pools SpatViT features
# (mode=spat_pool) -- SpecViT/SEM are unused. This is the
# evaluate_hypersigma_pca100.ipynb recipe with mode="spatial_only".
#
# Per dataset (decided from the registry band count):
#   houston (144 bands): PCA->100 (fitted below).
#   trento (63) / muufl (64): spectral resample raw bands -> 100 (no PCA).
#
# Each run lands under experiments/<name>/ (config, metadata, logs,
# results.json, plots), so scripts/aggregate_experiment_results.py picks it up.
#
# Usage:
#   bash scripts/run_hypersigma_spatial_pca100.sh                 # all three datasets
#   DEVICE=cuda:1 EPOCHS=2000 bash scripts/run_hypersigma_spatial_pca100.sh
#   EPOCHS=20 bash scripts/run_hypersigma_spatial_pca100.sh houston   # smoke test
#
# Adaptation is Level-2: the pretrained SpatViT body (trained on 64x64/patch-8)
# stays frozen; only patch_embed.proj + pos_embed (forced to change by the new
# 11x11/patch-3 input) + the MAE decoder train. Schedule (lr 1e-5, batch 128)
# mirrors evaluate_hypersigma_pca100.ipynb.
#
# Env vars:
#   DEVICE      device for adapt + eval (default: cuda). e.g. cuda:1 / cpu
#   EPOCHS      adaptation epochs per dataset (default: 2000)
#   LR          adaptation learning rate (default: 1e-5)
#   BATCH_SIZE  adaptation batch size (default: 128)

set -euo pipefail

# --- Settings ---------------------------------------------------------------
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-2000}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PYTHON="${PYTHON:-python}"

# Datasets: positional args override the default of all three.
if [[ "$#" -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=(houston trento muufl)
fi

# --- Locate repo root (this script lives in <repo>/scripts) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CKPT_DIR="checkpoints/hypersigma"

echo "=============================================================="
echo "HyperSIGMA spatial-only pca100 — datasets: ${DATASETS[*]}"
echo "  device=${DEVICE}  epochs=${EPOCHS}  lr=${LR}  batch_size=${BATCH_SIZE}  repo=${REPO_ROOT}"
echo "=============================================================="

# --- Step 1: ensure pretrained SpatViT/SpecViT checkpoints ------------------
if [[ ! -f "${CKPT_DIR}/spat-vit-base.pth" || ! -f "${CKPT_DIR}/spec-vit-base.pth" ]]; then
    echo "[setup] Pretrained HyperSIGMA checkpoints missing — downloading ..."
    bash scripts/download_hypersigma_checkpoints.sh "${CKPT_DIR}"
else
    echo "[setup] Pretrained HyperSIGMA checkpoints present — skipping download."
fi

# --- Step 2: fit houston PCA->100 (trento/muufl resample, no fit needed) -----
if printf '%s\n' "${DATASETS[@]}" | grep -qx houston; then
    if [[ ! -f "${CKPT_DIR}/pca_houston_100band.pkl" ]]; then
        echo "[setup] Fitting houston PCA->100 (writes .pkl + _stats.pkl) ..."
        "${PYTHON}" scripts/fit_pca_hypersigma.py --dataset houston --n-components 100
    else
        echo "[setup] houston PCA->100 already present — skipping fit."
    fi
fi

# --- Step 3: adapt + evaluate each dataset ----------------------------------
for ds in "${DATASETS[@]}"; do
    echo
    echo "--------------------------------------------------------------"
    echo ">>> ${ds}: spatial_only adaptation + spat_pool evaluation (100 bands)"
    echo "--------------------------------------------------------------"
    "${PYTHON}" scripts/run_hypersigma_spatial_pca100.py \
        --dataset "${ds}" \
        --device "${DEVICE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --batch-size "${BATCH_SIZE}"
done

echo
echo "=============================================================="
echo "Done. Results written under:"
for ds in "${DATASETS[@]}"; do
    echo "  experiments/hypersigma_${ds}_pca100_spatial_only_run1/evaluations/"
done
echo "Aggregate with: ${PYTHON} scripts/aggregate_experiment_results.py"
echo "=============================================================="

#!/usr/bin/env bash
#
# HyperSIGMA native-geometry SEM-tuning with PADDED inputs, for all three datasets.
#
# Keeps the ORIGINAL pretrained encoders frozen (SpatViT 64x64/patch-8/100ch,
# SpecViT 64x64) and MAE-tunes ONLY the fusion path (adapt_mode=sem_only), but
# fits the 11x11 patch to 64x64 by centered ZERO-PADDING (input_fit=pad) instead
# of bicubic upscaling. Then runs the few-shot fused evaluation on the adapted
# checkpoint -- same eval params as the upscale baseline (hypersigma_native_sem_run1)
# so the two are directly comparable.
#
# Per dataset:
#   1. adapt:  scripts/adapt_hypersigma.py   --config configs/hypersigma/<ds>_backbonenative_pad_sem_only.yaml
#   2. eval:   scripts/evaluate_hypersigma.py --native-geometry --input-fit pad ...
#
# Outputs:
#   checkpoints/hypersigma_adapted/<ds>_native_sem_pad/checkpoint_final.pth
#   experiments/hypersigma_native_sem_pad_run1/evaluations/native_sem_pad_<ds>/results.json
#
# Usage:
#   bash scripts/reproduce/run_native_sem_pad_experiments.sh                  # all three, GPU 2
#   GPU=0 bash scripts/reproduce/run_native_sem_pad_experiments.sh            # pick a GPU
#   DATASETS="trento muufl" bash scripts/reproduce/run_native_sem_pad_experiments.sh   # subset
#   SKIP_ADAPT=1 bash scripts/reproduce/run_native_sem_pad_experiments.sh     # eval only (reuse checkpoints)
#   SKIP_EVAL=1  bash scripts/reproduce/run_native_sem_pad_experiments.sh     # adapt only
#
set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/../.."

# --- knobs (all overridable from the environment) -------------------------
GPU="${GPU:-2}"                  # physical GPU index; pinned via CUDA_VISIBLE_DEVICES
PY="${PYTHON:-python}"           # python interpreter
DATASETS="${DATASETS:-houston trento muufl}"
EXPERIMENT="${EXPERIMENT:-hypersigma_native_sem_pad_run1}"
SKIP_ADAPT="${SKIP_ADAPT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# Eval params -- mirror the upscale baseline (experiments/hypersigma_native_sem_run1)
# so pad-vs-upscale is an apples-to-apples comparison.
K_SHOT="${K_SHOT:-5}"
K_QUERY="${K_QUERY:-100}"
NUM_EPISODES="${NUM_EPISODES:-2000}"
SPLIT="${SPLIT:-all}"
DISTANCE_METRIC="${DISTANCE_METRIC:-euclidean}"
TEMPERATURE="${TEMPERATURE:-10.0}"

# Pin the GPU: configs use device "cuda" and the eval uses --device cuda, so both
# land on the chosen physical GPU once it's the only one visible.
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=============================================================="
echo " HyperSIGMA native-geometry SEM-tuning -- PADDED inputs (11x11 -> 64x64, centered zero-pad)"
echo "   datasets        : ${DATASETS}"
echo "   GPU (CUDA_VIS..): ${CUDA_VISIBLE_DEVICES}"
echo "   experiment dir  : experiments/${EXPERIMENT}"
echo "   eval            : k_shot=${K_SHOT} k_query=${K_QUERY} episodes=${NUM_EPISODES} split=${SPLIT} metric=${DISTANCE_METRIC}"
echo "   skip_adapt=${SKIP_ADAPT}  skip_eval=${SKIP_EVAL}"
echo "=============================================================="

# Houston native geometry needs a 144->100 spatial PCA. Trento/MUUFL (<100 bands)
# spectrally resample on the fly, so they need nothing here.
HOUSTON_PCA="checkpoints/hypersigma/pca_houston_100band.pkl"

declare -a RESULT_PATHS=()

for ds in ${DATASETS}; do
  echo
  echo "##############  ${ds}  ######################################"

  config="configs/hypersigma/${ds}_backbonenative_pad_sem_only.yaml"
  ckpt_dir="checkpoints/hypersigma_adapted/${ds}_native_sem_pad"
  ckpt="${ckpt_dir}/checkpoint_final.pth"
  eval_dir="experiments/${EXPERIMENT}/evaluations/native_sem_pad_${ds}"
  results="${eval_dir}/results.json"

  if [ ! -f "${config}" ]; then
    echo "ERROR: missing config ${config}" >&2
    exit 1
  fi

  # ---- 1. adaptation (sem_only, frozen native-geometry encoders, padded input)
  if [ "${SKIP_ADAPT}" != "1" ]; then
    if [ "${ds}" = "houston" ] && [ ! -f "${HOUSTON_PCA}" ]; then
      echo "[${ds}] fitting 144->100 spatial PCA (one-off): ${HOUSTON_PCA}"
      "${PY}" scripts/fit_pca_hypersigma.py --dataset houston --n-components 100
    fi
    echo "[${ds}] ADAPT -> ${ckpt_dir}"
    "${PY}" scripts/adapt_hypersigma.py --config "${config}"
  else
    echo "[${ds}] SKIP_ADAPT=1 -> reusing ${ckpt}"
  fi

  # ---- 2. few-shot fused evaluation on the adapted checkpoint
  if [ "${SKIP_EVAL}" != "1" ]; then
    if [ ! -f "${ckpt}" ]; then
      echo "ERROR: adapted checkpoint not found: ${ckpt}" >&2
      echo "       (run adaptation first, or unset SKIP_ADAPT)" >&2
      exit 1
    fi
    echo "[${ds}] EVAL -> ${results}"
    "${PY}" scripts/evaluate_hypersigma.py \
      --dataset "${ds}" \
      --mode fused \
      --native-geometry \
      --input-fit pad \
      --pad-anchor center \
      --adapted-checkpoint "${ckpt}" \
      --k-shot "${K_SHOT}" \
      --k-query "${K_QUERY}" \
      --num-episodes "${NUM_EPISODES}" \
      --split "${SPLIT}" \
      --distance-metric "${DISTANCE_METRIC}" \
      --temperature "${TEMPERATURE}" \
      --num-example-episodes 1 \
      --max-tsne-samples 100 \
      --device cuda \
      --output "${results}" \
      --output-dir "${eval_dir}/plots"
    RESULT_PATHS+=("${results}")
  else
    echo "[${ds}] SKIP_EVAL=1 -> no evaluation"
  fi
done

echo
echo "=============================================================="
echo " Done. Results:"
if [ "${#RESULT_PATHS[@]}" -eq 0 ]; then
  echo "   (no evaluations run)"
else
  for r in "${RESULT_PATHS[@]}"; do
    echo "   ${r}"
  done
fi
echo "=============================================================="

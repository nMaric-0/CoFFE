#!/bin/bash
#
# Trento Cosine Similarity Few-Shot Evaluation Script
#
# Evaluates MFT-CPEA models on Trento dataset using prototypical networks with
# cosine similarity (parameter-free, ideal for zero-shot evaluation).
#
# Usage:
#   ./scripts/run_trento_cosine_eval.sh <checkpoint_path> [n_way] [k_shot] [output_dir] [distance_metric] [temperature]
#
# Examples:
#   # Evaluate 5-way 5-shot on Trento with default settings
#   ./scripts/run_trento_cosine_eval.sh checkpoints/pretrained/trento_enhanced/encoder_final.pth
#
#   # Evaluate 6-way 5-shot (all classes)
#   ./scripts/run_trento_cosine_eval.sh checkpoints/pretrained/trento_enhanced/encoder_final.pth 6 5
#
#   # Evaluate 5-way 1-shot with Euclidean distance
#   ./scripts/run_trento_cosine_eval.sh checkpoints/pretrained/trento_enhanced/encoder_final.pth 5 1 results/trento_1shot euclidean
#
#   # Evaluate random initialization baseline
#   ./scripts/run_trento_cosine_eval.sh random

set -e  # Exit on error

# ======================
# Configuration
# ======================

# Arguments
CHECKPOINT=${1:-"checkpoints/pretrained/trento_enhanced/checkpoint_epoch_325.pth"}
N_WAY=${2:-6}
K_SHOT=${3:-5}
OUTPUT_DIR=${4:-"results/cosine_eval/trento_${N_WAY}way_${K_SHOT}shot"}
DISTANCE_METRIC=${5:-"euclidean"}  # "cosine" or "euclidean"
TEMPERATURE=${6:-10.0}          # Temperature for cosine similarity
PROTOTYPE_MODE=${7:-"mean_features"}  # "mean_features" or "mean_distances"

# Dataset (fixed for Trento)
DATASET="trento"

# Model hyperparameters (match pretraining config)
EMBED_DIM=128
NUM_HEADS=8
NUM_LAYERS=4
PATCH_SIZE=11
LAMBDA_FACTOR=1.0
DROPOUT=0.1

# Evaluation settings
K_QUERY=19
NUM_EPISODES=600
SPLIT="all"
DATA_ROOT="./data/raw"
SEED=42

# ======================
# Setup
# ======================

echo "=========================================="
echo "Trento Cosine Similarity Few-Shot Evaluation"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Setting: ${N_WAY}-way ${K_SHOT}-shot"
echo "Distance Metric: ${DISTANCE_METRIC}"
echo "Prototype Mode: ${PROTOTYPE_MODE}"
if [ "${DISTANCE_METRIC}" = "cosine" ]; then
    echo "Temperature: ${TEMPERATURE}"
fi
echo "Episodes: ${NUM_EPISODES}"
echo "Split: ${SPLIT}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/eval_${TIMESTAMP}.log"

# ======================
# Run Evaluation
# ======================

echo "Starting Trento cosine similarity evaluation..."
echo "Log file: ${LOG_FILE}"

python scripts/evaluate_cosine.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --n-way ${N_WAY} \
    --k-shot ${K_SHOT} \
    --k-query ${K_QUERY} \
    --num-episodes ${NUM_EPISODES} \
    --split "${SPLIT}" \
    --data-root "${DATA_ROOT}" \
    --patch-size ${PATCH_SIZE} \
    --embed-dim ${EMBED_DIM} \
    --num-heads ${NUM_HEADS} \
    --num-layers ${NUM_LAYERS} \
    --lambda-factor ${LAMBDA_FACTOR} \
    --dropout ${DROPOUT} \
    --distance-metric "${DISTANCE_METRIC}" \
    --temperature ${TEMPERATURE} \
    --prototype-mode "${PROTOTYPE_MODE}" \
    --seed ${SEED} \
    --output "${OUTPUT_DIR}/results.json" \
    2>&1 | tee "${LOG_FILE}"

# ======================
# Summary
# ======================

echo ""
echo "=========================================="
echo "Trento Evaluation Complete!"
echo "=========================================="
echo "Method: MFT-CPEA-Cosine (${DISTANCE_METRIC}, ${PROTOTYPE_MODE})"
if [ "${DISTANCE_METRIC}" = "cosine" ]; then
    echo "Temperature: ${TEMPERATURE}"
fi
echo "Results saved to: ${OUTPUT_DIR}/results.json"
echo "Log saved to: ${LOG_FILE}"
echo "=========================================="

# Extract key metrics from results
if [ -f "${OUTPUT_DIR}/results.json" ]; then
    echo ""
    echo "Quick Summary:"
    python -c "
import json
with open('${OUTPUT_DIR}/results.json') as f:
    data = json.load(f)
    print(f\"  OA:    {data['OA']['mean']:.2f}% ± {data['OA']['ci_95']:.2f}%\")
    print(f\"  AA:    {data['AA']['mean']:.2f}% ± {data['AA']['ci_95']:.2f}%\")
    print(f\"  Kappa: {data['Kappa']['mean']:.2f} ± {data['Kappa']['ci_95']:.2f}\")
" 2>/dev/null || echo "  (Could not parse results)"
fi

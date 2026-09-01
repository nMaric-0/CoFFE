#!/bin/bash
#
# Few-Shot Evaluation Driver
#
# Evaluates a frozen CoFFE (or MFT control) encoder by nearest class mean.
# The default distance is euclidean, as in every paper run, but the episode
# defaults below are NOT the paper's: k_query=30 / 600 episodes against the
# paper's 100 / 1000. Pass them explicitly, or use lib/eval_runner.py.
#
# Usage:
#   ./scripts/run_eval.sh <checkpoint_path> <dataset> <n_way> <k_shot> [output_dir] [distance_metric] [temperature]
#
# Examples:
#   # Houston, all 15 classes, 5-shot, Euclidean NCM. NOTE the script's own
#   # defaults are k_query=30 / 600 episodes, NOT the paper's 100 / 1000.
#   ./scripts/run_eval.sh checkpoints/pretrained/houston_enhanced/encoder_final.pth houston 15 5
#
#   # 6-way 1-shot on Trento
#   ./scripts/run_eval.sh checkpoints/pretrained/trento_enhanced/encoder_final.pth trento 6 1 results/trento_1shot euclidean
#
#   # cosine distance with a custom temperature (not the paper protocol)
#   ./scripts/run_eval.sh checkpoints/pretrained/houston_enhanced/encoder_final.pth houston 15 5 results/houston cosine 5.0
#
#   # Evaluate random initialization baseline
#   ./scripts/run_eval.sh random houston 15 5
 
set -e  # Exit on error
 
# ======================
# Configuration
# ======================
 
# Arguments
CHECKPOINT=${1:-"checkpoints/pretrained/houston_enhanced_CenterWeightedLoss_2layers/checkpoint_epoch_10000.pth"}
DATASET=${2:-"houston"}
N_WAY=${3:-15}
K_SHOT=${4:-5}
OUTPUT_DIR=${5:-"results/eval/${DATASET}_${N_WAY}way_${K_SHOT}shot"}
DISTANCE_METRIC=${6:-"euclidean"}  # "cosine" or "euclidean"
TEMPERATURE=${7:-10.00}          # Temperature for cosine similarity
PROTOTYPE_MODE=${8:-"mean_features"}  # "mean_features" or "mean_distances"
 
# Model hyperparameters (match pretraining config)
EMBED_DIM=128
NUM_HEADS=2
NUM_LAYERS=2
PATCH_SIZE=11
LAMBDA_FACTOR=0.5
DROPOUT=0.1
 
# Evaluation settings
K_QUERY=30
NUM_EPISODES=600
SPLIT="all"
DATA_ROOT="./data/raw"
SEED=42
 
# ======================
# Setup
# ======================
 
echo "=========================================="
echo "Few-Shot Evaluation (nearest class mean)"
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
 
echo "Starting few-shot evaluation..."
echo "Log file: ${LOG_FILE}"
 
python scripts/evaluate.py \
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
    --no-projection \
    2>&1 | tee "${LOG_FILE}"
 
# ======================
# Summary
# ======================
 
echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Method: nearest-class-mean (${DISTANCE_METRIC}, ${PROTOTYPE_MODE})"
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
 
#!/usr/bin/env python
"""
Few-shot evaluation for CoFFE and the MFT architectural control.

The encoder is frozen and never finetuned: each episode's class prototype is
the mean of its K support features, and queries are assigned by
nearest-class-mean (PAPER_CANON §4). The paper protocol is Euclidean NCM
(``--distance-metric euclidean``); cosine is kept as an option value only.

There are no learnable similarity parameters — any DenseSimilarity weights in
an old checkpoint are ignored on load.

Usage:
    # Paper protocol: N-way (all classes), 5-shot, Euclidean NCM
    python scripts/evaluate.py \
        --checkpoint experiments/<run>/checkpoints/checkpoint_epoch_950.pth \
        --dataset houston \
        --k-shot 5 --distance-metric euclidean --no-projection
"""

import argparse
import sys
from pathlib import Path

# Keep the repo root importable so `python scripts/evaluate.py` works in a tree
# that has not been `pip install -e .`-ed, exactly like the other entry points
# under scripts/. Harmless when the package is installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from coffe.eval.episodic import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-shot evaluation for CoFFE / MFT (nearest-class-mean on frozen features)"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Checkpoint path or 'random' for no pretraining")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["houston", "trento", "muufl"])

    # Few-shot settings
    parser.add_argument("--n-way", type=int, default=None,
                       help="Number of classes per episode. Default (and the paper "
                            "protocol): all classes in the scene, i.e. N-way with "
                            "N = 15 Houston / 6 Trento / 11 MUUFL.")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--k-query", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=2000)

    # Dataset
    parser.add_argument("--data-root", type=str, default="./data/raw")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test", "all"])
    parser.add_argument("--patch-size", type=int, default=11)

    # Model config
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lambda-factor", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-projection", action="store_true", default=True,
                       help="Use projection head (default: True, matching pretraining)")
    parser.add_argument("--no-projection", dest="use_projection", action="store_false",
                       help="Disable projection head")
    parser.add_argument("--use-aux", action="store_true", default=True,
                       help="Concatenate aux (LiDAR) bands with HSI (default: True). "
                            "Must match how the checkpoint was pretrained.")
    parser.add_argument("--no-aux", dest="use_aux", action="store_false",
                       help="HSI-only: ignore aux/LiDAR bands (for HSI-only checkpoints)")

    # Nearest-class-mean settings. Default is euclidean, the paper protocol
    # (phase-4 gate decision 2026-09-01, PAPER_CANON §1: cosine may exist as an
    # option value but never as a default). Cosine still works, with
    # --temperature.
    parser.add_argument("--distance-metric", type=str, default="euclidean",
                       choices=["euclidean", "cosine"],
                       help="Distance to the class means (default: euclidean, "
                            "the paper protocol).")
    parser.add_argument("--temperature", type=float, default=10.0,
                       help="Temperature scaling for cosine similarity")
    parser.add_argument("--prototype-mode", type=str, default="mean_features",
                       choices=["mean_features", "mean_distances"],
                       help="Class mean computation: 'mean_features' (avg the K support "
                            "features, then measure distance; the paper protocol) or "
                            "'mean_distances' (distance to each support, then avg)")
    parser.add_argument("--pool-sigma", type=float, default=None,
                       help="Gaussian sigma for center-weighted spatial pooling. "
                            "None = uniform mean pooling (default). Recommended: 2.0")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (used if --seeds not provided)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="Multiple seeds for evaluation. Runs once per seed and reports "
                            "aggregate stats. Example: --seeds 42 123 456")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU evaluation (legacy flag; --device cpu also works).")
    parser.add_argument("--device", type=str, default="auto",
                       help="'cuda' | 'cuda:N' | 'auto' | 'cpu'. Ignored if --cpu is set.")
    parser.add_argument("--output", type=str, default=None)

    # Visualization
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory for output plots. Auto-generated if not set.")
    parser.add_argument("--no-plots", action="store_true", default=False,
                       help="Disable visualization plot generation")
    parser.add_argument("--num-example-episodes", type=int, default=3,
                       help="Number of episodes to generate per-episode t-SNE "
                            "feature space plots for. 0 to skip.")
    parser.add_argument("--max-tsne-samples", type=int, default=0,
                       help="Max samples per class for aggregated t-SNE plot. "
                            "0 = use all samples (no subsampling).")

    args = parser.parse_args()
    main(args)

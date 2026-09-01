#!/usr/bin/env python
"""Few-shot evaluation for HyperSIGMA on the same episodes as CoFFE.

Mirrors ``scripts/evaluate.py`` so result files are
directly comparable. Differences:

* Encoder is ``HyperSIGMAFewShot`` (a frozen wrapper around
  ``HyperSIGMADual``), not ``CoFFE``.
* The eval loop accumulates *both* a cosine and a Euclidean confusion
  matrix from the same per-batch features so the JSON has parallel
  ``cosine`` and ``euclidean`` result blocks.
* CLI flags add ``--mode {fused, spat_pool, spec_pool}``,
  ``--spat-patch-k {1, 3}``, ``--pca-spat-path``, and
  ``--adapted-checkpoint`` (use the literal string ``none`` to skip
  loading adapted weights — that's the "unadapted" ablation).

Usage:
    python scripts/evaluate_hypersigma.py \
        --pca-spat-path checkpoints/hypersigma/pca_houston_3band.pkl \
        --spat-ckpt checkpoints/hypersigma/spat-vit-base.pth \
        --spec-ckpt checkpoints/hypersigma/spec-vit-base.pth \
        --adapted-checkpoint checkpoints/hypersigma_adapted/houston_k3/checkpoint.pth \
        --mode fused
"""

import argparse
import sys
from pathlib import Path

# Keep the repo root importable so `python scripts/evaluate_hypersigma.py` works in a tree
# that has not been `pip install -e .`-ed, exactly like the other entry points
# under scripts/. Harmless when the package is installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from coffe.eval.hypersigma import DATASETS, main
from coffe.models.hypersigma import HyperSIGMAFewShot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="houston", choices=list(DATASETS))
    parser.add_argument("--n-way", type=int, default=None,
                        help="Number of classes per episode. Default: all "
                             "available classes for the dataset (C-way).")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--k-query", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=600)
    parser.add_argument("--data-root", type=str, default="./data/raw")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--mode", type=str, default="fused",
                        choices=list(HyperSIGMAFewShot.SUPPORTED_MODES))
    parser.add_argument("--spat-patch-k", type=int, default=3, choices=[1, 3])
    parser.add_argument("--spat-resample-to", type=int, default=None,
                        help="Fixed spatial channel width (e.g. 100). Datasets with "
                             "fewer bands than this and no matching PCA pickle spectrally "
                             "resample their bands up to it (Trento 63 / MUUFL 64 -> 100); "
                             "datasets with a PCA pickle (Houston 144->100) use PCA.")
    parser.add_argument("--native-geometry", action="store_true", default=False,
                        help="Native-geometry ablation: keep the encoder at its "
                             "pretrained input size (SpatViT 64x64/patch-8/100ch, "
                             "SpecViT 64x64) so the pretrained projections load, and "
                             "resize the 11x11 input up to it. Single-branch + unadapted only.")
    parser.add_argument("--input-fit", type=str, default="upscale", choices=["upscale", "pad"],
                        help="How to fit 11x11 to the native input size (native-geometry only).")
    parser.add_argument("--pad-anchor", type=str, default="center", choices=["center", "top-left"],
                        help="Placement for --input-fit pad (native-geometry only).")
    parser.add_argument("--interp-mode", type=str, default="bicubic", choices=["bicubic", "bilinear"],
                        help="Interpolation for --input-fit upscale (native-geometry only).")
    parser.add_argument("--native-pca-spat-path", type=str, default=None,
                        help="100-band spatial PCA pickle for native geometry. If omitted, "
                             "derived from --dataset (pca_<ds>_100band.pkl); if absent, the "
                             "spatial branch spectrally resamples raw bands -> 100.")
    parser.add_argument("--pca-spat-path", type=str, default=None,
                        help="Spatial PCA pickle. If omitted, derived from "
                             "--dataset (checkpoints/hypersigma/pca_<ds>_3band.pkl).")
    parser.add_argument("--pca-stats-path", type=str, default=None,
                        help="Pickled per-channel mean/std for PCA-output standardization. "
                             "If omitted, the PCA output is fed raw to SpatViT.")
    parser.add_argument("--spat-ckpt", type=str,
                        default="checkpoints/hypersigma/spat-vit-base.pth")
    parser.add_argument("--spec-ckpt", type=str,
                        default="checkpoints/hypersigma/spec-vit-base.pth")
    parser.add_argument("--adapted-checkpoint", type=str, default=None,
                        help="Path to adapted checkpoint; 'auto' derives "
                             "checkpoints/hypersigma_adapted/<ds>_k<k>/checkpoint.pth; "
                             "omit or 'none' for the unadapted ablation")
    parser.add_argument("--distance-metric", type=str, default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Primary distance to the class means (default: "
                             "euclidean, the paper protocol). Both blocks are "
                             "always reported for HyperSIGMA.")
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--prototype-mode", type=str, default="mean_features",
                        choices=["mean_features"])  # mean_distances not used for the dual feature
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device for eval, e.g. 'cuda', 'cuda:1', 'cpu'. "
                             "Ignored if --cpu is set or CUDA is unavailable.")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Convenience for --device cuda:<int> (overrides --device).")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--num-example-episodes", type=int, default=3)
    parser.add_argument("--max-tsne-samples", type=int, default=0)
    args = parser.parse_args()
    main(args)

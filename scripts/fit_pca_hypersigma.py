#!/usr/bin/env python
"""Fit and save the spatial PCA used by the HyperSIGMA SpatViT branch.

For each supported dataset we fit a sklearn ``PCA(n_components=3)`` to
every pixel of the already-normalized HSI patches (``split='all'``) and
pickle it under ``checkpoints/hypersigma/pca_<dataset>_3band.pkl``.

No spectral PCA is fit — the SpecViT branch ingests raw bands and
handles arbitrary band counts via its built-in
``AdaptiveAvgPool1d(NUM_TOKENS)``.

Usage:
    python scripts/fit_pca_hypersigma.py --dataset houston \
        --data-root ./data/raw \
        --out-dir checkpoints/hypersigma
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import DATASET_REGISTRY, get_spec  # noqa: E402
from models.hypersigma.preprocessing import (  # noqa: E402
    fit_dataset_pca,
    fit_pca_output_stats,
)

DATASETS = {name: spec.patched_cls for name, spec in DATASET_REGISTRY.items()}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-root", type=str, default="./data/raw")
    parser.add_argument("--out-dir", type=str, default="checkpoints/hypersigma")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument(
        "--n-components", type=int, default=None,
        help="Number of PCA components. Default: the dataset's spat_components "
             "(3, headline adapt pipeline). Use 100 to fit the native-geometry "
             "spatial PCA (pca_<ds>_100band.pkl). Must be <= the band count.",
    )
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    n_components = args.n_components if args.n_components is not None else spec.spat_components
    if n_components > spec.hsi_channels:
        raise ValueError(
            f"--n-components={n_components} exceeds {args.dataset}'s {spec.hsi_channels} "
            "bands; PCA cannot produce more components than bands. For native geometry, "
            "below-100-band datasets are handled by on-the-fly spectral resampling at "
            "eval time (no PCA fit needed)."
        )

    dataset = spec.patched_cls(
        data_root=args.data_root,
        patch_size=args.patch_size,
        split="all",
        normalize=True,
    )
    logger.info("Loaded %s (split=all): %d samples", args.dataset, len(dataset))

    # Paths follow the registry convention (pca_<ds>_<n>band[.|_stats.]pkl) so
    # adapt/eval pick them up automatically. We also fit the per-channel output
    # stats used to standardize the SpatViT input (mean=0/std=1).
    pca_path = f"{args.out_dir}/pca_{args.dataset}_{n_components}band.pkl"
    stats_path = f"{args.out_dir}/pca_{args.dataset}_{n_components}band_stats.pkl"
    pca = fit_dataset_pca(
        dataset=dataset,
        n_components=n_components,
        save_path=pca_path,
    )
    fit_pca_output_stats(dataset=dataset, pca=pca, save_path=stats_path)


if __name__ == "__main__":
    main()

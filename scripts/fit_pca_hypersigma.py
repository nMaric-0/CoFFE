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

from data.datasets import (  # noqa: E402
    HoustonPatchedDataset,
    TrentoPatchedDataset,
    MUUFLPatchedDataset,
)
from models.hypersigma.preprocessing import (  # noqa: E402
    DATASET_PCA_CONFIG,
    fit_dataset_pca,
)

DATASETS = {
    "houston": HoustonPatchedDataset,
    "trento": TrentoPatchedDataset,
    "muufl": MUUFLPatchedDataset,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-root", type=str, default="./data/raw")
    parser.add_argument("--out-dir", type=str, default="checkpoints/hypersigma")
    parser.add_argument("--patch-size", type=int, default=11)
    args = parser.parse_args()

    cfg = DATASET_PCA_CONFIG[args.dataset]
    Dataset = DATASETS[args.dataset]
    dataset = Dataset(
        data_root=args.data_root,
        patch_size=args.patch_size,
        split="all",
        normalize=True,
    )
    logger.info("Loaded %s (split=all): %d samples", args.dataset, len(dataset))

    save_path = Path(args.out_dir) / f"pca_{args.dataset}_{cfg['spat_components']}band.pkl"
    fit_dataset_pca(
        dataset=dataset,
        n_components=cfg["spat_components"],
        save_path=str(save_path),
    )


if __name__ == "__main__":
    main()

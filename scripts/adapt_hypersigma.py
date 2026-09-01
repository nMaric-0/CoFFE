#!/usr/bin/env python
"""Level-2 MAE adaptation of HyperSIGMA on a target dataset.

Trains the small randomly-initialized components of the dual encoder
(SpatViT ``patch_embed`` + ``pos_embed``, SpecViT ``spat_map`` +
``pos_embed`` + ``l1``, plus the SEM and the ``UnifiedBandMasking``
fill values) using masked HSI reconstruction. SpatViT/SpecViT
transformer bodies remain frozen.

Optimizer recipe follows the CoFFE pretraining configs: lr=1.5e-4,
weight_decay=0.05, warmup=100, batch_size=64, AMP off. The decoders are
HyperSIGMA-specific (`spat_decoder_hidden`/`spec_decoder_hidden` 256,
`fused_decoder_hidden` 512) and there is no centre-weighting here. Masking here is HyperSIGMA's own 75% token masking
(``pretrain.mask_ratio``), not CoFFE's band/token pair, and the epoch count is
per-config (2000-3000).

The target dataset is selected entirely by the ``data.dataset`` field
of the config (``houston`` | ``trento`` | ``muufl``). Band count and the
PCA / checkpoint paths default to the conventions in
``coffe/data/datasets/registry.py`` when not given explicitly, so switching
datasets is a one-field change.

Usage:
    python scripts/adapt_hypersigma.py \
        --config configs/hypersigma/houston_patchnative_joint_sem.yaml
"""

import argparse
import sys
from pathlib import Path

# Keep the repo root importable so `python scripts/adapt_hypersigma.py` works in a tree
# that has not been `pip install -e .`-ed, exactly like the other entry points
# under scripts/. Harmless when the package is installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from coffe.pretrain.hypersigma_adapt import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)

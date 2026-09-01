#!/usr/bin/env python
"""
Per-scene masked pretraining for CoFFE and the MFT architectural control.

HSI + LiDAR are concatenated as a single per-pixel band vector. Two composable
SimMIM-style masks are supported (configured via the pretrain YAML):
  - band_mask_ratio: per-(pixel, band) Bernoulli mask on the raw input.
  - spatial_mask_ratio: whole-pixel-token (spatial) masking, in place.
The MAE objective (token drop + transformer decoder) is selected with
``pretrain.objective: "mae"``.

Usage:
    python scripts/pretrain.py --config configs/coffe/houston_simmim.yaml

    # Resume training
    python scripts/pretrain.py --config configs/coffe/houston_simmim.yaml --resume checkpoint_epoch_400.pth
"""

import argparse
import sys
from pathlib import Path

# Keep the repo root importable so `python scripts/pretrain.py` works in a tree
# that has not been `pip install -e .`-ed, exactly like the other entry points
# under scripts/. Harmless when the package is installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from coffe.pretrain.loop import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-scene masked pretraining (SimMIM / MAE) for CoFFE or the MFT control"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path"
    )

    args = parser.parse_args()
    main(args)

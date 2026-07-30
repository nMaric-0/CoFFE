"""Run ONE ablation training run.

Builds overrides = baseline (cloned from the canonical Houston-spatial run) deep-
merged with the variant's single-field change, plus forced epochs/save_interval/
seed/device, then launches via lib.pretrain_runner.run_pretrain.

Idempotent: exits 0 without retraining if checkpoint_epoch_<EPOCHS>.pth exists.

Example:
    python scripts/ablation_pretrain_worker.py --slug center_off --seed 42 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import ablation_config as cfg  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="One ablation training run.")
    p.add_argument("--slug", required=True, choices=cfg.SLUGS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True, help="e.g. cuda:0")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    name = cfg.experiment_name(args.slug, args.seed)
    ckpt = cfg.EXPERIMENTS_ROOT / name / "checkpoints" / f"checkpoint_epoch_{args.epochs}.pth"
    if ckpt.exists() and not args.overwrite:
        print(f"[skip] {name}: {ckpt.name} already exists.")
        return 0

    overrides = cfg.build_overrides(args.slug, args.seed, args.device)
    if args.epochs != cfg.EPOCHS:
        overrides["pretrain"]["epochs"] = args.epochs

    description = (
        f"Ablation [{args.slug}]: Houston enhanced spatial HSI+LiDAR, seed {args.seed}, "
        f"{args.epochs} epochs on {args.device}. One-factor change vs baseline."
    )
    print(f"[train] {name} on {args.device} | overrides={overrides}")

    from lib.pretrain_runner import run_pretrain

    run_pretrain(
        name=name,
        description=description,
        config=cfg.BASE_CONFIG,
        overrides=overrides,
        overwrite=args.overwrite,
    )
    print(f"[done] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

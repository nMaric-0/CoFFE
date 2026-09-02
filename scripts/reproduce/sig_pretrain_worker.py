"""Run ONE training run for the significance experiment.

Resolves the (group, dataset, variant) cell, clones the canonical seed-42
``pretrain`` overrides when the group needs per-variant mask ratios (enhanced /
hsi_only) else uses the self-contained base config, then forces ``epochs`` /
``save_interval`` / ``seed`` / ``device`` and launches via
lib.pretrain_runner.run_pretrain (same path the notebook uses).

Idempotent: exits 0 without retraining if the target experiment already has a
checkpoint_epoch_<EPOCHS>.pth.

Example:
    python scripts/reproduce/sig_pretrain_worker.py \
        --group mft_mae --dataset houston --variant mae --seed 123 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce import sig_significance_config as cfg


def main() -> int:
    p = argparse.ArgumentParser(description="One significance-experiment training run.")
    p.add_argument("--group", required=True, choices=list(cfg.GROUPS))
    p.add_argument("--dataset", required=True, choices=cfg.DATASETS)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True, help="e.g. cuda:0")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    group = cfg.GROUPS[args.group]
    if args.variant not in group.variants:
        p.error(
            f"variant '{args.variant}' not valid for group '{args.group}' (valid: {group.variants})"
        )

    name = group.experiment_name(args.dataset, args.variant, args.seed)
    ckpt = cfg.EXPERIMENTS_ROOT / name / "checkpoints" / f"checkpoint_epoch_{args.epochs}.pth"
    if ckpt.exists() and not args.overwrite:
        print(f"[skip] {name}: {ckpt.name} already exists.")
        return 0

    pretrain_overrides = cfg.canonical_pretrain_overrides(args.group, args.dataset, args.variant)
    pretrain_overrides["epochs"] = args.epochs
    pretrain_overrides["save_interval"] = cfg.SAVE_INTERVAL

    overrides = {
        "pretrain": pretrain_overrides,
        "hardware": {"seed": args.seed, "device": args.device},
    }

    base = group.base_config(args.dataset, args.variant)
    description = (
        f"Significance experiment [{args.group}]: {args.dataset} {args.variant}, "
        f"seed {args.seed}, {args.epochs} epochs on {args.device}. Base config {base}."
    )
    if group.clone_mask:
        description += f" Cloned recipe from {group.canonical_dir(args.dataset, args.variant)}."

    print(f"[train] {name} on {args.device} | base={base} | overrides={overrides}")

    from coffe.runners.pretrain_runner import run_pretrain

    run_pretrain(
        name=name,
        description=description,
        config=base,
        overrides=overrides,
        overwrite=args.overwrite,
    )
    print(f"[done] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

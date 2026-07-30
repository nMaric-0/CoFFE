"""Run ONE best-config training run.

Overrides = native masking recipe (cloned from the cell's canonical run) + the
winning architecture trio (dropout=0, embed_dim=256, num_layers=4) + forced
epochs/save_interval/seed/device. Launches via lib.pretrain_runner.run_pretrain.

Idempotent: skip if checkpoint_epoch_<EPOCHS>.pth exists.

Example:
    python scripts/bestcfg_pretrain_worker.py \
        --group enhanced --dataset houston --variant spatial --seed 42 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bestcfg_config as cfg  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="One best-config training run.")
    p.add_argument("--group", required=True, choices=cfg.GROUPS_USED)
    p.add_argument("--dataset", required=True, choices=cfg.DATASETS)
    p.add_argument("--variant", required=True, choices=cfg.VARIANTS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True)
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    name = cfg.experiment_name(args.group, args.dataset, args.variant, args.seed)
    exp_dir = cfg.EXPERIMENTS_ROOT / name
    ckpt_dir = exp_dir / "checkpoints"
    final_ckpt = ckpt_dir / f"checkpoint_epoch_{args.epochs}.pth"
    if final_ckpt.exists() and not args.overwrite:
        print(f"[skip] {name}: {final_ckpt.name} already exists.")
        return 0

    # Resume from the latest partial checkpoint if the run was interrupted, so a
    # restart (e.g. after freeing a GPU) continues instead of starting over.
    resume = None
    reuse_dir = exp_dir.exists() and not args.overwrite
    if reuse_dir and ckpt_dir.exists():
        parts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        if parts:
            resume = str(parts[-1])

    overrides = cfg.build_overrides(args.group, args.dataset, args.variant, args.seed, args.device)
    if args.epochs != cfg.EPOCHS:
        overrides["pretrain"]["epochs"] = args.epochs

    base = cfg.base_config(args.group, args.dataset, args.variant)
    description = (
        f"Best-config rerun [{args.group}/{args.dataset}/{args.variant}]: arch trio "
        f"(dropout0, embed256, layers4) on native masking, seed {args.seed}, "
        f"{args.epochs} epochs on {args.device}."
    )
    print(f"[train] {name} on {args.device} | base={base} | "
          f"resume={resume or 'fresh'} | overrides={overrides}")

    from lib.pretrain_runner import run_pretrain

    run_pretrain(
        name=name,
        description=description,
        config=base,
        overrides=overrides,
        resume=resume,
        overwrite=args.overwrite or reuse_dir,
    )
    print(f"[done] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

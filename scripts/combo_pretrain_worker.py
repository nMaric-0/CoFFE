"""Run ONE combination training run.

--factors is a comma-joined list of pool factor slugs (e.g.
"decoder512,dropout0,mask0p90"). Overrides = baseline + each factor's single-
field change + forced epochs/save_interval/seed/device.

Idempotent: skip if checkpoint_epoch_<EPOCHS>.pth exists.

Example:
    python scripts/combo_pretrain_worker.py \
        --factors decoder512,dropout0,mask0p90 --seed 42 --device cuda:2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import combo_config as cfg  # noqa: E402


def _parse_factors(s: str):
    factors = [f for f in s.split(",") if f]
    bad = [f for f in factors if f not in cfg.POOL]
    if bad:
        raise SystemExit(f"unknown factors {bad}; pool={cfg.POOL}")
    return factors


def main() -> int:
    p = argparse.ArgumentParser(description="One combination training run.")
    p.add_argument("--factors", required=True, help="comma-joined pool slugs")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True)
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    factors = _parse_factors(args.factors)
    name = cfg.experiment_name(factors, args.seed)
    ckpt = cfg.EXPERIMENTS_ROOT / name / "checkpoints" / f"checkpoint_epoch_{args.epochs}.pth"
    if ckpt.exists() and not args.overwrite:
        print(f"[skip] {name}: {ckpt.name} already exists.")
        return 0

    overrides = cfg.build_overrides(factors, args.seed, args.device)
    if args.epochs != cfg.EPOCHS:
        overrides["pretrain"]["epochs"] = args.epochs

    description = (
        f"Combination [{'+'.join(cfg._canonical(factors))}]: Houston enhanced spatial "
        f"HSI+LiDAR, seed {args.seed}, {args.epochs} epochs on {args.device}."
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

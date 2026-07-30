"""Run ONE combination eval (single combo/seed at a single epoch).

Evaluates checkpoint_epoch_<epoch>.pth into
experiments/<name>/evaluations/abl_eval_epoch<epoch>/results.json (same eval
protocol as the OFAT ablation; arch auto-read from the frozen config).

Idempotent: skip if that eval's results.json exists.

Example:
    python scripts/combo_eval_worker.py \
        --factors decoder512,dropout0 --seed 42 --epoch 1000 --device cuda:2
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
    p = argparse.ArgumentParser(description="One combination eval run.")
    p.add_argument("--factors", required=True, help="comma-joined pool slugs")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--epoch", required=True, type=int, choices=cfg.EVAL_EPOCHS)
    p.add_argument("--device", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    factors = _parse_factors(args.factors)
    name = cfg.experiment_name(factors, args.seed)
    eval_name = cfg.eval_name_for_epoch(args.epoch)
    results_path = cfg.EXPERIMENTS_ROOT / name / "evaluations" / eval_name / "results.json"
    if results_path.exists() and not args.overwrite:
        print(f"[skip] {name}: {eval_name}/results.json exists.")
        return 0

    eval_params = cfg.eval_params()
    eval_params["device"] = args.device

    print(f"[eval] {name} epoch {args.epoch} ({eval_name}) on {args.device}")

    from lib.eval_runner import run_evaluation

    run_evaluation(
        experiment_name=name,
        eval_name=eval_name,
        epoch=args.epoch,
        eval_params=eval_params,
        overwrite=args.overwrite,
    )
    print(f"[done] {name} @epoch{args.epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

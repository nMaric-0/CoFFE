"""Run ONE ablation eval (a single variant/seed at a single epoch).

Evaluates checkpoint_epoch_<epoch>.pth into
experiments/<name>/evaluations/abl_eval_epoch<epoch>/results.json with the
shared eval protocol (no plots). Architecture is auto-read from the run's frozen
pretrain_config.yaml.

Idempotent: exits 0 if that eval's results.json already exists.

Example:
    python scripts/ablation_eval_worker.py --slug baseline --seed 42 --epoch 800 --device cuda:0
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
    p = argparse.ArgumentParser(description="One ablation eval run.")
    p.add_argument("--slug", required=True, choices=cfg.SLUGS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--epoch", required=True, type=int, choices=cfg.EVAL_EPOCHS)
    p.add_argument("--device", required=True, help="e.g. cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    name = cfg.experiment_name(args.slug, args.seed)
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

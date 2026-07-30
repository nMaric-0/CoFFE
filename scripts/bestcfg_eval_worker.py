"""Run ONE best-config eval (single cell/seed at epoch 1000).

Writes experiments/<name>/evaluations/best_eval_epoch1000/results.json with the
shared eval protocol (arch auto-read from the frozen pretrain_config.yaml, so the
embed256/layers4 architecture is rebuilt correctly).

Idempotent: skip if results.json exists.

Example:
    python scripts/bestcfg_eval_worker.py \
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
    p = argparse.ArgumentParser(description="One best-config eval run.")
    p.add_argument("--group", required=True, choices=cfg.GROUPS_USED)
    p.add_argument("--dataset", required=True, choices=cfg.DATASETS)
    p.add_argument("--variant", required=True, choices=cfg.VARIANTS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True)
    p.add_argument("--epoch", type=int, default=cfg.EVAL_EPOCH)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    name = cfg.experiment_name(args.group, args.dataset, args.variant, args.seed)
    results_path = cfg.EXPERIMENTS_ROOT / name / "evaluations" / cfg.EVAL_NAME / "results.json"
    if results_path.exists() and not args.overwrite:
        print(f"[skip] {name}: {cfg.EVAL_NAME}/results.json exists.")
        return 0

    eval_params = cfg.eval_params(args.dataset)
    eval_params["device"] = args.device

    print(f"[eval] {name} epoch {args.epoch} on {args.device}")

    from lib.eval_runner import run_evaluation

    run_evaluation(
        experiment_name=name,
        eval_name=cfg.EVAL_NAME,
        epoch=args.epoch,
        eval_params=eval_params,
        overwrite=args.overwrite,
    )
    print(f"[done] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run ONE epoch-700 evaluation for the significance experiment.

Resolves the (group, dataset, variant, seed) cell to its experiment dir and
evaluates checkpoint_epoch_700.pth with the shared eval protocol.
lib.eval_runner pulls the architecture (model name, attention_type, mlp_dim,
use_aux, use_projection) from the run's frozen pretrain_config.yaml, so the same
call handles every family (coffe / mft_original, LiDAR / HSI-only).

Writes experiments/<name>/evaluations/sig_eval_epoch700/results.json.
Idempotent: exits 0 without re-running if that results.json already exists.

Example:
    python scripts/reproduce/sig_eval_worker.py \
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
    p = argparse.ArgumentParser(description="One significance-experiment eval run.")
    p.add_argument("--group", required=True, choices=list(cfg.GROUPS))
    p.add_argument("--dataset", required=True, choices=cfg.DATASETS)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True, help="e.g. cuda:0")
    p.add_argument("--epoch", type=int, default=cfg.EVAL_EPOCH)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    group = cfg.GROUPS[args.group]
    if args.variant not in group.variants:
        p.error(
            f"variant '{args.variant}' not valid for group '{args.group}' (valid: {group.variants})"
        )

    name = group.experiment_name(args.dataset, args.variant, args.seed)
    results_path = cfg.EXPERIMENTS_ROOT / name / "evaluations" / cfg.EVAL_NAME / "results.json"
    if results_path.exists() and not args.overwrite:
        print(f"[skip] {name}: {cfg.EVAL_NAME}/results.json exists.")
        return 0

    eval_params = cfg.eval_params(args.dataset)
    eval_params["device"] = args.device

    print(f"[eval] {name} epoch {args.epoch} on {args.device}")

    from coffe.runners.eval_runner import run_evaluation

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

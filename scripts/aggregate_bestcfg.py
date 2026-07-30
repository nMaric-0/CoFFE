"""Aggregate the best-config rerun -> mean +/- std +/- 95% CI per cell.

For each cell (group x dataset x variant) reads the 5 seeds'
best_eval_epoch1000/results.json, computes across-seed OA/AA/Kappa stats (t,
df=4). Writes experiments/bestcfg_report.json and prints a markdown table per
modality group (enhanced == HSI+LiDAR, hsi_only == HSI-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bestcfg_config as cfg  # noqa: E402

METRICS = ["OA", "AA", "Kappa"]
GROUP_LABEL = {"enhanced": "HSI+LiDAR", "hsi_only": "HSI-only"}


def _read_means(name: str) -> Optional[Dict[str, float]]:
    path = cfg.EXPERIMENTS_ROOT / name / "evaluations" / cfg.EVAL_NAME / "results.json"
    if not path.exists():
        return None
    with path.open() as f:
        res = json.load(f)
    return {m: float(res[m]["mean"]) for m in METRICS
            if isinstance(res.get(m), dict) and "mean" in res[m]}


def _stats(vals: List[float]) -> Dict[str, float]:
    n = len(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
    if n < 2:
        ci = float("nan")
    else:
        from scipy.stats import t as t_dist
        ci = float(t_dist.ppf(0.975, df=n - 1) * std / np.sqrt(n))
    return {"mean": mean, "std": std, "ci_95": ci, "n": n}


def main() -> int:
    report: Dict[str, object] = {
        "experiment": "bestcfg_rerun",
        "best_config": cfg.BEST_MODEL_OVERRIDES,
        "epoch": cfg.EVAL_EPOCH,
        "seeds": cfg.SEEDS,
        "eval_protocol": cfg.eval_params("<dataset>"),
        "cells": {}, "incomplete": [],
    }
    cells: Dict[str, dict] = report["cells"]      # type: ignore
    incomplete: List[dict] = report["incomplete"]  # type: ignore

    for group, dataset, variant in cfg.cells():
        key = f"{group}/{dataset}/{variant}"
        per_seed, missing = {}, []
        for seed in cfg.SEEDS:
            name = cfg.experiment_name(group, dataset, variant, seed)
            means = _read_means(name)
            (per_seed.__setitem__(str(seed), means) if means else missing.append(seed))
        cell = {"group": group, "dataset": dataset, "variant": variant,
                "per_seed": per_seed, "n": len(per_seed), "missing_seeds": missing, "stats": {}}
        for m in METRICS:
            vals = [per_seed[s][m] for s in per_seed if m in per_seed[s]]
            if vals:
                cell["stats"][m] = _stats(vals)
        cells[key] = cell
        if missing:
            incomplete.append({"cell": key, "missing_seeds": missing})

    out_path = cfg.EXPERIMENTS_ROOT / "bestcfg_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\n# Best-config rerun — arch trio {cfg.BEST_MODEL_OVERRIDES}, "
          f"epoch {cfg.EVAL_EPOCH}, {len(cfg.SEEDS)} seeds {cfg.SEEDS}\n")
    for group in cfg.GROUPS_USED:
        print(f"\n## {group}  ({GROUP_LABEL.get(group, group)})\n")
        print("| Dataset | Variant | n | OA (mean±std,95%CI) | AA (mean±std) | Kappa (mean±std) |")
        print("|" + "---|" * 6)
        for _g, dataset, variant in cfg.cells([group]):
            cell = cells[f"{group}/{dataset}/{variant}"]
            def fmt(m, ci=True):
                st = cell["stats"].get(m)
                if not st:
                    return "—"
                return f"{st['mean']:.2f} ± {st['std']:.2f}" + (f" (±{st['ci_95']:.2f})" if ci else "")
            print(f"| {dataset} | {variant} | {cell['n']} | {fmt('OA')} | "
                  f"{fmt('AA', False)} | {fmt('Kappa', False)} |")

    if incomplete:
        print("\n## Incomplete cells:")
        for it in incomplete:
            print(f"  - {it['cell']}: missing seeds {it['missing_seeds']}")
    else:
        print("\nAll 18 cells complete across all 5 seeds.")
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Aggregate the multi-seed significance experiment into mean +/- std +/- 95% CI.

For each cell (group x dataset x variant) collects the per-seed epoch-700
``evaluations/sig_eval_epoch700/results.json`` files, reads the top-level
OA / AA / Kappa means (euclidean, the eval's primary metric), and computes the
across-seed mean, sample std, and 95% confidence interval (t-distribution,
df=n-1).

Writes results/significance_report.json and prints a markdown table grouped
by family. Missing / incomplete cells are reported explicitly, never silently
dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce import sig_significance_config as cfg  # noqa: E402

METRICS = ["OA", "AA", "Kappa"]


def _results_path(name: str) -> Path:
    return cfg.EXPERIMENTS_ROOT / name / "evaluations" / cfg.EVAL_NAME / "results.json"


def _read_metric_means(name: str) -> Optional[Dict[str, float]]:
    """Return {metric: mean} for one run, or None if results.json is missing."""
    path = _results_path(name)
    if not path.exists():
        return None
    with path.open() as f:
        res = json.load(f)
    out: Dict[str, float] = {}
    for m in METRICS:
        block = res.get(m)
        if isinstance(block, dict) and "mean" in block:
            out[m] = float(block["mean"])
    return out


def _ci95(values: List[float]) -> float:
    """95% CI half-width via the t-distribution (matches scripts/evaluate.py)."""
    n = len(values)
    if n < 2:
        return float("nan")
    from scipy.stats import t as t_dist
    std = float(np.std(values, ddof=1))
    return float(t_dist.ppf(0.975, df=n - 1) * std / np.sqrt(n))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups", nargs="+", choices=list(cfg.GROUPS), default=None,
                   help="Groups to report (default: all).")
    args = p.parse_args()
    groups = args.groups or cfg.GROUP_ORDER

    report: Dict[str, object] = {
        "experiment": "significance",
        "epochs": cfg.EPOCHS,
        "eval_name": cfg.EVAL_NAME,
        "seeds": cfg.SEEDS,
        "groups": groups,
        "eval_protocol": cfg.eval_params("<dataset>"),
        "cells": {},
        "incomplete": [],
    }
    cells_out: Dict[str, dict] = report["cells"]  # type: ignore
    incomplete: List[dict] = report["incomplete"]  # type: ignore

    for group, dataset, variant in cfg.cells(groups):
        g = cfg.GROUPS[group]
        key = f"{group}/{dataset}/{variant}"
        per_seed: Dict[str, Dict[str, float]] = {}
        missing = []
        for seed in cfg.SEEDS:
            name = g.experiment_name(dataset, variant, seed)
            means = _read_metric_means(name)
            if means is None:
                missing.append(seed)
            else:
                per_seed[str(seed)] = means

        cell = {"group": group, "dataset": dataset, "variant": variant,
                "per_seed": per_seed, "n": len(per_seed),
                "missing_seeds": missing, "stats": {}}
        for m in METRICS:
            vals = [per_seed[s][m] for s in per_seed if m in per_seed[s]]
            if vals:
                cell["stats"][m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                    "ci_95": _ci95(vals),
                    "n": len(vals),
                }
        cells_out[key] = cell
        if missing:
            incomplete.append({"cell": key, "missing_seeds": missing})

    out_path = REPO_ROOT / "results" / "significance_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    # ---- markdown tables, one per family ----
    print(f"\n# Significance report (epoch {cfg.EPOCHS}, "
          f"{len(cfg.SEEDS)} seeds: {cfg.SEEDS})\n")
    for group in groups:
        print(f"\n## {group}\n")
        print("| Dataset | Variant | n | OA (mean±std, 95%CI) | "
              "AA (mean±std, 95%CI) | Kappa (mean±std, 95%CI) |")
        print("|" + "---|" * 6)
        for _g, dataset, variant in cfg.cells([group]):
            cell = cells_out[f"{group}/{dataset}/{variant}"]
            txt = []
            for m in METRICS:
                st = cell["stats"].get(m)
                txt.append("—" if st is None
                           else f"{st['mean']:.2f} ± {st['std']:.2f} (±{st['ci_95']:.2f})")
            print(f"| {dataset} | {variant} | {cell['n']} | " + " | ".join(txt) + " |")

    if incomplete:
        print("\n## Incomplete cells (missing seeds):")
        for item in incomplete:
            print(f"  - {item['cell']}: missing seeds {item['missing_seeds']}")
    else:
        print("\nAll selected cells complete across all seeds.")

    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

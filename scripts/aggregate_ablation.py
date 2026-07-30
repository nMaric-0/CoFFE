"""Aggregate the OFAT ablation into mean +/- std +/- 95% CI per variant per epoch.

For each variant and each eval epoch (800, 1000) reads the 3 seeds'
abl_eval_epoch<e>/results.json, takes the top-level OA/AA/Kappa means, and
computes across-seed mean, sample std, and 95% CI (t, df=2). Writes
experiments/ablation_report.json and prints one markdown table per epoch sorted
by OA desc, with the baseline row marked and delta-vs-baseline shown.
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

from scripts import ablation_config as cfg  # noqa: E402

METRICS = ["OA", "AA", "Kappa"]


def _read_means(name: str, epoch: int) -> Optional[Dict[str, float]]:
    path = (cfg.EXPERIMENTS_ROOT / name / "evaluations"
            / cfg.eval_name_for_epoch(epoch) / "results.json")
    if not path.exists():
        return None
    with path.open() as f:
        res = json.load(f)
    out = {}
    for m in METRICS:
        block = res.get(m)
        if isinstance(block, dict) and "mean" in block:
            out[m] = float(block["mean"])
    return out


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
        "experiment": "ablation",
        "dataset": cfg.DATASET,
        "epochs_evaluated": cfg.EVAL_EPOCHS,
        "seeds": cfg.SEEDS,
        "baseline": "baseline",
        "eval_protocol": cfg.eval_params(),
        "cells": {},
        "incomplete": [],
    }
    cells: Dict[str, dict] = report["cells"]  # type: ignore
    incomplete: List[dict] = report["incomplete"]  # type: ignore

    for slug in cfg.SLUGS:
        for epoch in cfg.EVAL_EPOCHS:
            key = f"{slug}@e{epoch}"
            per_seed = {}
            missing = []
            for seed in cfg.SEEDS:
                name = cfg.experiment_name(slug, seed)
                means = _read_means(name, epoch)
                if means is None:
                    missing.append(seed)
                else:
                    per_seed[str(seed)] = means
            cell = {"slug": slug, "epoch": epoch, "per_seed": per_seed,
                    "n": len(per_seed), "missing_seeds": missing, "stats": {}}
            for m in METRICS:
                vals = [per_seed[s][m] for s in per_seed if m in per_seed[s]]
                if vals:
                    cell["stats"][m] = _stats(vals)
            cells[key] = cell
            if missing:
                incomplete.append({"cell": key, "missing_seeds": missing})

    out_path = cfg.EXPERIMENTS_ROOT / "ablation_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    # ---- one table per epoch, sorted by OA desc, delta vs baseline ----
    print(f"\n# Ablation report — {cfg.DATASET} enhanced spatial HSI+LiDAR "
          f"({len(cfg.SEEDS)} seeds: {cfg.SEEDS})\n")
    for epoch in cfg.EVAL_EPOCHS:
        base = cells.get(f"baseline@e{epoch}", {}).get("stats", {}).get("OA")
        base_oa = base["mean"] if base else None
        rows = []
        for slug in cfg.SLUGS:
            cell = cells[f"{slug}@e{epoch}"]
            st = cell["stats"].get("OA")
            oa = st["mean"] if st else None
            rows.append((slug, cell, oa))
        rows.sort(key=lambda r: (r[2] is not None, r[2] if r[2] is not None else -1), reverse=True)

        print(f"\n## Epoch {epoch}\n")
        print("| Variant | n | OA (mean±std, 95%CI) | ΔOA vs base | AA (mean±std) | Kappa (mean±std) |")
        print("|" + "---|" * 6)
        for slug, cell, oa in rows:
            def fmt(m, with_ci=True):
                st = cell["stats"].get(m)
                if not st:
                    return "—"
                s = f"{st['mean']:.2f} ± {st['std']:.2f}"
                return s + (f" (±{st['ci_95']:.2f})" if with_ci else "")
            delta = "—" if (oa is None or base_oa is None) else f"{oa - base_oa:+.2f}"
            tag = " ⟵baseline" if slug == "baseline" else ""
            print(f"| {slug}{tag} | {cell['n']} | {fmt('OA')} | {delta} | "
                  f"{fmt('AA', False)} | {fmt('Kappa', False)} |")

    if incomplete:
        print("\n## Incomplete cells:")
        for item in incomplete:
            print(f"  - {item['cell']}: missing seeds {item['missing_seeds']}")
    else:
        print("\nAll variants complete across all seeds and epochs.")

    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

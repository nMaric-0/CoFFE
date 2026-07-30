"""Aggregate the combination study -> mean +/- std +/- 95% CI per combo per epoch.

For every combo (subset sizes 2..6 of the pool) reads the 3 seeds'
abl_eval_epoch<e>/results.json, computes across-seed OA/AA/Kappa stats, and adds:
  - dOA          : actual OA - baseline OA
  - add_dOA      : sum of member factors' individual OFAT dOA (additive expectation)
  - interaction  : dOA - add_dOA  (>0 = synergy, <0 = redundancy/interference)
Baseline OA and per-factor OFAT deltas are read from experiments/ablation_report.json.

Writes experiments/combo_report.json and prints one table per epoch sorted by OA
desc. Combos with no results yet are listed as pending (the study is staged).
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

from scripts import combo_config as cfg  # noqa: E402

METRICS = ["OA", "AA", "Kappa"]


def _read_means(name: str, epoch: int) -> Optional[Dict[str, float]]:
    path = (cfg.EXPERIMENTS_ROOT / name / "evaluations"
            / cfg.eval_name_for_epoch(epoch) / "results.json")
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


def _load_ofat():
    """baseline OA and per-factor OFAT dOA, per epoch, from ablation_report.json."""
    p = cfg.EXPERIMENTS_ROOT / "ablation_report.json"
    base = {e: None for e in cfg.EVAL_EPOCHS}
    fdelta = {e: {} for e in cfg.EVAL_EPOCHS}
    if not p.exists():
        return base, fdelta
    rep = json.load(open(p))
    cells = rep["cells"]
    for e in cfg.EVAL_EPOCHS:
        b = cells.get(f"baseline@e{e}", {}).get("stats", {}).get("OA")
        base[e] = b["mean"] if b else None
        for f in cfg.POOL:
            st = cells.get(f"{f}@e{e}", {}).get("stats", {}).get("OA")
            if st and base[e] is not None:
                fdelta[e][f] = st["mean"] - base[e]
    return base, fdelta


def main() -> int:
    base_oa, fdelta = _load_ofat()
    report: Dict[str, object] = {
        "experiment": "combination",
        "dataset": "houston",
        "pool": cfg.POOL,
        "epochs_evaluated": cfg.EVAL_EPOCHS,
        "seeds": cfg.SEEDS,
        "baseline_OA": base_oa,
        "cells": {}, "pending": [],
    }
    cells: Dict[str, dict] = report["cells"]      # type: ignore
    pending: List[str] = report["pending"]        # type: ignore

    for factors in cfg.all_combos():
        cname = cfg.combo_name(factors)
        for epoch in cfg.EVAL_EPOCHS:
            key = f"{cname}@e{epoch}"
            per_seed, missing = {}, []
            for seed in cfg.SEEDS:
                name = cfg.experiment_name(factors, seed)
                means = _read_means(name, epoch)
                (per_seed.__setitem__(str(seed), means) if means else missing.append(seed))
            cell = {"combo": cname, "factors": list(factors), "size": len(factors),
                    "epoch": epoch, "n": len(per_seed), "missing_seeds": missing, "stats": {}}
            for m in METRICS:
                vals = [per_seed[s][m] for s in per_seed if m in per_seed[s]]
                if vals:
                    cell["stats"][m] = _stats(vals)
            oa = cell["stats"].get("OA", {}).get("mean")
            if oa is not None and base_oa[epoch] is not None:
                cell["dOA"] = oa - base_oa[epoch]
                add = sum(fdelta[epoch].get(f, 0.0) for f in factors)
                cell["add_dOA"] = add
                cell["interaction"] = cell["dOA"] - add
            cells[key] = cell
            if not per_seed:
                pending.append(key)

    out_path = cfg.EXPERIMENTS_ROOT / "combo_report.json"

    print(f"\n# Combination study — Houston enhanced spatial HSI+LiDAR "
          f"({len(cfg.SEEDS)} seeds {cfg.SEEDS}; pool={cfg.POOL})\n")
    for epoch in cfg.EVAL_EPOCHS:
        b = base_oa[epoch]
        rows = [(k, c) for k, c in cells.items()
                if c["epoch"] == epoch and c["stats"].get("OA")]
        rows.sort(key=lambda kc: kc[1]["stats"]["OA"]["mean"], reverse=True)
        print(f"\n## Epoch {epoch}  (baseline OA {b:.2f})\n" if b else f"\n## Epoch {epoch}\n")
        print("| Combo | k | n | OA (mean±std,95%CI) | ΔOA | add-Δ | interaction |")
        print("|" + "---|" * 7)
        for k, c in rows:
            st = c["stats"]["OA"]
            oa = f"{st['mean']:.2f} ± {st['std']:.2f} (±{st['ci_95']:.2f})"
            d = f"{c.get('dOA', float('nan')):+.2f}" if "dOA" in c else "—"
            a = f"{c.get('add_dOA', float('nan')):+.2f}" if "add_dOA" in c else "—"
            it = f"{c.get('interaction', float('nan')):+.2f}" if "interaction" in c else "—"
            print(f"| {c['combo']} | {c['size']} | {c['n']} | {oa} | {d} | {a} | {it} |")

    # ---- paired 75% vs 90% masking, per architecture/loss stack ----
    MASK = "mask0p90"
    arch_tag = lambda S: ("-".join(cfg.TAGS[f] for f in S) or "baseline")

    def combo_oa(factors, epoch):
        st = cells.get(f"{cfg.combo_name(factors)}@e{epoch}", {}).get("stats", {}).get("OA")
        return st["mean"] if st else None

    mask_pairs = {}
    for epoch in cfg.EVAL_EPOCHS:
        rows = []
        for k, c in cells.items():
            if c["epoch"] != epoch or MASK not in c["factors"]:
                continue
            oa90 = c["stats"].get("OA", {}).get("mean")
            if oa90 is None:
                continue
            S = cfg._canonical([f for f in c["factors"] if f != MASK])
            if len(S) >= 2:
                oa75 = combo_oa(S, epoch)
            elif len(S) == 1:
                d = fdelta[epoch].get(S[0])
                oa75 = (base_oa[epoch] + d) if (d is not None and base_oa[epoch] is not None) else None
            else:
                oa75 = base_oa[epoch]
            rows.append({"stack": arch_tag(S), "factors": list(S),
                         "oa75": oa75, "oa90": oa90,
                         "delta_90_minus_75": (oa90 - oa75) if oa75 is not None else None})
        rows.sort(key=lambda r: r["oa90"], reverse=True)
        mask_pairs[str(epoch)] = rows
        print(f"\n## Masking 75% vs 90% per stack — epoch {epoch}\n")
        print("| Stack (arch/loss factors) | OA@75% | OA@90% | Δ(90−75) |")
        print("|" + "---|" * 4)
        for r in rows:
            f75 = "—" if r["oa75"] is None else f"{r['oa75']:.2f}"
            d = "—" if r["delta_90_minus_75"] is None else f"{r['delta_90_minus_75']:+.2f}"
            print(f"| {r['stack']} | {f75} | {r['oa90']:.2f} | {d} |")
    report["masking_pairs"] = mask_pairs

    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    done = sum(1 for c in cells.values() if c["n"] > 0) // len(cfg.EVAL_EPOCHS)
    total = len(cfg.all_combos())
    print(f"\nCombos with results: {done}/{total}.  Pending (not yet run): "
          f"{len(pending) // 1} cell-epochs.")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

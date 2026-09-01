#!/usr/bin/env python
"""Compile the MFT "faithful" experiments into a single JSON with metadata.

Targets the six runs ``mft_original_<dataset>_<variant>_faithful`` where
``dataset in {houston, muufl, trento}`` and ``variant in {spatial, mae}``.

For each run it collects:
    * pretrain_metadata.json   -> ["metadata"]
    * pretrain_config.yaml     -> ["config"]
    * evaluations/<eval>/      -> ["evaluations"][<eval>] with the full
          eval_config.json / eval_metadata.json / results.json payloads.

A flat ``summary`` table (one row per run, headline OA/AA/Kappa + key eval
settings) is also emitted up top for quick reading. Per-class blocks and every
nested result are preserved verbatim in the per-experiment records.

Usage:
    python scripts/reports/compile_mft_faithful_results.py \
        [--experiments-root experiments] \
        [--output results/mft_faithful_results.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DATASETS = ["houston", "muufl", "trento"]
VARIANTS = ["spatial", "mae"]


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _read_yaml(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)


def _metric_block(results: Optional[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    if isinstance(results, dict) and isinstance(results.get(key), dict):
        return results[key]
    return None


def collect(experiments_root: Path, eval_filter: Optional[str] = None) -> Dict[str, Any]:
    experiments: Dict[str, Any] = {}
    summary: List[Dict[str, Any]] = []
    missing: List[str] = []
    total_evals = 0

    for dataset in DATASETS:
        for variant in VARIANTS:
            name = f"mft_original_{dataset}_{variant}_faithful"
            exp_dir = experiments_root / name
            if not exp_dir.is_dir():
                missing.append(name)
                continue

            meta = _read_json(exp_dir / "pretrain_metadata.json")
            config = _read_yaml(exp_dir / "pretrain_config.yaml")

            evaluations: Dict[str, Any] = {}
            evals_dir = exp_dir / "evaluations"
            if evals_dir.is_dir():
                for eval_dir in sorted(evals_dir.iterdir()):
                    if not eval_dir.is_dir():
                        continue
                    if eval_filter and eval_filter not in eval_dir.name:
                        continue
                    evaluations[eval_dir.name] = {
                        "config": _read_json(eval_dir / "eval_config.json"),
                        "metadata": _read_json(eval_dir / "eval_metadata.json"),
                        "results": _read_json(eval_dir / "results.json"),
                        "path": str(eval_dir),
                    }
            total_evals += len(evaluations)

            experiments[name] = {
                "name": (meta or {}).get("name", name),
                "dataset": dataset,
                "variant": variant,
                "path": str(exp_dir),
                "metadata": meta,
                "config": config,
                "num_evaluations": len(evaluations),
                "evaluations": evaluations,
            }

            # Flat summary row(s) — one per evaluation.
            for eval_name, ev in evaluations.items():
                res = ev["results"]
                row: Dict[str, Any] = {
                    "experiment": name,
                    "dataset": dataset,
                    "variant": variant,
                    "eval_name": eval_name,
                }
                if isinstance(res, dict):
                    row.update(
                        {
                            "model_type": res.get("model_type"),
                            "distance_metric": res.get("distance_metric"),
                            "n_way": res.get("n_way"),
                            "k_shot": res.get("k_shot"),
                            "k_query": res.get("k_query"),
                            "num_episodes": res.get("num_episodes"),
                            "checkpoint": res.get("checkpoint"),
                        }
                    )
                    for metric in ("OA", "AA", "Kappa"):
                        blk = _metric_block(res, metric)
                        if blk:
                            row[f"{metric}_mean"] = blk.get("mean")
                            row[f"{metric}_std"] = blk.get("std")
                            row[f"{metric}_ci_95"] = blk.get("ci_95")
                summary.append(row)

    return {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "experiments_root": str(experiments_root),
        "selector": "mft_original_{dataset}_{variant}_faithful",
        "eval_filter": eval_filter,
        "datasets": DATASETS,
        "variants": VARIANTS,
        "num_experiments": len(experiments),
        "num_evaluations": total_evals,
        "missing": missing,
        "summary": summary,
        "experiments": experiments,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments-root", default="experiments", type=Path)
    ap.add_argument(
        "--eval-filter",
        default=None,
        help="Only include eval dirs whose name contains this substring "
        "(e.g. 'ep1500'). Default: include every eval dir.",
    )
    ap.add_argument(
        "--output",
        default="results/mft_faithful_results.json",
        type=Path,
    )
    args = ap.parse_args()

    data = collect(args.experiments_root, eval_filter=args.eval_filter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=False)

    print(f"Wrote {args.output}")
    print(
        f"  experiments: {data['num_experiments']}  "
        f"evaluations: {data['num_evaluations']}  "
        f"missing: {data['missing'] or 'none'}"
    )
    for row in data["summary"]:
        oa = row.get("OA_mean")
        oa_s = f"{oa:.2f}" if isinstance(oa, (int, float)) else "n/a"
        print(
            f"  {row['dataset']:<8} {row['variant']:<8} "
            f"OA={oa_s:>6}  ({row['eval_name']})"
        )


if __name__ == "__main__":
    main()

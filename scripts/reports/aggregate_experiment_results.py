#!/usr/bin/env python
"""Aggregate every experiment's results + metadata into a single JSON file.

Walks ``experiments/<name>/`` and, for each real experiment (the reference
``_example`` and anything with ``status == "example"`` is skipped), collects:

    * pretrain_metadata.json   -> ["metadata"]
    * pretrain_config.yaml     -> ["config"]
    * evaluations/<eval>/      -> ["evaluations"][<eval>] with the full
          eval_config.json / eval_metadata.json / results.json payloads.

The output is keyed by experiment name so a single file holds the complete
record of every run. Per-class blocks and cosine/euclidean sub-results are
preserved verbatim — nothing is flattened or dropped.

Usage:
    python scripts/reports/aggregate_experiment_results.py \
        [--experiments-root experiments] \
        [--output results/aggregated_results.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _read_yaml(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)


def _is_example(meta: dict[str, Any] | None, name: str) -> bool:
    if name == "_example":
        return True
    return isinstance(meta, dict) and meta.get("status") == "example"


def collect(experiments_root: Path) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    skipped: list[str] = []
    total_evals = 0

    for exp_dir in sorted(experiments_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        meta = _read_json(exp_dir / "pretrain_metadata.json")
        if _is_example(meta, exp_dir.name):
            skipped.append(exp_dir.name)
            continue

        config = _read_yaml(exp_dir / "pretrain_config.yaml")

        evaluations: dict[str, Any] = {}
        evals_dir = exp_dir / "evaluations"
        if evals_dir.is_dir():
            for eval_dir in sorted(evals_dir.iterdir()):
                if not eval_dir.is_dir():
                    continue
                evaluations[eval_dir.name] = {
                    "config": _read_json(eval_dir / "eval_config.json"),
                    "metadata": _read_json(eval_dir / "eval_metadata.json"),
                    "results": _read_json(eval_dir / "results.json"),
                    "path": str(eval_dir),
                }
        total_evals += len(evaluations)

        experiments[exp_dir.name] = {
            "name": (meta or {}).get("name", exp_dir.name),
            "path": str(exp_dir),
            "metadata": meta,
            "config": config,
            "num_evaluations": len(evaluations),
            "evaluations": evaluations,
        }

    return {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "experiments_root": str(experiments_root),
        "num_experiments": len(experiments),
        "num_evaluations": total_evals,
        "skipped": skipped,
        "experiments": experiments,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments-root", default="experiments", type=Path)
    ap.add_argument(
        "--output",
        default="results/aggregated_results.json",
        type=Path,
    )
    args = ap.parse_args()

    data = collect(args.experiments_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=False)

    n_with_evals = sum(1 for e in data["experiments"].values() if e["num_evaluations"] > 0)
    print(f"Wrote {args.output}")
    print(f"  experiments: {data['num_experiments']} ({n_with_evals} with >=1 evaluation)")
    print(f"  evaluations: {data['num_evaluations']}")
    print(f"  skipped:     {data['skipped']}")


if __name__ == "__main__":
    main()

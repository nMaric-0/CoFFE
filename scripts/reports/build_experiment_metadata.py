#!/usr/bin/env python
"""Build a distilled, paper-ready metadata record for every experiment run.

Unlike ``aggregate_experiment_results.py`` (which dumps every config/result
verbatim into one large file), this walks ``experiments/<name>/`` and produces a
*structured* summary aimed at writing an exhaustive experiments section:

    * identity     -- name, derived method family, description, git SHA, status
    * timing       -- start/finish, wall-clock seconds & hours (estimated from
                      checkpoint mtimes when a run is still flagged "running")
    * hardware     -- GPU model, device, seed, determinism
    * model        -- full architecture config block
    * pretrain     -- full pretraining / adaptation hyperparameters
    * training     -- epochs run, final train loss, best val loss, #checkpoints
    * evaluations  -- per eval: few-shot protocol, checkpoint/epoch used,
                      eval runtime, and OA/AA/Kappa (+cosine/euclidean split and
                      per-class accuracy when present)

A top-level ``summary`` block reports totals, GPU-hours, the date range, GPUs
used, and a per-family breakdown.

Usage:
    python scripts/reports/build_experiment_metadata.py \
        [--experiments-root experiments] \
        [--output results/experiment_metadata.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coffe.compat import normalize_model_name, normalize_objective

# --------------------------------------------------------------------------- #
# small readers / sanitisers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)  # tolerant of Infinity/NaN (non-standard JSON)


def _read_yaml(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)


def _clean(obj: Any) -> Any:
    """Recursively replace inf/-inf/nan with None so output is valid JSON."""
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _parse_iso(ts: str | None) -> _dt.datetime | None:
    if not ts:
        return None
    try:
        return _dt.datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _round(x: float | None, n: int = 2) -> float | None:
    return None if x is None else round(x, n)


# --------------------------------------------------------------------------- #
# derivations
# --------------------------------------------------------------------------- #
def _is_example(meta: dict[str, Any] | None, name: str) -> bool:
    if name == "_example":
        return True
    return isinstance(meta, dict) and meta.get("status") == "example"


def _method_family(
    name: str,
    config: dict[str, Any] | None,
    evals: dict[str, Any],
) -> str:
    """Group runs by the actual model/objective, not the (often misleading) name."""
    model = (config or {}).get("model") if isinstance(config, dict) else None
    pre = (config or {}).get("pretrain") if isinstance(config, dict) else None
    if isinstance(model, dict):
        mname = model.get("name")
        if mname == "hypersigma_dual":
            return "hypersigma_adapt" if model.get("adapt_mode") else "hypersigma"
        if mname == "mft_original":
            return "mft_original_mae"
        if normalize_model_name(mname, origin="pretrain_config.yaml") == "coffe":
            objective = normalize_objective(
                (pre or {}).get("objective") if isinstance(pre, dict) else None,
                origin="pretrain_config.yaml",
            )
            if objective == "mae":
                return "coffe_mae"
            return "coffe_simmim"
    # configs without a model block (ablation / notebook-only): use eval model_type
    model_types = {(e.get("model_type") or "") for e in evals.values()}
    if any("HyperSIGMA" in m for m in model_types):
        adapted = any((e.get("checkpoint") or "none") not in ("none", None) for e in evals.values())
        return "hypersigma_adapt" if adapted else "hypersigma_ablation"
    # meta-only configs / runs whose eval has no results yet: fall back to name
    low = name.lower()
    if "hypersigma" in low or "native" in low:
        if "ablation" in low:
            return "hypersigma_ablation"
        if any(
            t in low
            for t in (
                "adapt",
                "sem",
                "spatial_only",
                "spat_only",
                "spectral_only",
                "joint",
                "pca",
                "native",
            )
        ):
            return "hypersigma_adapt"
        return "hypersigma"
    return "unknown"


_EPOCH_RE = re.compile(r"epoch_(\d+)")


def _checkpoint_epoch(ckpt: str | None) -> Any | None:
    if not ckpt or ckpt == "none":
        return None
    base = Path(ckpt).name
    m = _EPOCH_RE.search(base)
    if m:
        return int(m.group(1))
    if "final" in base:
        return "final"
    return None


def _estimate_runtime(exp_dir: Path, started: _dt.datetime | None) -> dict[str, Any] | None:
    """Fallback wall-clock estimate from latest checkpoint mtime - start time."""
    if started is None:
        return None
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    mtimes = [p.stat().st_mtime for p in ckpt_dir.glob("*.pth")]
    if not mtimes:
        return None
    last = _dt.datetime.fromtimestamp(max(mtimes))
    secs = (last - started).total_seconds()
    if secs <= 0:
        return None
    return {
        "estimated_finished_at": last.isoformat(timespec="seconds"),
        "estimated_wallclock_seconds": _round(secs),
        "estimated_wallclock_hours": _round(secs / 3600.0, 3),
        "estimate_source": "latest_checkpoint_mtime",
    }


# --------------------------------------------------------------------------- #
# evaluation distillation
# --------------------------------------------------------------------------- #
_METRIC_KEYS = ("OA", "AA", "Kappa")


def _metric_block(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k in _METRIC_KEYS:
        v = d.get(k)
        if isinstance(v, dict):
            out[k] = {
                "mean": _round(v.get("mean"), 3),
                "std": _round(v.get("std"), 3),
                "ci_95": _round(v.get("ci_95"), 4),
            }
    return out


def _distill_eval(eval_dir: Path) -> dict[str, Any]:
    cfg = _read_json(eval_dir / "eval_config.json") or {}
    meta = _read_json(eval_dir / "eval_metadata.json") or {}
    res = _read_json(eval_dir / "results.json") or {}

    # eval runtime: prefer metadata wallclock, else derive from start/finish
    wall = meta.get("wallclock_seconds")
    started = _parse_iso(meta.get("started_at"))
    finished = _parse_iso(meta.get("finished_at"))
    if wall is None and started and finished:
        wall = (finished - started).total_seconds()

    ckpt = res.get("checkpoint") or cfg.get("checkpoint") or res.get("adapted_checkpoint")

    out: dict[str, Any] = {
        "eval_name": eval_dir.name,
        "status": meta.get("status", "complete" if res else "incomplete"),
        "has_results": bool(res),
        "dataset": res.get("dataset") or cfg.get("dataset"),
        "split": res.get("split") or cfg.get("split"),
        "model_type": res.get("model_type"),
        # few-shot protocol
        "protocol": {
            "n_way": res.get("n_way") or cfg.get("n_way"),
            "k_shot": res.get("k_shot") or cfg.get("k_shot"),
            "k_query": res.get("k_query") or cfg.get("k_query"),
            "num_episodes": res.get("num_episodes") or cfg.get("num_episodes"),
            "seed": res.get("seed") or cfg.get("seed"),
            "distance_metric": res.get("distance_metric") or cfg.get("distance_metric"),
            "temperature": res.get("temperature") or cfg.get("temperature"),
            "prototype_mode": res.get("prototype_mode") or cfg.get("prototype_mode"),
            "use_projection": cfg.get("use_projection"),
        },
        "checkpoint": ckpt,
        "checkpoint_epoch": _checkpoint_epoch(ckpt),
        "runtime": {
            "started_at": meta.get("started_at"),
            "finished_at": meta.get("finished_at"),
            "wallclock_seconds": _round(wall),
            "git_sha": meta.get("git_sha"),
        },
        "metrics": _metric_block(res),
    }

    # HyperSIGMA-specific geometry knobs (only when present)
    for k in ("mode", "input_fit", "native_geometry", "pad_anchor", "spat_patch_k"):
        if k in res:
            out.setdefault("hypersigma", {})[k] = res[k]

    # cosine / euclidean split, if the run reported both
    by_dist = {}
    for dk in ("cosine", "euclidean"):
        if isinstance(res.get(dk), dict):
            by_dist[dk] = _metric_block(res[dk])
    if by_dist:
        out["metrics_by_distance"] = by_dist

    # per-class accuracy (selected distance metric, top-level)
    if isinstance(res.get("per_class"), dict):
        names = res.get("class_names") or []
        per_class = {}
        for cid, stats in res["per_class"].items():
            label = cid
            try:
                idx = int(cid) - 1
                if 0 <= idx < len(names):
                    label = f"{cid}:{names[idx]}"
            except (ValueError, TypeError):
                pass
            if isinstance(stats, dict):
                per_class[label] = {
                    "accuracy": _round(stats.get("accuracy", stats.get("mean")), 3),
                    "std": _round(stats.get("std"), 3),
                    "ci_95": _round(stats.get("ci_95"), 4),
                }
            else:
                per_class[label] = stats
        out["per_class_accuracy"] = per_class

    return out


# --------------------------------------------------------------------------- #
# experiment distillation
# --------------------------------------------------------------------------- #
def _distill_experiment(exp_dir: Path) -> dict[str, Any]:
    meta = _read_json(exp_dir / "pretrain_metadata.json") or {}
    config = _read_yaml(exp_dir / "pretrain_config.yaml")
    config = config if isinstance(config, dict) else {}

    # evaluations
    evaluations: dict[str, Any] = {}
    evals_dir = exp_dir / "evaluations"
    if evals_dir.is_dir():
        for ed in sorted(p for p in evals_dir.iterdir() if p.is_dir()):
            evaluations[ed.name] = _distill_eval(ed)

    started = _parse_iso(meta.get("started_at"))
    finished = _parse_iso(meta.get("finished_at"))
    wall = meta.get("wallclock_seconds")
    if wall is None and started and finished:
        wall = (finished - started).total_seconds()

    timing: dict[str, Any] = {
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "wallclock_seconds": _round(wall),
        "wallclock_hours": _round(wall / 3600.0, 3) if wall else None,
    }
    if wall is None:  # still "running" / crashed metadata -> estimate
        est = _estimate_runtime(exp_dir, started)
        if est:
            timing.update(est)

    history = meta.get("history") or {}
    n_ckpt = (
        len(list((exp_dir / "checkpoints").glob("*.pth")))
        if (exp_dir / "checkpoints").is_dir()
        else 0
    )

    record: dict[str, Any] = {
        "name": meta.get("name", exp_dir.name),
        "method_family": _method_family(exp_dir.name, config, evaluations),
        "description": meta.get("description"),
        "status": meta.get("status", "unknown"),
        "git_sha": meta.get("git_sha"),
        "base_config_path": meta.get("base_config_path"),
        "path": str(exp_dir),
        "timing": timing,
        "hardware": {
            "gpu": meta.get("cuda_device_name"),
            "device": meta.get("resolved_device") or (config.get("hardware") or {}).get("device"),
            "seed": (config.get("hardware") or {}).get("seed"),
            "deterministic": (config.get("hardware") or {}).get("deterministic"),
        },
        "model": config.get("model"),
        "data": config.get("data"),
        "pretrain": config.get("pretrain"),
        "training": {
            "epochs_run": history.get("epochs_run"),
            "final_train_loss": history.get("final_train_loss"),
            "best_val_loss": history.get("best_val_loss"),
            "num_checkpoints": n_ckpt,
        },
        "overrides": meta.get("overrides"),
        "num_evaluations": len(evaluations),
        "num_evaluations_with_results": sum(1 for e in evaluations.values() if e["has_results"]),
        "evaluations": evaluations,
    }
    return record


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def build(experiments_root: Path) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    skipped: list[str] = []

    for exp_dir in sorted(p for p in experiments_root.iterdir() if p.is_dir()):
        meta = _read_json(exp_dir / "pretrain_metadata.json")
        # skip the reference example, helper/output dirs ("_analysis", ...), and
        # any dir that is not actually an experiment (no metadata, config, or evals)
        is_real = (
            (exp_dir / "pretrain_metadata.json").exists()
            or (exp_dir / "pretrain_config.yaml").exists()
            or (exp_dir / "evaluations").is_dir()
        )
        if _is_example(meta, exp_dir.name) or exp_dir.name.startswith("_") or not is_real:
            skipped.append(exp_dir.name)
            continue
        experiments[exp_dir.name] = _distill_experiment(exp_dir)

    # ---- summary roll-up -------------------------------------------------- #
    total_wall = 0.0
    family_counts: dict[str, int] = {}
    gpus: dict[str, int] = {}
    starts: list[_dt.datetime] = []
    n_eval = n_eval_done = 0
    status_counts: dict[str, int] = {}

    for e in experiments.values():
        family_counts[e["method_family"]] = family_counts.get(e["method_family"], 0) + 1
        status_counts[e["status"]] = status_counts.get(e["status"], 0) + 1
        w = e["timing"].get("wallclock_seconds") or e["timing"].get("estimated_wallclock_seconds")
        if w:
            total_wall += w
        g = e["hardware"].get("gpu")
        if g:
            gpus[g] = gpus.get(g, 0) + 1
        s = _parse_iso(e["timing"].get("started_at"))
        if s:
            starts.append(s)
        n_eval += e["num_evaluations"]
        n_eval_done += e["num_evaluations_with_results"]

    summary = {
        "num_experiments": len(experiments),
        "num_evaluations": n_eval,
        "num_evaluations_with_results": n_eval_done,
        "total_wallclock_seconds": _round(total_wall),
        "total_gpu_hours": _round(total_wall / 3600.0, 2),
        "experiments_by_family": dict(sorted(family_counts.items())),
        "experiments_by_status": dict(sorted(status_counts.items())),
        "gpus_used": gpus,
        "date_range": {
            "earliest_start": min(starts).isoformat(timespec="seconds") if starts else None,
            "latest_start": max(starts).isoformat(timespec="seconds") if starts else None,
        },
        "skipped": skipped,
        "note": (
            "Wall-clock for runs still flagged status=running is reported under "
            "timing.estimated_wallclock_seconds (from latest checkpoint mtime). "
            "total_gpu_hours includes those estimates."
        ),
    }

    return _clean(
        {
            "generated_from": str(experiments_root),
            "summary": summary,
            "experiments": experiments,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments-root", default="experiments", type=Path)
    ap.add_argument("--output", default="results/experiment_metadata.json", type=Path)
    args = ap.parse_args()

    data = build(args.experiments_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)

    s = data["summary"]
    print(f"Wrote {args.output}")
    print(f"  experiments: {s['num_experiments']}")
    print(
        f"  evaluations: {s['num_evaluations']} ({s['num_evaluations_with_results']} with results)"
    )
    print(f"  total GPU-hours (incl. estimates): {s['total_gpu_hours']}")
    print(f"  by family: {s['experiments_by_family']}")
    print(f"  by status: {s['experiments_by_status']}")


if __name__ == "__main__":
    main()

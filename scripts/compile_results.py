#!/usr/bin/env python3
"""Compile few-shot evaluation results into a single presentation-ready JSON.

Scans ``experiments/*/evaluations/*/results.json``, normalises the two result
schemas (MFT-CPEA-Cosine single-metric vs HyperSIGMA dual cosine/euclidean),
applies light curation (drops scratch/test runs and obviously mislabelled
evals, collapses repeat runs of the same config), and writes a curated headline
set to ``docs/presentation/RESULTS.json``.

This is a read-only aggregator: it only reads result files and writes the
output JSON. It does not run any model. Re-run it after new evaluations land to
refresh the compilation::

    python scripts/compile_results.py

The masking *regime* (spectral / spatial / combined / MAE) is a property of how
a model was *pretrained*, so it is derived from the experiment directory name,
not from the (sometimes mislabelled) evaluation name.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"
OUT = REPO / "docs" / "presentation" / "RESULTS.json"

DATASETS = ("houston", "trento", "muufl")
NATURAL_N_WAY = {"houston": 15, "trento": 6, "muufl": 11}
N_CLASSES = {"houston": 15, "trento": 6, "muufl": 11}

# Experiments that are scratch/test runs or otherwise not presentation-grade.
_TEST_MARKERS = ("test_run", "spatial_mask_test", "_example")


def _metric_block(block: dict) -> dict | None:
    """Pull {OA, AA, Kappa} -> {mean, ci_95} out of a results block."""
    if not block:
        return None
    out = {}
    for key in ("OA", "AA", "Kappa"):
        m = block.get(key)
        if isinstance(m, dict) and "mean" in m:
            out[key] = {"mean": round(m["mean"], 3), "ci_95": round(m.get("ci_95", 0.0), 3)}
    return out or None


def classify(exp: str, ev: str, data: dict):
    """Return (model, regime, modality) for a result, or None if excluded.

    ``regime`` is taken from the *experiment* name (pretraining config), since
    eval names occasionally relabel the same checkpoint.
    """
    exp_l, ev_l = exp.lower(), ev.lower()
    ds = data.get("dataset")
    mt = data.get("model_type")

    # Drop scratch/test runs and the placeholder experiment.
    if any(m in exp_l for m in _TEST_MARKERS):
        return None, "scratch/test run", None, None

    # Drop evals whose experiment names a *different* dataset than the result
    # records (mislabelled cross-dataset eval).
    named = [d for d in DATASETS if d in exp_l]
    if named and ds not in named:
        return None, f"dataset mismatch (dir says {named}, result says {ds})", None, None

    if mt == "HyperSIGMADual":
        mode = data.get("mode")
        if "spectral" in exp_l or mode == "spec_pool":
            regime = "spectral_only (spec_pool)"
        elif ("spat" in exp_l) or mode == "spat_pool":
            regime = "spatial_only (spat_pool)"
        else:  # baseline / joint_sem
            regime = "joint_sem (fused)"
        return "HyperSIGMA", regime, "HSI-only", None

    # MFT-CPEA-Cosine
    modality = "HSI-only" if "no_lidar" in exp_l else "HSI+LiDAR"
    if "_mae_" in exp_l or exp_l.endswith("_mae"):
        regime = "MAE"
    else:
        has_spec = "spectral" in exp_l
        has_spat = "spatial" in exp_l or "_spat" in exp_l
        if has_spec and has_spat:
            regime = "Enhanced: combined"
        elif has_spat:
            regime = "Enhanced: spatial"
        elif has_spec:
            regime = "Enhanced: spectral"
        else:
            regime = "Enhanced"
    return "MFT-CPEA", regime, modality, None


def metrics_of(data: dict) -> dict:
    """Normalise metrics to {metric_name: {OA,AA,Kappa}}.

    HyperSIGMA results carry both cosine and euclidean blocks; MFT-CPEA carry a
    single block under its ``distance_metric``.
    """
    if data.get("model_type") == "HyperSIGMADual":
        out = {}
        for name in ("cosine", "euclidean"):
            mb = _metric_block(data.get(name))
            if mb:
                out[name] = mb
        return out
    dm = data.get("distance_metric") or "euclidean"
    mb = _metric_block(data)
    return {dm: mb} if mb else {}


def main() -> None:
    records: list[dict] = []
    excluded: list[dict] = []

    for path in sorted(EXPERIMENTS.glob("*/evaluations/*/results.json")):
        rel = path.relative_to(REPO).as_posix()
        exp, ev = path.parts[-4], path.parts[-2]
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            excluded.append({"path": rel, "reason": f"unreadable: {exc}"})
            continue

        model, regime, modality, _ = classify(exp, ev, data)
        if model is None:  # excluded; `regime` holds the reason
            excluded.append({"path": rel, "reason": regime})
            continue

        mets = metrics_of(data)
        if not mets:
            excluded.append({"path": rel, "reason": "no OA/AA/Kappa block"})
            continue

        records.append(
            {
                "dataset": data.get("dataset"),
                "model": model,
                "regime": regime,
                "modality": modality,
                "n_way": data.get("n_way"),
                "k_shot": data.get("k_shot"),
                "k_query": data.get("k_query"),
                "num_episodes": data.get("num_episodes"),
                "metrics": mets,
                "experiment": exp,
                "eval_name": ev,
                "source": rel,
                "mtime": int(path.stat().st_mtime),
            }
        )

    # Collapse repeat runs of the same (dataset, model, regime, modality):
    # representative = matches natural n-way, then most episodes, then newest.
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["dataset"], r["model"], r["regime"], r["modality"]), []).append(r)

    datasets_out: dict[str, dict] = {
        ds: {"n_classes": N_CLASSES[ds], "natural_n_way": NATURAL_N_WAY[ds], "entries": []}
        for ds in DATASETS
    }

    for (ds, *_), runs in sorted(groups.items()):
        runs.sort(
            key=lambda r: (
                r["n_way"] == NATURAL_N_WAY.get(ds),
                r["num_episodes"] or 0,
                r["mtime"],
            ),
            reverse=True,
        )
        rep, *others = runs
        entry = {
            "model": rep["model"],
            "regime": rep["regime"],
            "modality": rep["modality"],
            "eval_protocol": {
                "n_way": rep["n_way"],
                "k_shot": rep["k_shot"],
                "k_query": rep["k_query"],
                "num_episodes": rep["num_episodes"],
            },
            "metrics": rep["metrics"],
            "source": rep["source"],
        }
        if others:
            entry["other_runs"] = [
                {
                    "source": o["source"],
                    "num_episodes": o["num_episodes"],
                    "metrics": {m: blk.get("OA") for m, blk in o["metrics"].items()},
                }
                for o in others
            ]
        datasets_out.setdefault(
            ds, {"n_classes": N_CLASSES.get(ds), "natural_n_way": NATURAL_N_WAY.get(ds), "entries": []}
        )["entries"].append(entry)

    # Stable, presentation-friendly ordering of entries within each dataset.
    model_order = {"MFT-CPEA": 0, "HyperSIGMA": 1}
    for ds in datasets_out.values():
        ds["entries"].sort(key=lambda e: (model_order.get(e["model"], 9), e["regime"], e["modality"]))

    out = {
        "_meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_by": "scripts/compile_results.py",
            "description": (
                "Curated headline few-shot results for the CoFFE project. Metrics are "
                "mean +/- 95% CI over episodes (OA == AA when query sets are class-balanced). "
                "Regime is the pretraining masking config (derived from the experiment dir)."
            ),
            "metric_keys": "OA = Overall Accuracy, AA = Average Accuracy, Kappa = Cohen's kappa (all %).",
            "eval_protocol": {
                "MFT-CPEA": "5-shot, k_query=100, 1000 episodes, euclidean prototypes.",
                "HyperSIGMA": "5-shot, k_query=30 (Houston) / 100 (Trento), 600-2000 episodes; cosine & euclidean reported.",
            },
            "curation": (
                "Dropped scratch/test runs (*_test_run*, *_spatial_mask_test*), the _example "
                "placeholder, and one mislabelled cross-dataset eval. Repeat runs of the same "
                "config are collapsed to a representative (natural n-way, most episodes, newest); "
                "alternates are kept under 'other_runs'."
            ),
            "kept_results": len(records),
            "excluded_results": len(excluded),
        },
        "datasets": datasets_out,
        "excluded": sorted(excluded, key=lambda e: e["path"]),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT.relative_to(REPO)}: {len(records)} kept, {len(excluded)} excluded.")
    for ds, blk in datasets_out.items():
        print(f"  {ds}: {len(blk['entries'])} headline entries")


if __name__ == "__main__":
    main()

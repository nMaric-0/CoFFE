#!/usr/bin/env python3
"""Compile few-shot evaluation results into a single presentation-ready JSON.

Scans ``experiments/*/evaluations/*/results.json``, normalises the two result
schemas (CoFFE/MFT single-metric vs HyperSIGMA dual cosine/euclidean),
applies light curation (drops scratch/test runs and obviously mislabelled
evals, collapses repeat runs of the same config), and writes a curated headline
set to ``docs/presentation/RESULTS.json``.

This is a read-only aggregator: it only reads result files and writes the
output JSON. It does not run any model. Re-run it after new evaluations land to
refresh the compilation::

    python scripts/compile_results.py

The masking *regime* (SimMIM band / token / band+token, or MAE) is a property of
how a model was *pretrained*, so it is derived from the experiment directory
name, not from the (sometimes mislabelled) evaluation name. Those directory
names use the pre-paper vocabulary (``spectral``/``spatial``/``both``) and are
frozen (PAPER_CANON §7.3): the substring tests below match them as-is and the
canonical regime label is what gets written out.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coffe.compat import normalize_model_type

EXPERIMENTS = REPO / "experiments"
DEFAULT_OUT = REPO / "docs" / "presentation" / "RESULTS.json"

DATASETS = ("houston", "trento", "muufl")
NATURAL_N_WAY = {"houston": 15, "trento": 6, "muufl": 11}
N_CLASSES = {"houston": 15, "trento": 6, "muufl": 11}

#: The two frozen on-disk conventions that mean "no LiDAR": the canonical
#: Table 2 runs use ``_no_lidar``, the 5-seed significance runs use
#: ``_hsi_only`` (``sig_significance_config.py`` names them
#: ``<scene>_hsi_only_<variant>_seed<s>`` and
#: ``<scene>_enhanced_mae_hsi_only_seed<s>``). Both are DO-NOT-RENAME
#: (PAPER_CANON §7.3), so this reader accepts both. Before the phase-6 gate only
#: the first was matched, which labelled every significance HSI-only run
#: "HSI+LiDAR".
HSI_ONLY_MARKERS = ("no_lidar", "hsi_only")

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

    ``model`` is one of ``"CoFFE"``, ``"MFT (original)"`` or ``"HyperSIGMA"``.

    ``regime`` is taken from the *experiment* name (pretraining config), since
    eval names occasionally relabel the same checkpoint.
    """
    exp_l = exp.lower()
    ds = data.get("dataset")
    # Old results.json files record the pre-paper class names
    # (PAPER_CANON §8 D16); normalise before any comparison.
    mt = normalize_model_type(data.get("model_type"), origin=f"{exp}/{ev}/results.json")

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

    # CoFFE and the MFT control share this result schema, so the model has to
    # come from `model_type`, not from the schema. Both used to get the same
    # label, which let an MFT-control run stand in as the representative of a
    # CoFFE cell.
    model = "MFT (original)" if mt == "MFTOriginal" or "mft_original" in exp_l else "CoFFE"

    modality = "HSI-only" if any(m in exp_l for m in HSI_ONLY_MARKERS) else "HSI+LiDAR"
    if "_mae_" in exp_l or exp_l.endswith("_mae"):
        regime = "MAE"
    else:
        # Frozen dir names: "spectral" = band masking, "spatial" = token masking.
        has_spec = "spectral" in exp_l
        has_spat = "spatial" in exp_l or "_spat" in exp_l
        if has_spec and has_spat:
            regime = "SimMIM band+token"
        elif has_spat:
            regime = "SimMIM token"
        elif has_spec:
            regime = "SimMIM band"
        else:
            regime = "SimMIM"
    return model, regime, modality, None


def metrics_of(data: dict) -> dict:
    """Normalise metrics to {metric_name: {OA,AA,Kappa}}.

    HyperSIGMA results carry both cosine and euclidean blocks; CoFFE/MFT carry a
    single block under its ``distance_metric``.
    """
    if normalize_model_type(data.get("model_type")) == "HyperSIGMADual":
        out = {}
        for name in ("cosine", "euclidean"):
            mb = _metric_block(data.get(name))
            if mb:
                out[name] = mb
        return out
    dm = data.get("distance_metric") or "euclidean"
    mb = _metric_block(data)
    return {dm: mb} if mb else {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="output path (default: %(default)s)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite --out if it already exists (it is a committed artifact)",
    )
    return parser.parse_args(argv)


def _guard_out(out: Path, force: bool) -> None:
    """Refuse to clobber a committed artifact without an explicit ``--force``.

    Checked before any work is done, so a mistaken invocation costs nothing.
    Phase-6 gate decision: these report scripts used to take no arguments and
    ignore ``argv``, so a bare ``--help`` during CLI smoke regenerated their
    artifact — twice.
    """
    if out.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite {out}\n"
            "It is a committed artifact. Pass --force to regenerate it, or "
            "--out PATH to write elsewhere."
        )


def main() -> None:
    args = _parse_args()
    _guard_out(args.out, args.force)

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
            ds,
            {"n_classes": N_CLASSES.get(ds), "natural_n_way": NATURAL_N_WAY.get(ds), "entries": []},
        )["entries"].append(entry)

    # Stable, presentation-friendly ordering of entries within each dataset.
    model_order = {"CoFFE": 0, "MFT (original)": 1, "HyperSIGMA": 2}
    for ds in datasets_out.values():
        ds["entries"].sort(
            key=lambda e: (model_order.get(e["model"], 9), e["regime"], e["modality"])
        )

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
                "CoFFE": "5-shot, k_query=100, 1000 episodes, Euclidean nearest-class-mean.",
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.out}: {len(records)} kept, {len(excluded)} excluded.")
    for ds, blk in datasets_out.items():
        print(f"  {ds}: {len(blk['entries'])} headline entries")


if __name__ == "__main__":
    main()

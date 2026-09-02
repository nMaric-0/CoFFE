#!/usr/bin/env python
"""Gather a focused, single-file results JSON for a specific requested set of runs.

Groups (see user request):
  * CoFFE SimMIM/MAE baselines: houston_enhanced_spatial_no_lidar, muufl_mae_no_lidar
    (experiment dir names are frozen, PAPER_CANON §7.3)
  * HyperSIGMA PCA-100 spatial_only  (all 3 datasets, most recent run)
  * HyperSIGMA PCA-100 joint_sem     (all 3 datasets, most recent run; PCA-100 only)
  * HyperSIGMA spectral_only         (all 3 datasets; NO PCA-100)
  * HyperSIGMA native ablation       (all 12 evals + experiment metadata)
  * HyperSIGMA native_sem_pad        (houston + placeholders for trento/muufl)
  * MFT-original MAE                 (houston/muufl/trento, NOT the *_faithful runs)
  * MFT-original spatial             (houston/muufl/trento, NOT the *_faithful runs)

Missing results are emitted as explicit placeholders (status="MISSING") with a note.
Output: results/gathered_results.json
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
RESULTS = ROOT / "results"


def read_json(p: Path) -> Any | None:
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        return {"__read_error__": str(e)}


def metrics_only(results: dict | None) -> dict | None:
    """Pull the headline metrics + per-class out of a results.json."""
    if not results:
        return None
    keep = [
        "model_type",
        "distance_metric",
        "temperature",
        "prototype_mode",
        "checkpoint",
        "dataset",
        "split",
        "n_way",
        "k_shot",
        "k_query",
        "seeds",
        "num_episodes",
        "OA",
        "AA",
        "Kappa",
        "per_class",
    ]
    return {k: results[k] for k in keep if k in results}


def load_eval(exp: str, eval_subdir: str) -> dict[str, Any]:
    """Load one evaluation, returning a placeholder dict if anything is missing."""
    evdir = EXP / exp / "evaluations" / eval_subdir
    res_path = evdir / "results.json"
    if not res_path.exists():
        return {
            "status": "MISSING",
            "experiment": exp,
            "eval_subdir": eval_subdir,
            "eval_path": str(evdir.relative_to(ROOT)) if evdir.exists() else None,
            "note": "PLACEHOLDER - results.json not found",
        }
    results = read_json(res_path)
    md = read_json(evdir / "eval_metadata.json") or {}
    oa = results.get("OA", {}) if isinstance(results, dict) else {}
    return {
        "status": "OK",
        "experiment": exp,
        "eval_subdir": eval_subdir,
        "eval_path": str(evdir.relative_to(ROOT)),
        "eval_finished_at": md.get("finished_at"),
        "eval_started_at": md.get("started_at"),
        "results_mtime": _dt.datetime.fromtimestamp(os.path.getmtime(res_path)).isoformat(
            timespec="seconds"
        ),
        "OA_mean": oa.get("mean") if isinstance(oa, dict) else oa,
        "results": metrics_only(results),
    }


def placeholder(reason: str, **extra) -> dict[str, Any]:
    d = {"status": "MISSING", "note": "PLACEHOLDER - " + reason}
    d.update(extra)
    return d


def exp_metadata(exp: str) -> dict | None:
    return read_json(EXP / exp / "pretrain_metadata.json")


# ---------------------------------------------------------------------------
report: dict[str, Any] = {
    "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    "experiments_root": "experiments",
    "description": (
        "Focused gather of requested runs. HyperSIGMA 'adapt_*' dirs hold adaptation "
        "checkpoints; the few-shot eval results live in the sibling non-'adapt' dirs - "
        "those eval dirs are the ones loaded here. 'Most recent run' resolved per group; "
        "spatial/joint_sem restricted to the PCA-100 variants, spectral_only to non-PCA-100. "
        "Missing items are explicit placeholders."
    ),
    "groups": {},
}
G = report["groups"]

# --- 1. CoFFE SimMIM / MAE baselines --------------------------------------
G["coffe_baselines"] = {
    "houston_enhanced_spatial_no_lidar": load_eval(
        "houston_enhanced_spatial_no_lidar", "houston_enhanced_spatial_no_lidar_eval"
    ),
    "muufl_mae_no_lidar": load_eval("muufl_mae_no_lidar", "muufl_mae_no_lidar_eval"),
}

# --- 2. HyperSIGMA PCA-100 spatial_only (most recent run) -----------------
G["hypersigma_pca100_spatial_only"] = {
    "_note": "spatial branch, PCA-100 (Houston real PCA 144->100; MUUFL/Trento "
    "SpectralResample 64/63->100). Adapt dirs: hypersigma_adapt_<ds>_pca100_spatial_only_run1.",
    "houston": load_eval(
        "hypersigma_houston_pca100_spatial_only_run1",
        "hypersigma_houston_pca100_C15way_5shot_adapted_spatial_only_run1",
    ),
    "muufl": load_eval(
        "hypersigma_muufl_pca100_spatial_only_run1",
        "hypersigma_muufl_pca100_C11way_5shot_adapted_spatial_only_run1",
    ),
    "trento": load_eval(
        "hypersigma_trento_pca100_spatial_only_run1",
        "hypersigma_trento_pca100_C6way_5shot_adapted_spatial_only_run1",
    ),
}

# --- 3. HyperSIGMA PCA-100 joint_sem (most recent run, PCA-100 only) ------
G["hypersigma_pca100_joint_sem"] = {
    "_note": "joint_sem (SEM fusion, 512-d), PCA-100 variant only.",
    "houston": load_eval(
        "hypersigma_houston_pca100_joint_sem_run1",
        "hypersigma_houston_pca100_C15way_5shot_adapted_joint_sem_run1",
    ),
    "muufl": placeholder(
        "MUUFL PCA-100 joint_sem was never evaluated: adapt run "
        "hypersigma_adapt_muufl_pca100_joint_sem_run1 reports complete but no eval dir "
        "(hypersigma_muufl_pca100_joint_sem_run1) exists.",
        adapt_dir="hypersigma_adapt_muufl_pca100_joint_sem_run1",
    ),
    "trento": load_eval(
        "hypersigma_trento_pca100_joint_sem_run1",
        "hypersigma_trento_pca100_C6way_5shot_adapted_joint_sem_run1",
    ),
}

# --- 4. HyperSIGMA spectral_only (NO PCA-100) -----------------------------
G["hypersigma_spectral_only"] = {
    "_note": "spectral branch only, NOT PCA-100. Adapt dirs: "
    "hypersigma_adapt_<ds>_spectral_only_run1 (checkpoints). Eval dirs: "
    "hypersigma_<ds>_spectral_only_run1.",
    "houston": placeholder(
        "No adapted-houston spectral_only eval dir exists "
        "(hypersigma_houston_spectral_only_run1 absent). The adapt checkpoint exists "
        "(hypersigma_adapt_houston_spectral_only_run1, complete, 3000 epochs) but was "
        "never few-shot evaluated. Only an UN-adapted baseline exists: "
        "hypersigma_baseline_spectral_only_run1 (OA 15.13).",
        adapt_dir="hypersigma_adapt_houston_spectral_only_run1",
    ),
    "muufl": placeholder(
        "MUUFL spectral_only eval did not produce results.json (eval dir "
        "hypersigma_muufl_spectral_only_run1/.../adapted_spectral_only_run1 exists with "
        "a plots/ folder but no results.json; pretrain_metadata status='running').",
        eval_path="hypersigma_muufl_spectral_only_run1/evaluations/"
        "hypersigma_muufl_C11way_5shot_adapted_spectral_only_run1",
    ),
    "trento": load_eval(
        "hypersigma_trento_spectral_only_run1",
        "hypersigma_trento_C6way_5shot_adapted_spectral_only_run1",
    ),
}

# --- 5. HyperSIGMA native ablation (all 12 evals + metadata) --------------
ABL = "hypersigma_native_ablation_run1"
abl_evals = {}
for ds in ("houston", "muufl", "trento"):
    for branch in ("spatial", "spectral"):
        for geom in ("pad", "upscale"):
            sub = f"native_{ds}_{branch}_{geom}"
            abl_evals[sub] = load_eval(ABL, sub)
G["hypersigma_native_ablation"] = {
    "_note": "Off-the-shelf FROZEN HyperSIGMA single-branch feature extractor; 11x11 "
    "patch fit to native 64x64 by zero-PAD or bicubic UPSCALE. 3 datasets x "
    "{spatial,spectral} x {pad,upscale} = 12 evals.",
    "experiment_metadata": exp_metadata(ABL),
    "evaluations": abl_evals,
}

# --- 6. HyperSIGMA native_sem_pad (houston + placeholders) ----------------
G["hypersigma_native_sem_pad"] = {
    "_note": "SEM-only adaptation, 11x11 padded to 64x64; 512-d fused feature.",
    "houston": load_eval("hypersigma_native_sem_pad_run1", "native_sem_pad_houston"),
    "trento": placeholder(
        "No native_sem_pad Trento run/eval completed. Config exists "
        "(configs/hypersigma/trento_backbonenative_pad_sem_only.yaml) but no experiment dir."
    ),
    "muufl": placeholder(
        "No native_sem_pad MUUFL run/eval completed. Config exists "
        "(configs/hypersigma/muufl_backbonenative_pad_sem_only.yaml) but no experiment dir."
    ),
}

# --- 7. MFT-original MAE (not faithful) -----------------------------------
G["mft_original_mae"] = {
    "_note": "MFT-original masked-autoencoder pretrain runs; the *_faithful runs are EXCLUDED.",
    "houston": load_eval("mft_original_houston_mae_run1", "mft_original_houston_mae_eval"),
    "muufl": load_eval("mft_original_muufl_mae_run1", "mft_original_muufl_mae_eval"),
    "trento": load_eval("mft_original_trento_mae_run1", "mft_original_trento_mae_eval"),
}

# --- 8. MFT-original spatial (not faithful) -------------------------------
G["mft_original_spatial"] = {
    "_note": "MFT-original spatial-MAE pretrain runs; the *_faithful runs are EXCLUDED.",
    "houston": load_eval("mft_original_houston_spatial_run1", "mft_original_houston_spatial_eval"),
    "muufl": placeholder(
        "MUUFL spatial eval has no results.json (mft_original_muufl_spatial, "
        "pretrain_metadata status='running').",
        experiment="mft_original_muufl_spatial",
    ),
    "trento": placeholder(
        "Trento spatial eval has no results.json (mft_original_trento_spatial status='failed'; "
        "mft_original_trento_spatial_run1 status='running'). No completed eval.",
        experiments=["mft_original_trento_spatial", "mft_original_trento_spatial_run1"],
    ),
}

# ---------------------------------------------------------------------------
out = RESULTS / "gathered_results.json"
with out.open("w") as f:
    json.dump(report, f, indent=2)

# summary
n_ok = n_missing = 0


def walk(o):
    global n_ok, n_missing
    if isinstance(o, dict):
        st = o.get("status")
        if st == "OK":
            n_ok += 1
        elif st == "MISSING":
            n_missing += 1
        for v in o.values():
            walk(v)


walk(G)
print(f"Wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
print(f"  OK results: {n_ok}   MISSING placeholders: {n_missing}")

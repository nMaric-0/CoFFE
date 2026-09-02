#!/usr/bin/env python
"""Gather VERBATIM raw config/metadata/results for the HyperSIGMA native-sem and
PCA-100 experiment families into a single JSON blob.

This is the *raw data* half of the experiment report. Nothing is flattened or
dropped: every pretrain_config.yaml, pretrain_overrides.yaml, pretrain_metadata.json,
README.md, and per-evaluation eval_config.json / eval_metadata.json / results.json
is embedded as-is. The companion human/analysis layer (per-experiment verified
design write-ups produced by the audit agents) lives in experiments/_analysis/*.json
and is merged in by build_native_pca100_report.py.

Output: results/_report_raw.json
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
RESULTS = ROOT / "results"

# The 15 experiment dirs that make up the native-sem + PCA-100 families.
EXPERIMENT_DIRS = [
    # native full-band ablation (pad vs upscale x spatial/spectral x 3 datasets)
    "hypersigma_native_ablation_run1",
    # native semantic (SEM) family - Houston
    "hypersigma_adapt_houston_native_sem_run1",
    "hypersigma_native_sem_run1",
    "hypersigma_native_sem_pad_run1",
    # PCA-100 adapt runs
    "hypersigma_adapt_houston_pca100_joint_sem_run1",
    "hypersigma_adapt_houston_pca100_spatial_only_run1",
    "hypersigma_adapt_muufl_pca100_joint_sem_run1",
    "hypersigma_adapt_muufl_pca100_spatial_only_run1",
    "hypersigma_adapt_trento_pca100_joint_sem_run1",
    "hypersigma_adapt_trento_pca100_spatial_only_run1",
    # PCA-100 eval runs (consume the adapted checkpoints)
    "hypersigma_houston_pca100_joint_sem_run1",
    "hypersigma_houston_pca100_spatial_only_run1",
    "hypersigma_muufl_pca100_spatial_only_run1",
    "hypersigma_trento_pca100_joint_sem_run1",
    "hypersigma_trento_pca100_spatial_only_run1",
]

# Base configs referenced by these families, embedded verbatim for provenance.
BASE_CONFIGS = [
    "configs/hypersigma/houston_patchnative_joint_sem.yaml",
    "configs/hypersigma/houston_backbonenative_upscale_sem_only.yaml",
    "configs/hypersigma/houston_backbonenative_pad_sem_only.yaml",
    "configs/hypersigma/houston_patchnative_pca100_joint_sem.yaml",
    "configs/hypersigma/muufl_backbonenative_pad_sem_only.yaml",
    "configs/hypersigma/trento_backbonenative_pad_sem_only.yaml",
]


def read_json(p: Path) -> Any | None:
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def read_yaml(p: Path) -> Any | None:
    if not p.exists():
        return None
    with p.open() as f:
        return yaml.safe_load(f)


def read_text(p: Path) -> str | None:
    if not p.exists():
        return None
    return p.read_text()


def list_checkpoints(ckpt_dir: Path) -> dict:
    if not ckpt_dir.is_dir():
        return {"present": False, "files": [], "has_final": False, "max_epoch": None}
    files = sorted(p.name for p in ckpt_dir.iterdir() if p.is_file())
    epochs = []
    for f in files:
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth"):
            with contextlib.suppress(ValueError):
                epochs.append(int(f[len("checkpoint_epoch_") : -len(".pth")]))
    return {
        "present": True,
        "files": files,
        "has_final": "checkpoint_final.pth" in files,
        "has_interrupted": "checkpoint_interrupted.pth" in files,
        "max_epoch": max(epochs) if epochs else None,
    }


def collect_experiment(exp_dir: Path) -> dict:
    out: dict = {
        "name": exp_dir.name,
        "path": str(exp_dir.relative_to(ROOT)),
        "exists": exp_dir.is_dir(),
        "readme": read_text(exp_dir / "README.md"),
        "pretrain_config": read_yaml(exp_dir / "pretrain_config.yaml"),
        "pretrain_overrides": read_yaml(exp_dir / "pretrain_overrides.yaml"),
        "pretrain_metadata": read_json(exp_dir / "pretrain_metadata.json"),
        "checkpoints": list_checkpoints(exp_dir / "checkpoints"),
        "evaluations": {},
    }
    evals_dir = exp_dir / "evaluations"
    if evals_dir.is_dir():
        for ev in sorted(evals_dir.iterdir()):
            if not ev.is_dir():
                continue
            out["evaluations"][ev.name] = {
                "path": str(ev.relative_to(ROOT)),
                "eval_config": read_json(ev / "eval_config.json"),
                "eval_metadata": read_json(ev / "eval_metadata.json"),
                "results": read_json(ev / "results.json"),
                "has_results": (ev / "results.json").exists(),
            }
    out["num_evaluations"] = len(out["evaluations"])
    return out


def main() -> None:
    experiments = {name: collect_experiment(EXP / name) for name in EXPERIMENT_DIRS}
    base_configs = {c: read_yaml(ROOT / c) for c in BASE_CONFIGS}

    blob = {
        "experiments_root": "experiments",
        "num_experiments": len(experiments),
        "experiment_dirs": EXPERIMENT_DIRS,
        "base_configs": base_configs,
        "experiments": experiments,
    }
    out_path = RESULTS / "_report_raw.json"
    with out_path.open("w") as f:
        json.dump(blob, f, indent=2, sort_keys=False)

    n_eval = sum(e["num_evaluations"] for e in experiments.values())
    missing = [n for n, e in experiments.items() if not e["exists"]]
    print(f"Wrote {out_path}")
    print(f"  experiments: {len(experiments)} ({n_eval} evaluation subdirs total)")
    if missing:
        print(f"  MISSING dirs: {missing}")


if __name__ == "__main__":
    main()

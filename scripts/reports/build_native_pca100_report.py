#!/usr/bin/env python
"""Build the thorough HyperSIGMA native-SEM + PCA-100 experiment report.

Merges:
  * results/_report_raw.json            (verbatim configs / metadata / results)
  * experiments/_analysis/*.json        (per-experiment verified design write-ups
                                          produced by the audit agents)
with a hand-authored overview / methodology section and a flat results table.

Output: results/hypersigma_native_sem_pca100_report.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
RESULTS = ROOT / "results"
DEFAULT_OUT = RESULTS / "hypersigma_native_sem_pca100_report.json"

# analysis experiment-name -> raw experiment-dir alias (agents occasionally keyed
# the eval object by the eval subdir name rather than the experiment dir).
NAME_ALIAS = {
    "hypersigma_trento_pca100_C6way_5shot_adapted_spatial_only_run1": "hypersigma_trento_pca100_spatial_only_run1",
}

# ---------------------------------------------------------------------------
# Hand-authored overview (grounded in the agent audits + code tracing).
# ---------------------------------------------------------------------------
OVERVIEW: dict[str, Any] = {
    "title": "HyperSIGMA native-SEM and PCA-100 few-shot experiments",
    "what_this_covers": (
        "All HyperSIGMA experiments in the 'native semantic (SEM)' and 'PCA-100' "
        "families: the native full-band pad-vs-upscale ablation, the SEM-adaptation "
        "runs, and the PCA-100 11x11-adapted runs (spatial_only and joint_sem) across "
        "Houston, MUUFL and Trento. For every run the report carries the verbatim "
        "pretrain/adapt config, metadata, eval config/metadata and full results.json, "
        "plus a verified write-up of purpose and design (README claims checked against "
        "the actual configs and the code that determines input handling)."
    ),
    "common_backbone": (
        "HyperSIGMA dual-branch ViT-B (a SpatViT spatial branch + a SpecViT spectral "
        "branch), ~180M parameters, loaded from the official HyperSIGMA pretrained "
        "checkpoints. In every experiment the two ViT-B transformer bodies are FROZEN; "
        "only lightweight, randomly-initialised front-ends / decoders / fusion modules "
        "are ever trained (see 'adaptation_modes')."
    ),
    "datasets": {
        "houston": {
            "hsi_bands": 144,
            "num_classes": 15,
            "note": "Houston2013; spatial branch uses a real PCA 144->100.",
        },
        "muufl": {
            "hsi_bands": 64,
            "num_classes": 11,
            "note": "PCA-100 impossible from 64 bands -> SpectralResample 64->100.",
        },
        "trento": {
            "hsi_bands": 63,
            "num_classes": 6,
            "note": "PCA-100 impossible from 63 bands -> SpectralResample 63->100.",
        },
    },
    "experiment_families": {
        "native_full_band": {
            "members": ["hypersigma_native_ablation_run1"],
            "description": (
                "Off-the-shelf, UN-adapted (frozen) HyperSIGMA used as a feature "
                "extractor. A single branch is used at a time (spatial_only OR "
                "spectral_only, no SEM fusion). The 11x11 few-shot patch is fit to the "
                "backbone's native 64x64 input either by centered zero-PADDING or by "
                "bicubic UPSCALING. This is the baseline that the SEM / PCA-100 "
                "adaptations are meant to improve on. Ablates pad-vs-upscale x "
                "spatial-vs-spectral x {houston,muufl,trento} = 12 evaluations."
            ),
            "feature_dim": 768,
        },
        "native_sem": {
            "members": [
                "hypersigma_adapt_houston_native_sem_run1",
                "hypersigma_native_sem_run1",
                "hypersigma_native_sem_pad_run1",
            ],
            "description": (
                "Keeps the native 64x64 geometry (11x11 padded or upscaled to 64x64) "
                "but ADAPTS the model with continued MAE using adapt_mode='sem_only': "
                "only the SEM (Spatial-Spectral Enhancement Module) + the fused decoder "
                "+ spectral L1 layer + 2 mask tokens are trained; both ViT bodies, the "
                "input projections and the PCA stay frozen. The few-shot feature is the "
                "512-d SEM 'fused' embedding. Tests whether cheaply tuning the fusion "
                "module on top of the frozen backbone beats the raw backbone. Houston "
                "only; an upscale eval and a pad eval were attempted."
            ),
            "feature_dim": 512,
        },
        "pca100_adapted_11x11": {
            "members": [
                "hypersigma_adapt_houston_pca100_joint_sem_run1",
                "hypersigma_houston_pca100_joint_sem_run1",
                "hypersigma_adapt_houston_pca100_spatial_only_run1",
                "hypersigma_houston_pca100_spatial_only_run1",
                "hypersigma_adapt_muufl_pca100_joint_sem_run1",
                "hypersigma_adapt_muufl_pca100_spatial_only_run1",
                "hypersigma_muufl_pca100_spatial_only_run1",
                "hypersigma_adapt_trento_pca100_joint_sem_run1",
                "hypersigma_trento_pca100_joint_sem_run1",
                "hypersigma_adapt_trento_pca100_spatial_only_run1",
                "hypersigma_trento_pca100_spatial_only_run1",
            ],
            "description": (
                "The model is ADAPTED to ingest the 11x11 patch NATIVELY (native_geometry"
                "=false: the 11x11 patch is reflect-padded to 12x12 and cut by a stride-3 "
                "conv into a 4x4 = 16-token grid; pos-embed is interpolated 8x8 -> 4x4) "
                "with the spectral dimension compressed to 100 channels. Two adaptation "
                "modes are compared: 'spatial_only' (cheap, ~1.1M trainable, 768-d spatial "
                "feature) and 'joint_sem' (richer, ~8.4M trainable incl. SEM fusion, 512-d "
                "fused feature). Each (dataset x mode) has a separate adapt run that "
                "produces a checkpoint, then an eval run that does few-shot on it."
            ),
            "feature_dim": {"spatial_only": 768, "joint_sem": 512},
        },
    },
    "input_geometry_modes": {
        "pad_to_64": (
            "11x11 patch centered in a 64x64 zero canvas, fed to the native backbone "
            "(used by native_full_band *_pad and native_sem_pad)."
        ),
        "upscale_to_64": (
            "11x11 patch bicubically resized to 64x64, fed to the native backbone "
            "(used by native_full_band *_upscale and the native_sem upscale run)."
        ),
        "native_11x11_adapted": (
            "11x11 reflect-padded to 12x12 -> stride-3 conv -> 4x4 (16) tokens; the "
            "model is retrained for this small geometry. Used by ALL pca100 runs. "
            "NOTE: the eval_metadata 'input_fit=upscale'/'native_geometry' fields in "
            "the pca100 evals are inert defaults and do NOT apply on this path."
        ),
        "native_backbone_input_size": "64x64 (SpatViT patch-8 -> 8x8 grid; SpecViT 64x64)",
    },
    "spectral_frontend": {
        "spectral_branch_full_bands": "SpecViT ingests the raw HSI bands (144/64/63).",
        "spatial_branch_100ch": (
            "SpatViT's patch-embed conv takes 100 input channels. Houston gets a REAL "
            "PCA 144->100 (pca_houston_100band.pkl, cumulative explained variance ~1.0). "
            "MUUFL (64 bands) and Trento (63 bands) cannot yield 100 PCs, so the pipeline "
            "falls back to a parameter-free SpectralResample (1-D linear interpolation) "
            "64->100 / 63->100. 'PCA-100' is therefore a MISNOMER for MUUFL and Trento."
        ),
    },
    "adaptation_modes": {
        "none_frozen": "native_full_band: no training, frozen backbone, single branch.",
        "sem_only": (
            "native_sem: trains SEM + fused decoder + spectral L1 + 2 mask tokens only "
            "(~8.4M region but most via fusion); both ViT bodies + input proj + PCA frozen."
        ),
        "spatial_only": (
            "pca100 spatial_only: trains SpatViT patch_embed.proj + pos_embed + deformable "
            "sampling_offsets + spat_mask_token + spat_decoder (~1.1M trainable); ViT body "
            "frozen; SpecViT and SEM unused."
        ),
        "joint_sem": (
            "pca100 joint_sem: trains both branch front-ends + mask tokens + per-branch "
            "decoders + SEM + fused decoder (~8.45M trainable); both ViT-B bodies frozen "
            "(~179.55M). Loss = L_spatial + L_spectral + L_fused."
        ),
        "adapt_procedure": (
            "Continued masked-autoencoder (MAE) pretraining, token masking ratio 0.75, "
            "from the HyperSIGMA pretrained checkpoint. native_sem adapt: ~2000 epochs, "
            "batch 64, lr 1.5e-4. pca100 adapt: 2000 epochs, batch 128, lr 1e-5. Seed 42, "
            "RTX 4090 throughout."
        ),
    },
    "few_shot_eval_protocol": {
        "classifier": "Nearest class mean (class mean = mean of L2-normalised support features).",
        "n_way": "Full label set per dataset (Houston 15, MUUFL 11, Trento 6).",
        "k_shot": 5,
        "n_query_per_class": 100,
        "num_episodes": 2000,
        "seed": 42,
        "distance_metrics": ["euclidean", "cosine"],
        "primary_metric": "euclidean (the top-level overall accuracy in results.json equals the euclidean result)",
        "temperature": 10.0,
        "feature_normalization": "L2 (applied for cosine; features mean-pooled from the backbone)",
        "projection_head_at_eval": False,
        "metrics_reported": "Overall Accuracy (OA), Average Accuracy (AA), Kappa, per-class accuracy, per-episode arrays; balanced n_query => OA == AA.",
    },
    "key_observations": [
        "Spatial features >> spectral features for few-shot everywhere; euclidean > cosine consistently.",
        "Houston: PCA-100 11x11-adapted (joint_sem 67.5% / spatial_only 66.4% euclidean OA) beats the "
        "un-adapted native spatial baseline (pad 59.3% / upscale 61.1%) and far beats native_sem pad (45.3%).",
        "Trento: un-adapted native spatial UPSCALE (91.1%) actually edges out the PCA-100 adapted runs "
        "(joint_sem 87.2% / spatial_only 86.0%); adaptation did not help Trento here.",
        "joint_sem (SEM fusion, 512-d) slightly beats spatial_only (768-d) on Houston and Trento; "
        "MUUFL joint_sem was never evaluated (adapt killed at 1050/2000 epochs).",
    ],
    "primary_results_at_a_glance": {
        "metric": "euclidean overall accuracy (%) / cosine in parentheses",
        "native_full_band_spatial": {
            "houston": {"pad": 59.3, "upscale": 61.1},
            "muufl": {"pad": 54.8, "upscale": 53.2},
            "trento": {"pad": 85.0, "upscale": 91.1},
        },
        "native_full_band_spectral": {
            "houston": {"pad": 30.1, "upscale": 30.7},
            "muufl": {"pad": 34.2, "upscale": 30.8},
            "trento": {"pad": 81.1, "upscale": 67.4},
        },
        "native_sem_houston": {
            "pad": {"euclidean": 45.32, "cosine": 38.18},
            "upscale": "CRASHED - no result",
        },
        "pca100_adapted": {
            "houston": {
                "joint_sem": {"euclidean": 67.48, "cosine": 60.67},
                "spatial_only": {"euclidean": 66.37, "cosine": 59.36},
            },
            "muufl": {
                "joint_sem": "NO EVAL (adapt killed @1050/2000)",
                "spatial_only": {"euclidean": 50.25, "cosine": 43.70},
            },
            "trento": {
                "joint_sem": {"euclidean": 87.19, "cosine": 79.27},
                "spatial_only": {"euclidean": 86.02, "cosine": 83.07},
            },
        },
        "note": "Verbatim per-run numbers (with std and 95% CI) are in results_summary_table and in each experiment's raw results.json.",
    },
}

COVERAGE_GAPS_AND_ANOMALIES = [
    "hypersigma_native_sem_run1 (UPSCALE eval) CRASHED: HyperSIGMADual.__init__() got an "
    "unexpected keyword 'spat_resample_to' (code-version mismatch at run time). No results.json; "
    "the native_sem upscale few-shot accuracy was never obtained.",
    "hypersigma_adapt_muufl_pca100_joint_sem_run1: adapt was KILLED at 1050/2000 epochs "
    "(no checkpoint_final.pth, metadata stuck at status='running'); no eval dir exists -> MUUFL "
    "joint_sem PCA-100 few-shot accuracy was never produced.",
    "native_sem coverage is Houston-only. Configs exist for MUUFL and Trento native_sem_pad "
    "(configs/hypersigma/{muufl,trento}_backbonenative_pad_sem_only.yaml) and the driver script "
    "defaults to all three datasets, but no MUUFL/Trento native_sem experiment dirs were completed "
    "(a Trento pad adapt was started to ~epoch 50 then abandoned).",
    "Most eval-container 'pretrain_metadata.json' files are stale stubs (status='running', "
    "overrides={}, one-line stub configs). The real adaptation metadata lives in the sibling "
    "'hypersigma_adapt_*' dirs; the eval results themselves are complete and valid.",
    "Several adapt READMEs name the base config by its pre-rename name 'hypersigma_<ds>_adapt.yaml' "
    "(now configs/hypersigma/<ds>_patchnative_joint_sem.yaml, the 3-band base) and "
    "realise PCA-100 / mode via driver overrides; the dedicated PCA-100 config "
    "(now configs/hypersigma/houston_patchnative_pca100_joint_sem.yaml) "
    "documents a different schedule (3000 epochs / batch 64 / lr 1.5e-4) than what actually ran "
    "(2000 / 128 / 1e-5).",
    "native_ablation: the spatial branch is NOT truly 'full band' on its input width -- it still uses "
    "the 100-channel front-end (Houston PCA 144->100; MUUFL/Trento SpectralResample). Only the "
    "spectral branch feeds raw bands. The top-level pretrain_metadata status is a stale 'running' "
    "while all 12 child evals are 'complete'.",
    "Trento adapt metadata for joint_sem reads status='interrupted, epochs_run=58' -- that describes a "
    "later re-run; the eval loads a checkpoint_final.pth from a fully completed 2000-epoch run "
    "(loaded cleanly, missing=0/unexpected=0), so the Trento joint_sem result is valid.",
]


def load_json(p: Path) -> Any:
    with p.open() as f:
        return json.load(f)


def normalize_acc(v: float | None) -> float | None:
    if v is None:
        return None
    return round(v * 100, 3) if v <= 1.5 else round(v, 3)


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

    raw = load_json(RESULTS / "_report_raw.json")
    analyses = {
        Path(p).stem: load_json(Path(p))
        for p in sorted(glob.glob(str(EXP / "_analysis" / "*.json")))
    }

    # index analysis experiment objects by (aliased) experiment dir name
    analysis_by_exp: dict[str, dict[str, Any]] = {}
    group_ctx_by_exp: dict[str, dict[str, Any]] = {}
    results_table: list[dict[str, Any]] = []

    for grp in analyses.values():
        group_ctx = {k: v for k, v in grp.items() if k not in ("experiments",)}
        for e in grp.get("experiments", []):
            name = NAME_ALIAS.get(e["name"], e["name"])
            analysis_by_exp[name] = e
            group_ctx_by_exp[name] = group_ctx
            dc = e.get("design_choices", {}) or {}
            for r in e.get("results", []):
                row = {
                    "experiment": name,
                    "family": grp.get("family"),
                    "dataset": r.get("dataset") or grp.get("dataset") or dc.get("dataset"),
                    "variant": grp.get("variant") or dc.get("variant") or dc.get("branch"),
                    "branch": r.get("branch") or dc.get("branch"),
                    "input_handling": r.get("input_handling")
                    or dc.get("input_handling")
                    or dc.get("input_geometry"),
                    "eval_name": r.get("eval_name"),
                    "metric": r.get("metric"),
                    "is_primary_metric": r.get("is_primary_metric", r.get("metric") == "euclidean"),
                    "n_way": r.get("n_way"),
                    "k_shot": r.get("k_shot"),
                    "n_query": r.get("n_query"),
                    "num_episodes": r.get("num_episodes"),
                    "accuracy_mean_raw": r.get("accuracy_mean"),
                    "accuracy_mean_pct": normalize_acc(r.get("accuracy_mean")),
                    "accuracy_std": round(r["accuracy_std"], 4)
                    if isinstance(r.get("accuracy_std"), (int, float))
                    else r.get("accuracy_std"),
                    "ci95": round(r["ci95"], 4)
                    if isinstance(r.get("ci95"), (int, float))
                    else r.get("ci95"),
                }
                results_table.append(row)

    results_table.sort(
        key=lambda x: (
            str(x["family"]),
            str(x["dataset"]),
            str(x["variant"]),
            str(x["branch"]),
            str(x["input_handling"]),
            str(x["metric"]),
        )
    )

    # merge per-experiment: analysis + group context + verbatim raw
    experiments: dict[str, Any] = {}
    for name, rawexp in raw["experiments"].items():
        a = analysis_by_exp.get(name)
        experiments[name] = {
            "name": name,
            "path": rawexp.get("path"),
            "family": (a or {}).get("family") or group_ctx_by_exp.get(name, {}).get("family"),
            "group_key": group_ctx_by_exp.get(name, {}).get("group_key"),
            "role": (a or {}).get("role"),
            "analysis": a,
            "group_context": group_ctx_by_exp.get(name),
            "raw": rawexp,
        }

    report = {
        "report_title": OVERVIEW["title"],
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/reports/build_native_pca100_report.py (raw: gather_native_pca100_raw.py; "
        "analysis: 8 parallel audit agents -> experiments/_analysis/*.json)",
        "num_experiments": len(experiments),
        "overview": OVERVIEW,
        "coverage_gaps_and_anomalies": COVERAGE_GAPS_AND_ANOMALIES,
        "results_summary_table": results_table,
        "base_configs": raw.get("base_configs"),
        "experiments": experiments,
    }

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    print(f"Wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  experiments: {len(experiments)}")
    print(f"  result rows: {len(results_table)}")
    no_analysis = [n for n, e in experiments.items() if e["analysis"] is None]
    if no_analysis:
        print(f"  WARNING no analysis matched: {no_analysis}")


if __name__ == "__main__":
    main()

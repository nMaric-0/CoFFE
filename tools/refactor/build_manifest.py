#!/usr/bin/env python3
"""Build ``docs/refactor/manifest.json`` from the phase-1 audit verdicts.

One record per tracked file:
``{path, class, verdict, target, evidence, notes, provisional?, approved}``

* ``class``   -- PAPER | INFRA | EXPLORATORY | DUPLICATE | DEAD | VENDORED
* ``verdict`` -- keep | rename | move | delete | archive
* ``target``  -- new path for rename/move (PAPER_CANON §1, §9), else null
* ``approved``-- set true only after Nikola signs off at the phase-1 gate

Classification evidence comes from ``docs/refactor/closure.json`` (static import
closure, package-``__init__`` edges included) plus the symbol-level usage scan
recorded in AUDIT.md. Rules encoded below are explicit, not inferred, so the
manifest can be re-generated and diffed.

Usage:  python tools/refactor/build_manifest.py [--out docs/refactor/manifest.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Explicit per-file verdicts. Anything not listed falls through to RULES.
# Each entry: path -> (class, verdict, target, evidence, notes)
# ---------------------------------------------------------------------------

V: dict[str, tuple] = {}


def rec(path, cls, verdict, target, evidence, notes="", provisional=False, approved=False):
    V[path] = (cls, verdict, target, evidence, notes, provisional, approved)


# ---- DEAD: closed import islands, reachable only via package re-export -----
_ISLAND = (
    "closed island: reachable only via models/__init__.py:5,12 re-export; "
    "no symbol used outside the island (AUDIT.md §2.2)"
)
for p in (
    "models/backbones/__init__.py",
    "models/backbones/base.py",
    "models/backbones/hsi_baseline.py",
    "models/backbones/mft_channel.py",
    "models/backbones/mft_pixel.py",
):
    rec(p, "DEAD", "delete", None, _ISLAND, "delete with models/__init__.py:5-11 re-export block")
rec("models/wrappers/__init__.py", "DEAD", "delete", None, _ISLAND,
    "delete with models/__init__.py:12 re-export")
rec("models/wrappers/pretrain_wrapper.py", "DEAD", "delete", None, _ISLAND,
    "holds ContrastiveHead / MAEDecoder; no contrastive learning in the paper (D10)")

# ---- DEAD: imported by nobody at all --------------------------------------
rec("data/transforms/__init__.py", "DEAD", "delete", None,
    "closure.json unreached; no in-repo importer of data.transforms (AUDIT.md §2.2)",
    "0% coverage at baseline")
rec("data/transforms/augmentations.py", "DEAD", "delete", None,
    "closure.json unreached; notebook 'Compose' hits are unrelated prose (verified)",
    "sole torchvision consumer; its removal drops that dependency (D6)")
rec("utils/checkpoints.py", "DEAD", "delete", None,
    "no importer; live twin is scripts/evaluate_cosine.py:107 fix_state_dict_keys",
    "RISK: PAPER_CANON §7.2 names this file as the key-map shim location - "
    "see AUDIT.md risk R2 before deleting")

# ---- DEAD: unused shim / stale config -------------------------------------
rec("scripts/adapt_hypersigma_houston.py", "DUPLICATE", "delete", None,
    "shim re-exporting scripts.adapt_hypersigma; no live importer "
    "(lib/adapt_runner.py:86 imports the real module)",
    "only markdown prose + captured log output mention it (D8)")
rec("configs/pretrain/base.yaml", "DEAD", "delete", None,
    "referenced by nothing; utils/io.py:10 load_config is a flat OmegaConf load "
    "with no inheritance, and no config declares a base",
    "D4: also stale (8 heads/4 layers/lambda=2.0/800 ep vs paper 2/2)")

# ---- DUPLICATE: notebook copies ------------------------------------------
for p in (
    "notebooks/pretrain copy.ipynb",
    "notebooks/pretrain copy 2.ipynb",
    "notebooks/pretrain copy 3.ipynb",
    "notebooks/pretrain_run2.ipynb",
    "notebooks/evaluate_hypersigma copy.ipynb",
    "notebooks/evaluate_hypersigma copy 2.ipynb",
    "notebooks/evaluate_hypersigma copy 3.ipynb",
    "notebooks/evaluate_hypersigma_pca100 copy.ipynb",
    "notebooks/evaluate_hypersigma_pca100 copy 3.ipynb",
):
    rec(p, "DUPLICATE", "delete", None,
        "divergent working copy of its base notebook; reaches no module the "
        "PAPER closure does not already reach (closure.json nb_other-only = none)",
        "D8 (count is 9 with pretrain_run2, not 7)")

# ---- EXPLORATORY: ablation / combo / bestcfg (D9) -------------------------
_EXPL = ("reachable only from the ablation/combo/bestcfg roots; appears in no "
         "paper table; pulls in no library code the PAPER closure lacks")
for p in (
    "scripts/run_ablation.py", "scripts/ablation_config.py",
    "scripts/ablation_pretrain_worker.py", "scripts/ablation_eval_worker.py",
    "scripts/aggregate_ablation.py",
    "scripts/run_combo.py", "scripts/combo_config.py",
    "scripts/combo_pretrain_worker.py", "scripts/combo_eval_worker.py",
    "scripts/aggregate_combo.py",
    "scripts/run_bestcfg.py", "scripts/bestcfg_config.py",
    "scripts/bestcfg_pretrain_worker.py", "scripts/bestcfg_eval_worker.py",
    "scripts/aggregate_bestcfg.py",
):
    rec(p, "EXPLORATORY", "archive", "archive/exploratory/" + p.split("/")[-1], _EXPL,
        "D9 APPROVED 2026-08-31: archive into an untracked archive/ tree "
        "('/archive/' added to .gitignore in phase 3, not phase 5)", approved=True)
for p in ("experiments/ablation_report.json", "experiments/combo_report.json",
          "experiments/bestcfg_report.json"):
    rec(p, "EXPLORATORY", "archive", "archive/exploratory/" + p.split("/")[-1],
        "report of an exploratory pipeline; no Table 2/3 cell traces to it",
        "D9 APPROVED 2026-08-31: archive, not delete", approved=True)

# ---- PAPER: report JSONs that provenance Table 2 / Table 3 ---------------
rec("experiments/significance_report.json", "PAPER", "keep", None,
    "supplies the across-seed std for 18 of 30 Table 2 cells (AUDIT.md §3 D1 table)",
    "5-seed, EPOCHS=700, eval_name=sig_eval_epoch700")
rec("experiments/significance_report copy.json", "PAPER", "rename",
    "experiments/significance_report_enhanced_v1.json",
    "NOT junk: supplies the across-seed std for the 12 'enhanced' (HSI+LiDAR) "
    "Table 2 cells, which significance_report.json does not contain "
    "(its groups list omits 'enhanced')",
    "RESCUE from the D8 delete list - see gate item")
rec("experiments/_report_raw.json", "PAPER", "keep", None,
    "holds 20 of 24 Table 3 cells with exact mean+CI match (AUDIT.md §3 D2)",
    "rename proposed but deferred: leading underscore is load-bearing for no reader")
rec("experiments/aggregated_results.json", "PAPER", "keep", None,
    "holds the single-run OA means for 21 of 30 Table 2 cells", "")
rec("experiments/experiment_metadata.json", "PAPER", "keep", None,
    "per-run distilled metadata; second source for Table 2 means", "")
rec("experiments/mft_faithful_results.json", "PAPER", "keep", None,
    "sole source of the 6 MFT-row means in Table 2 (epoch-950 checkpoints)", "")
rec("experiments/hypersigma_native_sem_pca100_report.json", "PAPER", "keep", None,
    "canon-named Table 3 report; its coverage_gaps_and_anomalies list is "
    "corroborated by AUDIT.md §3 D2/D13", "")
rec("experiments/gathered_results.json", "PAPER", "keep", None,
    "output of scripts/gather_requested_results.py (paper_compile root)", "")

# ---- INFRA / docs verdicts ------------------------------------------------
rec("SPLIT.md", "INFRA", "delete", None,
    "documents extraction from the private parent repo mft-cpea, naming an "
    "internal branch (SPLIT.md:3); internal-only",
    "D11 APPROVED 2026-08-31: delete", approved=True)
rec("setup.py", "INFRA", "delete", None,
    "closure.json unreached; placeholder metadata (name 'mft-cpea', "
    "author_email 'nikola@example.com', url 'yourusername'); install_requires "
    "lists einops/timm which nothing imports",
    "fold real packaging into pyproject.toml in phase 6")
rec("requirements.txt", "INFRA", "keep", None,
    "true third-party import set is torch, numpy, scipy, omegaconf, sklearn, "
    "matplotlib, seaborn, tqdm, PyYAML, tensorboard(soft), pytest",
    "D6: prune hydra-core, h5py, scikit-image, spectral, rasterio, timm, einops, "
    "wandb, torchvision; ADD PyYAML (imported by 9 files, currently undeclared)")
rec("README.md", "INFRA", "keep", None,
    "D7 confirmed: wrong title/branding, 'cosine ... prototypical network', "
    "n_way: 5 example, omits MFT and HyperSIGMA routes",
    "rewrite in phase 8 against PAPER_CANON §6")
rec("docs/COSINE_VARIANT.md", "INFRA", "rename", "docs/EVAL_PROTOCOL.md",
    "'Cosine' in a doc name is retired by PAPER_CANON §1; protocol is "
    "Euclidean NCM", "content rewrite in phase 6/8", True)
rec("docs/ENHANCED_PRETRAINING.md", "INFRA", "rename", "docs/PRETRAINING.md",
    "objective 'enhanced' -> 'simmim' per PAPER_CANON §1", "", True)
rec("docs/presentation/RESULTS.md", "INFRA", "keep", None,
    "D2: its HyperSIGMA rows are a superseded generation (joint_sem @600 "
    "episodes, cosine+euclidean) that no Table 3 cell uses; its CoFFE rows "
    "carry within-run CI, not the paper's across-seed std",
    "must be marked superseded, not silently updated")
rec("docs/presentation/RESULTS.json", "INFRA", "keep", None,
    "same generation as RESULTS.md; missing the headline Houston "
    "SimMIM-token HSI+LiDAR row (filtered by compile_results.py:35)", "")
rec("docs/presentation/PROJECT_OVERVIEW.md", "INFRA", "keep", None,
    "internal overview; audit for legacy vocabulary in phase 4", "")
rec("WORKFLOW.md", "INFRA", "keep", None,
    "developer workflow doc; audit for legacy vocabulary in phase 4", "")
rec("PAPER_CANON.md", "INFRA", "keep", None, "the source of truth itself", "")
rec("CLAUDE.md", "INFRA", "keep", None,
    "repo instructions; 'Environment facts' are wrong (phase-0 E1)", "")
rec("LICENSE", "INFRA", "keep", None, "repo license", "")
rec(".gitignore", "INFRA", "keep", None,
    "D5: the bare lib/ rule is already neutralised by the negations at "
    ":78-84 (verified with git check-ignore at phase 0)",
    "cosmetic removal of the shadowed rule in phase 5; phase 3 added the "
    "'/archive/' rule for the D9 archive tree")
rec("pyproject.toml", "INFRA", "keep", None,
    "no gpu/data pytest markers registered (phase-0 E3); addopts forces "
    "--cov on every run", "phase 2 registers the markers")

# ---- Renames mandated by PAPER_CANON §1 ----------------------------------
rec("models/mft_cpea_cosine.py", "PAPER", "rename", "coffe/models/coffe.py",
    "PAPER_CANON §1: module rename; class MFTCPEACosine -> CoFFE",
    "nn.Module ATTRIBUTE names stay frozen (§7.2)", True)
rec("models/mft_original.py", "PAPER", "rename", "coffe/models/mft_original.py",
    "PAPER_CANON §1: class MFTOriginalCosine -> MFTOriginal", "", True)
rec("models/hypersigma/hypersigma_cosine.py", "PAPER", "rename",
    "coffe/models/hypersigma/hypersigma_fewshot.py",
    "PAPER_CANON §1: HyperSIGMACosine -> HyperSIGMAFewShot", "", True)
rec("pretrain/masked_modeling_enhanced.py", "PAPER", "rename",
    "coffe/pretrain/simmim.py",
    "PAPER_CANON §1: EnhancedMaskedSpectralSpatialModel -> SimMIMPretrainModel", "", True)
rec("scripts/evaluate_cosine.py", "PAPER", "rename", "coffe/cli/evaluate.py",
    "PAPER_CANON §1 drops 'Cosine' from names; §9 CLI verb 'evaluate'",
    "also holds the live AA/kappa maths (:277-322) and the live "
    "fix_state_dict_keys (:107); user-facing labels at :789,823 rename per §1", True)
rec("scripts/pretrain_enhanced.py", "PAPER", "rename", "coffe/cli/pretrain.py",
    "PAPER_CANON §1 'enhanced' -> 'simmim'; §9 CLI verb 'pretrain'", "", True)
rec("scripts/adapt_hypersigma.py", "PAPER", "rename", "coffe/cli/adapt_hypersigma.py",
    "PAPER_CANON §9 CLI verb 'adapt-hypersigma'", "", True)
rec("scripts/evaluate_hypersigma_cosine.py", "PAPER", "rename",
    "coffe/cli/evaluate_hypersigma.py", "PAPER_CANON §1 drops 'Cosine'", "", True)
rec("scripts/run_cosine_eval.sh", "PAPER", "rename", "scripts/run_eval.sh",
    "PAPER_CANON §1 drops 'Cosine' from script names", "", True)
rec("scripts/run_trento_cosine_eval.sh", "PAPER", "rename",
    "scripts/run_eval_trento.sh", "PAPER_CANON §1 drops 'Cosine'", "", True)
rec("scripts/rerun_mft_faithful_eval_ep1500.sh", "EXPLORATORY", "delete", None,
    "its epoch-1500 evals are NOT in Table 2 (67.70 vs the paper's 67.49 for "
    "Houston MFT SimMIM token); superseded by the epoch-950 evals",
    "D1 - keeping it would document the wrong recipe")

# ---- D3 RESOLVED: keep lambda, document it honestly ----------------------
# Nikola first asked (2026-08-31) to remove the lambda/CPEA adaptation on the
# grounds that "it does not contribute". tools/refactor/lambda_probe.py measured
# the opposite on the headline Houston checkpoint: -1.70 pp OA and 5.4% of query
# predictions flipped. Presented with that, Nikola chose option 1: KEEP it and
# document it honestly. No numbers move.
rec("models/mft_cpea_cosine.py", "PAPER", "rename", "coffe/models/coffe.py",
    "PAPER_CANON §1: module rename; class MFTCPEACosine -> CoFFE",
    "D3 DECIDED 2026-08-31 (keep + document). Phase 4/6 must: (a) document the "
    "real eval feature z = mean_j(patch_emb_j) + 0.5*cls_emb on "
    "adapt_embeddings (:270-299) and in the eval docs; (b) give lambda_factor "
    "and adapt_embeddings honest names - lambda_factor is a plain float, NOT a "
    "state_dict key (42 keys, none matching), so both are rename-safe; "
    "(c) note the branch at :296-297 is inert at eval (use_projection=False -> "
    "nn.Identity). Class ATTRIBUTE names that ARE state_dict keys stay frozen "
    "(§7.2). forward_episode (:326) is DEAD - a non-executed duplicate of the "
    "live eval path; see R10.", True, approved=True)

# ---------------------------------------------------------------------------
# Fall-through rules, in order. (prefix, class, verdict, evidence)
# ---------------------------------------------------------------------------

RULES = [
    ("third_party/", "VENDORED", "keep",
     "PAPER_CANON §7.4: vendored, contents never modified"),
    ("tests/equivalence/", "INFRA", "keep",
     "phase-2 equivalence harness: the behavior-freeze arbiter (hard rule 1)"),
    ("tests/", "INFRA", "keep", "test suite"),
    ("tools/refactor/", "INFRA", "keep", "refactor tooling (this phase)"),
    ("docs/refactor/", "INFRA", "keep", "refactor record"),
    ("experiments/_example/", "INFRA", "keep",
     "example experiment tree used as a layout reference"),
    ("notebooks/", "PAPER", "keep", "kept notebook (paper route driver)"),
    ("configs/", "PAPER", "keep",
     "config for a paper route; RISK R1: committed configs do not reproduce the "
     "paper runs (see AUDIT.md D13)"),
    ("scripts/", "PAPER", "keep", "in a paper root's import closure"),
    ("lib/", "PAPER", "keep", "in a paper root's import closure"),
    ("models/", "PAPER", "keep", "in a paper root's import closure"),
    ("pretrain/", "PAPER", "keep", "in a paper root's import closure"),
    ("trainers/", "PAPER", "keep",
     "PretrainTrainer used by scripts/pretrain_enhanced.py:45 (D10 refuted)"),
    ("utils/", "PAPER", "keep", "in a paper root's import closure"),
    ("data/", "PAPER", "keep", "in a paper root's import closure"),
    ("experiments/", "PAPER", "keep", "paper-table provenance artifact"),
]


# ---------------------------------------------------------------------------
# Phase-3 gate (2026-08-31). Only 20 of the 43 actionable records were approved
# at the phase-1 gate (the 18 D9 archives + SPLIT.md); the phase-1 log left "a
# review of the remaining 24-entry delete list" open. Nikola approved the other
# 23 deletes in four groups at the start of phase 3. Applied as a sweep so every
# per-record ``evidence`` string above stays exactly as phase 1 wrote it.
# ---------------------------------------------------------------------------

PHASE3_DELETE_TAG = "delete-list APPROVED 2026-08-31 (phase-3 gate, all four groups)"

for _path, _v in list(V.items()):
    if _v[1] == "delete" and not _v[6]:
        _cls, _verdict, _target, _evidence, _notes, _prov, _ = _v
        _notes = f"{_notes} | {PHASE3_DELETE_TAG}" if _notes else PHASE3_DELETE_TAG
        V[_path] = (_cls, _verdict, _target, _evidence, _notes, _prov, True)

NOTE = (
    "verdicts are PROPOSALS until 'approved': true is set at the phase-1 gate"
    " | phase-3 gate 2026-08-31: the remaining 23 delete records approved by"
    " Nikola in four groups (notebook dups, dead code, superseded scripts,"
    " setup.py + utils/checkpoints.py); all 24 deletes and 18 archives executed."
)

# Which phase carried out each already-executed verdict. Such a path is no
# longer tracked, so it cannot come from ``git ls-files`` -- it is emitted from
# ``V`` with ``executed_in_phase`` so the manifest stays a complete ledger of
# every file the audit ruled on, not just the survivors. Phase 3 executed the
# deletes and archives; **a later phase that carries out ``rename``/``move``
# records must add its verdict here**, or ``main()`` exits non-zero rather than
# letting the record drop out of the ledger.
EXECUTED_IN_PHASE = {"delete": 3, "archive": 3}
LEDGER_THROUGH_PHASE = max(EXECUTED_IN_PHASE.values())


def _display(path: Path, root: Path) -> str:
    """Repo-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="docs/refactor/manifest.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    tracked = [
        p for p in subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.split("\x00") if p.strip()
    ]

    tracked_set = set(tracked)
    executed = sorted(
        p for p, v in V.items()
        if p not in tracked_set and v[1] in EXECUTED_IN_PHASE
    )
    missing = sorted(
        p for p in V
        if p not in tracked_set and p not in set(executed)
    )
    if missing:
        print(f"  ERROR: {len(missing)} verdict path(s) neither tracked nor "
              f"accounted for as executed: {missing}")
        print("  Add the executed verdict to EXECUTED_IN_PHASE (or restore the "
              "path) -- the ledger must not silently drop a record.")
        return 1

    records = []
    for path in sorted(set(tracked) | set(executed)):
        if path in V:
            cls, verdict, target, evidence, notes, prov, appr = V[path]
        else:
            for prefix, cls, verdict, evidence in RULES:
                if path.startswith(prefix):
                    target, notes, prov, appr = None, "", False, False
                    break
            else:
                cls, verdict, target, evidence, notes, prov, appr = (
                    "INFRA", "keep", None, "repo-root file, not classified by rule",
                    "", False, False)
        r = {
            "path": path,
            "class": cls,
            "verdict": verdict,
            "target": target,
            "evidence": evidence,
            "notes": notes,
            "approved": appr,
        }
        if prov:
            r["provisional"] = True
        if path not in tracked_set:
            r["executed_in_phase"] = EXECUTED_IN_PHASE[verdict]
        records.append(r)

    counts_class: dict[str, int] = {}
    counts_verdict: dict[str, int] = {}
    for r in records:
        counts_class[r["class"]] = counts_class.get(r["class"], 0) + 1
        counts_verdict[r["verdict"]] = counts_verdict.get(r["verdict"], 0) + 1

    payload = {
        "phase": 1,
        "ledger_through_phase": LEDGER_THROUGH_PHASE,
        "generated_by": "tools/refactor/build_manifest.py",
        "note": NOTE,
        "summary": {
            "files": len(records),
            "by_class": dict(sorted(counts_class.items())),
            "by_verdict": dict(sorted(counts_verdict.items())),
            "executed": len(executed),
            "still_tracked": len(tracked),
        },
        "files": records,
    }

    out = root / args.out
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {_display(out, root)}  ({len(records)} records)")
    print(f"  by class:   {payload['summary']['by_class']}")
    print(f"  by verdict: {payload['summary']['by_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

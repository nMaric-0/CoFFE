#!/usr/bin/env python
"""Regenerate ``golden/*.json`` and the ``fixtures/`` checkpoints.

    python tests/equivalence/make_golden.py            # all goldens
    python tests/equivalence/make_golden.py --only g1  # one group
    python tests/equivalence/make_golden.py --real      # optional local extra

WHY THIS SCRIPT REFUSES TO RUN
------------------------------
The goldens are a record of what the **pre-refactor** code computes. Regenerating
them on refactored code would silently redefine "correct" and defeat the entire
harness. So the script refuses unless:

1. every commit after the ``pre-refactor`` tag is a ``[phase 0]``, ``[phase 1]``
   or ``[phase 2]`` commit, and
2. no *tracked* file outside ``tests/equivalence/``, ``pyproject.toml`` and
   ``docs/`` has uncommitted modifications.

RE-BASELINING AFTER A TORCH UPGRADE
-----------------------------------
A torch/numpy upgrade can legitimately move the last bits of every float and
invalidate the goldens. That is the one case where regeneration is correct, and
it is Nikola's explicit call, not something a phase does on its own:

    git switch --detach pre-refactor
    python tests/equivalence/make_golden.py --rebaseline
    # commit the goldens on a scratch branch, then cherry-pick them onto
    # refactor/cleanup and record the reason in docs/refactor/LOG.md

``--rebaseline`` waives check (1) only — it still refuses on a dirty tree, and
it stamps ``rebaselined_from`` into ``golden/meta.json`` so the provenance of a
re-baselined golden is never in doubt.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.equivalence._harness import (  # noqa: E402
    COFFE_ARCH,
    FIXTURE_DIR,
    FIXTURE_SUFFIX,
    G1_BATCH_SIZE,
    G1_EPOCHS,
    G3_EPISODES,
    GOLDEN_DIR,
    HYPERSIGMA_REGIMES,
    MFT_OBJECTIVES,
    OBJECTIVES,
    SCENES,
    SceneSpec,
    assignment_hash,
    build_all_scenes,
    build_eval_model,
    build_hypersigma_model,
    build_pretrain_model,
    environment_metadata,
    episode_assignments,
    eval_params,
    fixed_input,
    live_eval_feature,
    load_eval_model_from_checkpoint,
    oa_fingerprint,
    pretrain_config,
    set_determinism,
    state_dict_fingerprint,
    tensor_fingerprint,
)

ALLOWED_DIRTY_PREFIXES = ("tests/equivalence/", "pyproject.toml", "docs/")
PHASE_RE = re.compile(r"^\[phase [012]\]")

log = logging.getLogger("make_golden")


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------


def _git(*args: str) -> str:
    return _git_raw(*args).strip()


def _git_raw(*args: str) -> str:
    """Unstripped stdout — `git status --porcelain` encodes state in columns 0-1,
    so the leading space of an unstaged change must survive."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout


def assert_safe_to_generate(rebaseline: bool) -> None:
    modified = [
        line[3:].strip()
        for line in _git_raw("status", "--porcelain").splitlines()
        if line and line[:2] != "??"
    ]
    offenders = [
        path for path in modified
        if not path.startswith(ALLOWED_DIRTY_PREFIXES)
    ]
    if offenders:
        raise SystemExit(
            "REFUSING to regenerate goldens: tracked files outside the phase-2 "
            "scope have uncommitted modifications, so the goldens would not "
            "describe pre-refactor behaviour.\n  "
            + "\n  ".join(offenders)
        )

    if rebaseline:
        log.warning("--rebaseline: skipping the commit-range check (torch upgrade path)")
        return

    subjects = [
        line for line in
        _git("log", "--format=%s", "pre-refactor..HEAD").splitlines() if line
    ]
    bad = [s for s in subjects if not PHASE_RE.match(s)]
    if bad:
        raise SystemExit(
            "REFUSING to regenerate goldens: commits beyond phases 0-2 exist on "
            "this branch, so the code is already refactored.\n  "
            + "\n  ".join(bad)
            + "\n\nTo re-baseline deliberately (e.g. after a torch upgrade), read "
              "the module docstring and pass --rebaseline from the pre-refactor tag."
        )


# ----------------------------------------------------------------------
# G1 — pretrain loss trajectory
# ----------------------------------------------------------------------


def make_g1(scene_root: Path, work: Path) -> Dict[str, Any]:
    from coffe.pretrain.loop import run_pretrain

    spec = SCENES["houston_mini"]
    entries: Dict[str, Any] = {}

    for objective_id in OBJECTIVES:
        for model_name in ("coffe", "mft_original"):
            if model_name == "mft_original" and objective_id not in MFT_OBJECTIVES:
                continue
            key = f"{model_name}:{objective_id}"
            cfg = pretrain_config(
                spec, objective_id, data_root=scene_root, model_name=model_name,
            )
            set_determinism()
            out = work / "g1" / key.replace(":", "_")
            history = run_pretrain(cfg, str(out / "checkpoints"), str(out / "log"))
            entries[key] = {
                # Per-epoch mean training loss, to 8 decimals.
                "train_losses": [round(float(x), 8) for x in history["train_losses"]],
                "epochs_run": int(history["epochs_run"]),
                "final_train_loss": round(float(history["final_train_loss"]), 8),
            }
            log.info("G1 %-38s %s", key, entries[key]["train_losses"])

    return {
        "description": (
            "Per-epoch mean pretraining loss for every objective in PAPER_CANON "
            "Table 2, via coffe.pretrain.loop.run_pretrain on houston_mini."
        ),
        "scene": spec.name,
        "epochs": G1_EPOCHS,
        "batch_size": G1_BATCH_SIZE,
        "entries": entries,
    }


# ----------------------------------------------------------------------
# G2 — encoder forward + init fingerprints
# ----------------------------------------------------------------------


def _g2_entry(model, spec: SceneSpec, use_aux: bool) -> Dict[str, Any]:
    hsi, aux = fixed_input(spec)
    feature = live_eval_feature(model, hsi, aux if use_aux else None)
    return {
        "feature": tensor_fingerprint(feature),
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "state_dict": state_dict_fingerprint(model.state_dict()),
    }


def make_g2(work: Path) -> Dict[str, Any]:
    spec = SCENES["houston_mini"]
    entries: Dict[str, Any] = {}

    for key, model_name, use_aux in (
        ("coffe_hsi_lidar", "coffe", True),
        ("coffe_hsi", "coffe", False),
        ("mft_original", "mft_original", True),
    ):
        set_determinism()
        model = build_eval_model(spec, model_name=model_name, use_aux=use_aux)
        entries[key] = _g2_entry(model, spec, use_aux)
        log.info("G2 %-30s params=%d", key, entries[key]["param_count"])

    for regime in HYPERSIGMA_REGIMES:
        set_determinism()
        model = build_hypersigma_model(spec, regime, work / "hypersigma")
        entries[f"hypersigma_{regime}"] = _g2_entry(model, spec, use_aux=False)
        log.info("G2 hypersigma_%-20s params=%d", regime,
                 entries[f"hypersigma_{regime}"]["param_count"])

    return {
        "description": (
            "Freshly-seeded model: fingerprints of every state_dict tensor (pins "
            "init) and of the LIVE pooled eval feature z (pins forward + the "
            "lambda adaptation of PAPER_CANON D3)."
        ),
        "scene": spec.name,
        "input": {"batch": 4, "patch_size": 11, "generator_seed": 20260231},
        "entries": entries,
    }


# ----------------------------------------------------------------------
# G3 / G4 — episodic eval + the committed checkpoint fixtures
# ----------------------------------------------------------------------

# Trained 3 epochs on houston_mini with the headline objective, then stripped to
# the paper checkpoint's shape: `model_state_dict` with the pretrain model's
# `encoder.`-prefixed keys, exactly what `fix_state_dict_keys` consumes. The
# optimiser/scheduler state is dropped so the fixtures stay small.
FIXTURES = {
    "coffe": {"model_name": "coffe", "objective": "simmim_token"},
    "mft_original": {"model_name": "mft_original", "objective": "simmim_token"},
}


def fixture_path(key: str) -> Path:
    return FIXTURE_DIR / f"{key}_houston_mini_simmim_token_ep3{FIXTURE_SUFFIX}"


def make_fixtures(scene_root: Path, work: Path) -> Dict[str, Any]:
    import torch

    from coffe.pretrain.loop import run_pretrain

    spec = SCENES["houston_mini"]
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Any] = {}

    for key, meta in FIXTURES.items():
        cfg = pretrain_config(
            spec, meta["objective"], data_root=scene_root, model_name=meta["model_name"],
        )
        set_determinism()
        out = work / "fixture" / key
        run_pretrain(cfg, str(out / "checkpoints"), str(out / "log"))
        full = torch.load(out / "checkpoints" / "final.pth", map_location="cpu",
                          weights_only=True)
        slim = {
            "epoch": full["epoch"],
            "global_step": full["global_step"],
            "model_state_dict": full["model_state_dict"],
            "config": full["config"],
            "train_losses": full["train_losses"],
        }
        dest = fixture_path(key)
        torch.save(slim, dest)
        written[key] = {
            "path": str(dest.relative_to(REPO_ROOT)),
            "bytes": dest.stat().st_size,
            "num_keys": len(slim["model_state_dict"]),
            "model_name": meta["model_name"],
            "objective": meta["objective"],
        }
        log.info("fixture %-14s %s (%.2f MB, %d keys)", key, dest.name,
                 written[key]["bytes"] / 1e6, written[key]["num_keys"])

    return written


def _fixture_key_report(spec: SceneSpec, key: str, model_name: str) -> Dict[str, Any]:
    """What `fix_state_dict_keys` makes of a fixture, against the eval model.

    `unexpected` is legitimately non-empty: a pretraining checkpoint carries
    pretrain-only tensors the eval encoder does not have. Recording the exact
    list means any *other* key drifting into it fails the harness.
    """
    from coffe.eval.episodic import (
        fix_state_dict_keys,
        load_checkpoint_with_key_mapping,
    )

    set_determinism()
    model = build_eval_model(spec, model_name=model_name, use_aux=True)
    model_state = model.state_dict()
    state_dict, _ = load_checkpoint_with_key_mapping(str(fixture_path(key)), "cpu")
    fixed = fix_state_dict_keys(state_dict, model_state)
    return {
        "checkpoint_keys": len(state_dict),
        "after_fix_keys": len(fixed),
        "model_keys": len(model_state),
        "unexpected": sorted(set(fixed) - set(model_state)),
        "missing": sorted(set(model_state) - set(fixed)),
    }


def make_g3(scene_root: Path) -> Dict[str, Any]:
    from coffe.eval.episodic import run_evaluation

    spec = SCENES["houston_mini"]
    entries: Dict[str, Any] = {}

    for key, meta in FIXTURES.items():
        params = eval_params(spec, data_root=scene_root, model_name=meta["model_name"])
        set_determinism()
        results = run_evaluation(checkpoint=str(fixture_path(key)), **params)
        assignments = episode_assignments(results)
        entries[key] = {
            "OA": oa_fingerprint(results),
            "n_way": len(assignments[0]["original_classes"]),
            "assignment_hash": assignment_hash(assignments),
            # The freshly-loaded encoder's forward, so a broken loader is caught
            # even if the OA happens to survive.
            "loaded_feature": tensor_fingerprint(
                live_eval_feature(
                    load_eval_model_from_checkpoint(
                        spec, fixture_path(key), model_name=meta["model_name"],
                    ),
                    *fixed_input(spec),
                )
            ),
            "key_report": _fixture_key_report(spec, key, meta["model_name"]),
        }
        log.info("G3 %-14s OA=%s hash=%s", key,
                 entries[key]["OA"]["mean"], entries[key]["assignment_hash"][:16])

    return {
        "description": (
            "Episodic eval via coffe.eval.episodic.run_evaluation against the "
            "committed pre-refactor fixtures: OA to 6 decimals plus a SHA-256 over "
            "the full per-episode per-query argmin assignment matrix."
        ),
        "scene": spec.name,
        "protocol": {
            "n_way": "full-class", "k_shot": 5, "k_query": 10,
            "num_episodes": G3_EPISODES, "distance_metric": "euclidean",
            "use_projection": False, "pool_sigma": None, "seed": 42,
        },
        "entries": entries,
    }


# ----------------------------------------------------------------------
# G5 — masking semantics (PAPER_CANON §3, Eq. 1)
# ----------------------------------------------------------------------


def make_g5(work: Path) -> Dict[str, Any]:
    import torch

    spec = SCENES["houston_mini"]
    entries: Dict[str, Any] = {}
    hsi, aux = fixed_input(spec, batch=8, seed=20260505)

    for objective_id in OBJECTIVES:
        set_determinism()
        model = build_pretrain_model(spec, objective_id)
        model.eval()

        entry: Dict[str, Any] = {}

        # Realized per-module mask rates, on a fixed seed.
        torch.manual_seed(777)
        if hasattr(model, "band_masking"):
            combined = torch.cat([hsi, aux], dim=1)
            _, band_mask = model.band_masking(combined)
            entry["band_mask_ratio_realized"] = round(float(band_mask.mean()), 4)
        if hasattr(model, "spatial_masking"):
            tokens = model.encoder.tokenize(hsi, aux)
            _, token_mask = model.spatial_masking(tokens)
            entry["token_mask_ratio_realized"] = round(float(token_mask.mean()), 4)

        # Union mask + masked loss for one fixed (input, prediction) pair.
        torch.manual_seed(777)
        with torch.no_grad():
            loss, info = model(hsi, aux)
        entry["union_mask_ratio"] = round(float(info["mask"].mean()), 4)
        entry["masked_loss"] = float(loss)
        entry["pred"] = tensor_fingerprint(info["pred"])
        if hasattr(model, "_recon_center_weights"):
            # PAPER_CANON §3: the Gaussian centre weight has mean one, so the
            # loss scale is unchanged by centre weighting.
            entry["recon_center_weight_mean"] = float(model._recon_center_weights.mean())
        entries[objective_id] = entry
        log.info("G5 %-32s %s", objective_id,
                 {k: v for k, v in entry.items() if not isinstance(v, dict)})

    return {
        "description": (
            "Masking semantics: realized band/token mask rates on a fixed seed "
            "(4 decimals, compared exactly), the union-mask rate, and the masked "
            "reconstruction loss for a fixed (input, prediction) pair. Pins Eq. 1: "
            "union of the two masks, centre weights with mean one."
        ),
        "scene": spec.name,
        "input": {"batch": 8, "generator_seed": 20260505, "mask_seed": 777},
        "entries": entries,
    }


# ----------------------------------------------------------------------
# Optional: real-checkpoint fingerprint (machine-specific, gitignored)
# ----------------------------------------------------------------------


def make_real(experiments_dir: Path) -> Dict[str, Any]:
    """50-episode fixed-seed eval of the headline `simmim_token` Houston run.

    Reads ``$COFFE_EXPERIMENTS_DIR``. Machine-specific, so the output goes to
    ``golden/real_local.json`` which is gitignored and never asserted against by
    ``test_equivalence.py``.
    """
    from coffe.eval.episodic import run_evaluation

    # PAPER_CANON §8 D14: Table 2's headline Houston cell (75.30) comes from this
    # directory, evaluated at epoch 950. The `seed52` in the name is a misnomer;
    # the run used seed 42.
    run = experiments_dir / "houston_enhanced_spatial_mask_test_run1_seed52"
    ckpt = run / "checkpoints" / "checkpoint_epoch_950.pth"
    if not ckpt.exists():
        log.warning("--real: %s not found; skipping", ckpt)
        return {}

    set_determinism()
    results = run_evaluation(
        checkpoint=str(ckpt),
        dataset="houston",
        data_root=str(REPO_ROOT / "data" / "raw"),
        split="all",
        n_way=None,
        k_shot=5,
        k_query=100,
        num_episodes=50,
        distance_metric="euclidean",
        use_projection=False,
        pool_sigma=None,
        prototype_mode="mean_features",
        temperature=10.0,
        seed=42,
        device="cpu",
        no_plots=True,
        num_example_episodes=0,
        max_tsne_samples=0,
        output=None,
        patch_size=11,
        name="coffe",
        **{k: v for k, v in COFFE_ARCH.items()},
    )
    return {
        "description": (
            "Local-only sanity fingerprint: 50 fixed-seed episodes on the real "
            "Houston scene with the headline epoch-950 checkpoint. Not asserted "
            "by the harness (machine-specific)."
        ),
        "checkpoint": str(ckpt),
        "num_episodes": 50,
        "k_query": 100,
        "OA_mean": round(float(results["OA"]["mean"]), 4),
        "OA_ci_95": round(float(results["OA"]["ci_95"]), 4),
    }


# ----------------------------------------------------------------------


GROUPS = ("g1", "g2", "g3", "g5")


def _write(name: str, payload: Dict[str, Any]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.json"
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log.info("wrote %s (%.1f KB)", path.relative_to(REPO_ROOT), path.stat().st_size / 1e3)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=GROUPS, action="append", default=None,
                        help="regenerate only these golden groups")
    parser.add_argument("--real", action="store_true",
                        help="also capture the optional local real-checkpoint fingerprint")
    parser.add_argument("--rebaseline", action="store_true",
                        help="waive the commit-range guard (torch upgrade; see docstring)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("scripts").setLevel(logging.WARNING)
    logging.getLogger("trainers").setLevel(logging.WARNING)

    assert_safe_to_generate(args.rebaseline)
    set_determinism()

    groups = tuple(args.only) if args.only else GROUPS
    meta = environment_metadata()
    if args.rebaseline:
        meta["rebaselined_from"] = meta["git_describe"]

    with tempfile.TemporaryDirectory(prefix="coffe-golden-") as tmp:
        work = Path(tmp)
        scene_root = work / "raw"
        build_all_scenes(scene_root)

        if "g1" in groups:
            _write("g1_pretrain_loss", make_g1(scene_root, work))
        if "g2" in groups:
            _write("g2_encoder_forward", make_g2(work))
        if "g3" in groups:
            # G4 is the fixtures themselves: written here, asserted by
            # test_equivalence.py against these G3 numbers.
            fixtures = make_fixtures(scene_root, work)
            payload = make_g3(scene_root)
            payload["fixtures"] = fixtures
            _write("g3_episodic_eval", payload)
        if "g5" in groups:
            _write("g5_masking", make_g5(work))

        if args.real:
            import os

            env = os.environ.get("COFFE_EXPERIMENTS_DIR")
            if not env:
                log.info("--real: COFFE_EXPERIMENTS_DIR unset; skipping silently")
            else:
                payload = make_real(Path(env))
                if payload:
                    _write("real_local", payload)

    # Provenance must survive a partial regeneration (`--only g5`), so
    # per-group records are merged rather than overwritten and the group list is
    # a union of everything ever generated into this golden/ directory.
    existing: Dict[str, Any] = {}
    meta_path = GOLDEN_DIR / "meta.json"
    if meta_path.exists():
        with meta_path.open() as fh:
            existing = json.load(fh)

    per_group: Dict[str, Any] = dict(existing.get("groups", {}))
    stamp = {k: meta[k] for k in ("git_sha", "git_describe", "python", "torch", "numpy")}
    stamp["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for group in groups:
        per_group[group] = dict(stamp)

    existing.update(meta)
    existing["groups"] = per_group
    existing["generated_groups"] = sorted(per_group)
    existing["last_generated_groups"] = list(groups)
    if set(per_group) != set(GROUPS):
        log.warning(
            "golden/ has records for %s but the harness defines %s — regenerate "
            "the missing groups before relying on this directory",
            sorted(per_group), list(GROUPS),
        )

    with meta_path.open("w") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log.info("wrote golden/meta.json (groups recorded: %s)", sorted(per_group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

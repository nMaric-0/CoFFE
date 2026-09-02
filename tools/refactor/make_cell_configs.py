#!/usr/bin/env python3
"""Generate one reproduction config per paper Table-2 cell.

Phase-4 gate decision (Nikola, 2026-09-01): the per-scene base configs are not
enough — every Table 2 cell gets a config carrying **the exact recipe of the run
that produced its published mean** (PAPER_CANON §8 D19/D20).

How a cell's config is built
----------------------------
1. The cell -> run mapping is parsed out of ``docs/refactor/AUDIT.md`` §3, so
   this tool cannot drift from the audit that established it.
2. Every ``model:`` / ``pretrain:`` / ``data:`` / ``hardware:`` value is copied
   **verbatim** from that run's frozen ``experiments/<run>/pretrain_config.yaml``.
   Nothing is invented, rounded or "corrected"; the only rewrites are the two
   vocabulary keys (``model.name``, ``pretrain.objective``) that PAPER_CANON §1
   renames, applied through ``coffe_compat``.
3. ``paths:`` is **not** copied into a *generated* config — each writes to
   ``./checkpoints/<route>/<cell>`` instead of the run's own output directory.
   A config that already existed and already matched its cell (the twelve MAE
   and MFT ones) keeps the ``paths:`` it has: config ``paths:`` values are on
   the audit's DO-NOT-RENAME list. Their header says where they write.
4. The header records the cell, its paper OA, the source run, the **evaluated
   checkpoint epoch** (D17: 950/975, not the final one) and the schedule length.

Requires the private ``experiments/`` tree, so it is a maintainer tool, not part
of the release checks. It refuses to touch a file whose recipe values it would
change: the twelve MAE and MFT cell configs already match their runs exactly,
and only gain a provenance header.

Usage::

    python tools/refactor/make_cell_configs.py             # check, write nothing
    python tools/refactor/make_cell_configs.py --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coffe.compat import normalize_model_name, normalize_objective

AUDIT = REPO_ROOT / "docs" / "refactor" / "AUDIT.md"
EXPERIMENTS = REPO_ROOT / "experiments"

#: Table-2 row label -> (route dir, regime slug). The row labels are the audit's.
REGIMES = {
    "CoFFE SimMIM band": ("coffe", "simmim_band"),
    "CoFFE SimMIM token": ("coffe", "simmim_token"),
    "CoFFE SimMIM band+token": ("coffe", "simmim_band_token"),
    "CoFFE MAE": ("coffe", "mae"),
    "MFT SimMIM token": ("mft", "simmim_token"),
    "MFT MAE": ("mft", "mae"),
}

#: Per-cell notes that the reader needs in order not to be misled.
CELL_NOTES = {
    ("coffe", "simmim_band_token", "houston", "hsi_lidar"): (
        "PAPER_CANON §8 D19: this cell was pretrained at band 0.75, not the 0.85 "
        "§1 states for the band+token regime. The rate below is the one that "
        "produced 64.63."
    ),
    ("coffe", "simmim_band", "trento", "hsi_lidar"): (
        "PAPER_CANON §8 D20: the published mean comes from this recipe (band "
        "0.85, run `trento_enhanced_spectral_run2`), but the ±0.6 in Table 2 is "
        "the across-seed std of 5 runs trained at band **0.75** (cloned from "
        "`trento_enhanced_spectral_run1`). Gate decision 2026-09-01: the cell "
        "config carries the mean's rate."
    ),
    ("coffe", "simmim_band_token", "trento", "hsi_lidar"): (
        "PAPER_CANON §8 D20: as above — mean from band 0.85 "
        "(`trento_enhanced_spectral_spatial_run2`), ±1.2 measured on 5 runs at "
        "band 0.75. The cell config carries the mean's rate."
    ),
    ("coffe", "simmim_token", "houston", "hsi_lidar"): (
        "The source run's directory name says `spatial_mask_test` and `seed52`; "
        "both are misnomers (PAPER_CANON §8 D14 — it ran with seed 42 and is the "
        "headline result). `scripts/compile_results.py` filters it out as a test "
        "run, which is why 75.30 is missing from RESULTS.json."
    ),
    ("coffe", "simmim_band", "houston", "hsi_lidar"): (
        "The odd one out among the Table 2 cells: every other one was evaluated "
        "at epoch 950 or 975 of a 1500- or 3000-epoch schedule."
    ),
}

#: Keys copied verbatim from the frozen run, in the order they are emitted.
MODEL_KEYS = (
    "name",
    "embed_dim",
    "num_heads",
    "num_layers",
    "mlp_dim",
    "attention_type",
    "lambda_factor",
    "dropout",
    "use_aux",
    "use_projection",
    "proj_hidden_dim",
    "proj_num_layers",
    "proj_l2_normalize",
)
DATA_KEYS = ("patch_size", "num_workers")
PRETRAIN_KEYS = (
    "datasets",
    "objective",
    "epochs",
    "batch_size",
    "val_split",
    "lr",
    "min_lr",
    "weight_decay",
    "warmup_epochs",
    "grad_clip",
    "decoder_hidden_dim",
    "band_mask_ratio",
    "spatial_mask_ratio",
    "mask_ratio",
    "decoder_dim",
    "decoder_depth",
    "decoder_heads",
    "norm_pix_loss",
    "recon_center_sigma",
    "pool_center_sigma",
    "save_interval",
    "val_interval",
    "log_interval",
    "use_amp",
)
HARDWARE_KEYS = ("device", "seed", "deterministic")

#: The recipe keys that must never differ between a generated config and the run
#: it reproduces. Compared on every run of this tool.
RECIPE_KEYS = (*(k for k in PRETRAIN_KEYS if k != "objective"), "objective")


class Cell(dict):
    """One Table-2 cell: label, scene, OA, source run, evaluated epoch."""


def parse_audit_cells() -> list[Cell]:
    """Pull the Table-2 mapping out of AUDIT.md §3."""
    cells: list[Cell] = []
    for line in AUDIT.read_text().splitlines():
        if not re.match(r"^\|\s*(\*\*)?(CoFFE|MFT) ", line):
            continue
        col = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(col) < 8:
            continue
        label = col[0].strip("*").strip()
        # "CoFFE SimMIM band HSI+LiDAR" -> regime label + modality
        for regime_label in sorted(REGIMES, key=len, reverse=True):
            if label.startswith(regime_label):
                modality_word = label[len(regime_label) :].strip()
                break
        else:
            raise SystemExit(f"unmapped Table-2 row label: {label!r}")
        route, regime = REGIMES[regime_label]
        oa = re.match(r"[\d.]+", col[2])
        cells.append(
            Cell(
                label=label,
                route=route,
                regime=regime,
                modality="hsi" if modality_word == "HSI" else "hsi_lidar",
                scene=col[1].strip().lower(),
                oa=oa.group(0) if oa else col[2],
                run=col[3].strip("` "),
                eval_epoch=col[5].strip("* "),
                schedule=col[6].strip("* "),
            )
        )
    return cells


def config_path(cell: Cell) -> Path:
    suffix = "_hsi" if cell["modality"] == "hsi" else ""
    return REPO_ROOT / "configs" / cell["route"] / f"{cell['scene']}_{cell['regime']}{suffix}.yaml"


def frozen_config(cell: Cell) -> dict[str, Any] | None:
    path = EXPERIMENTS / cell["run"] / "pretrain_config.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text()) or {}


#: Frozen keys that are informational only and intentionally not emitted.
IGNORED_FROZEN_KEYS = {"data.hsi_channels", "data.aux_channels", "paths", "hardware.device"}


def assert_no_dropped_keys(cell: Cell, frozen: dict[str, Any]) -> None:
    """Refuse to emit if the frozen run carries a key this tool would drop.

    The round-trip guard compares only keys the emitter knows about, so a key
    outside the emit lists would vanish silently. This closes that hole.
    """
    known = (
        {f"model.{k}" for k in MODEL_KEYS}
        | {f"data.{k}" for k in DATA_KEYS}
        | {f"pretrain.{k}" for k in PRETRAIN_KEYS}
        | {f"hardware.{k}" for k in HARDWARE_KEYS}
        | IGNORED_FROZEN_KEYS
    )
    dropped = [
        f"{section}.{key}"
        for section in ("model", "data", "pretrain", "hardware")
        for key in (frozen.get(section) or {})
        if f"{section}.{key}" not in known and section not in IGNORED_FROZEN_KEYS
    ]
    if dropped:
        raise SystemExit(
            f"{cell['run']}: frozen config carries keys this tool would drop "
            f"silently: {sorted(dropped)}. Add them to the emit lists."
        )


def _objective_of(pre: dict[str, Any]) -> str:
    """The objective a frozen `pretrain:` block selects.

    An explicit value always wins. Only when the key is absent does the default
    apply, and that default is the one `scripts/pretrain.py` itself uses —
    SimMIM — never inferred from which other keys happen to be present, so an
    MAE run that omitted the key cannot be silently relabelled.
    """
    if "objective" in pre and pre["objective"] is not None:
        return pre["objective"]
    if (
        pre.get("mask_ratio") is not None
        and not pre.get("band_mask_ratio")
        and not pre.get("spatial_mask_ratio")
    ):
        raise SystemExit(
            "frozen config omits `objective` but looks like MAE (mask_ratio set, "
            "band/spatial unset) — refusing to guess; state it in the run's config."
        )
    return "enhanced"


def recipe(cfg: dict[str, Any]) -> dict[str, Any]:
    """The values that define what a run computes (paths and comments excluded)."""
    pre = cfg.get("pretrain", {}) or {}
    mod = cfg.get("model", {}) or {}
    out = {f"pretrain.{k}": pre.get(k) for k in RECIPE_KEYS}
    out.update({f"model.{k}": mod.get(k) for k in MODEL_KEYS if k != "name"})
    out["model.name"] = normalize_model_name(mod.get("name"))
    out["pretrain.objective"] = normalize_objective(_objective_of(pre))
    out["data.patch_size"] = (cfg.get("data", {}) or {}).get("patch_size")
    return out


def _scalar(v: Any) -> str:
    """One YAML scalar, round-trip safe (no document markers, no float drift)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        # repr keeps the exact value, but PyYAML (YAML 1.1) only reads the
        # exponent form back as a float when the mantissa has a dot: bare
        # `1e-06` parses as the *string* "1e-06" and would reach the scheduler
        # as one. Force `1.0e-06`.
        text = repr(v)
        if "e" in text or "E" in text:
            mantissa, _, exponent = text.partition("e" if "e" in text else "E")
            if "." not in mantissa:
                mantissa += ".0"
            return f"{mantissa}e{exponent}"
        return text if "." in text else f"{text}.0"
    return str(v)


def _yaml_block(mapping: dict[str, Any], keys: tuple[str, ...], indent: str = "  ") -> str:
    lines = []
    for k in keys:
        if k not in mapping:
            continue
        v = mapping[k]
        if v is None and k not in ("recon_center_sigma", "pool_center_sigma"):
            continue
        if isinstance(v, list):
            lines.append(f"{indent}{k}:")
            lines.extend(f"{indent}  - {_scalar(item)}" for item in v)
        else:
            lines.append(f"{indent}{k}: {_scalar(v)}")
    return "\n".join(lines)


def provenance_header(cell: Cell, *, generated: bool) -> str:
    """The comment block that says which paper cell a config reproduces.

    ``generated`` distinguishes a file this tool writes end-to-end (fresh
    ``paths:``) from one that already matched its cell and is only being stamped
    (keeps its own ``paths:``).
    """
    suffix = "_hsi" if cell["modality"] == "hsi" else ""
    stem = f"{cell['scene']}_{cell['regime']}{suffix}"
    modality = "HSI-only" if cell["modality"] == "hsi" else "HSI+LiDAR"
    note = CELL_NOTES.get((cell["route"], cell["regime"], cell["scene"], cell["modality"]))
    lines = (
        [
            f"# {cell['label'].replace(' HSI+LiDAR', '').replace(' HSI', '')}"
            f" — {cell['scene'].capitalize()}, {modality}",
            "#",
            f"# Reproduces the paper's Table 2 cell **OA {cell['oa']}**.",
            "#",
            f"#   source run       experiments/{cell['run']}/",
            f"#   evaluated at     epoch {cell['eval_epoch']} of a {cell['schedule']}-epoch schedule",
            "#",
            "# Every recipe value below matches that run's frozen pretrain_config.yaml,",
            "# checked by tools/refactor/make_cell_configs.py.",
            "#",
        ]
        + (
            [
                "# Output paths are fresh (./checkpoints/<route>/<cell>), not the source",
                "# run's own output directory.",
            ]
            if generated
            else [
                "# NOTE this file keeps its original `paths:` (config paths are on the",
                "# audit's DO-NOT-RENAME list). Those directories do not exist today — the",
                "# paper's checkpoints are under experiments/<run>/checkpoints/ — but a",
                "# plain `python scripts/pretrain.py --config ...` will create and write",
                "# into them. Point `paths:` elsewhere if that matters to you.",
            ]
        )
        + [
            "#",
            "# NOTE the checkpoint the paper evaluated is NOT the final one: train the",
            f"# full {cell['schedule']} epochs, then evaluate epoch {cell['eval_epoch']}",
            "# (PAPER_CANON §8 D17).",
        ]
    )
    if note:
        lines += ["#"] + ["# " + line for line in _wrap(note, 74)]
    lines += [
        "#",
        "# Run with:",
        f"#   python scripts/pretrain.py --config configs/{cell['route']}/{stem}.yaml",
        "",
    ]
    return "\n".join(lines)


def render(cell: Cell, frozen: dict[str, Any]) -> str:
    model = dict(frozen.get("model", {}) or {})
    pre = dict(frozen.get("pretrain", {}) or {})
    data = dict(frozen.get("data", {}) or {})
    hardware = dict(frozen.get("hardware", {}) or {})

    model["name"] = normalize_model_name(model.get("name"))
    pre["objective"] = normalize_objective(_objective_of(pre))
    # The device a particular GPU box used is not part of the recipe.
    hardware["device"] = "cuda"

    suffix = "_hsi" if cell["modality"] == "hsi" else ""
    stem = f"{cell['scene']}_{cell['regime']}{suffix}"
    header = provenance_header(cell, generated=True).rstrip("\n").split("\n")

    body = [
        "",
        "model:",
        _yaml_block(model, MODEL_KEYS),
        "",
        "data:",
        _yaml_block(data, DATA_KEYS),
        "",
        "pretrain:",
        _yaml_block(pre, PRETRAIN_KEYS),
        "",
        "# Fresh output paths: the source run's checkpoint_dir holds the paper's",
        "# checkpoints and must not be written into.",
        "paths:",
        '  data_root: "./data/raw"',
        f'  checkpoint_dir: "./checkpoints/{cell["route"]}/{stem}"',
        f'  log_dir: "./logs/{cell["route"]}/{stem}"',
        "",
        "hardware:",
        _yaml_block(hardware, HARDWARE_KEYS),
        "",
    ]
    return "\n".join(header + body)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--apply", action="store_true", help="write the configs (default: check only)")
    args = ap.parse_args()

    cells = parse_audit_cells()
    print(f"{len(cells)} Table-2 cells parsed from {AUDIT.relative_to(REPO_ROOT)}")

    missing, written, unchanged, conflicts, stamped_paths = [], [], [], [], []
    for cell in cells:
        frozen = frozen_config(cell)
        if frozen is None:
            missing.append(cell["run"])
            continue
        path = config_path(cell)
        assert_no_dropped_keys(cell, frozen)
        want = recipe(frozen)
        if path.exists():
            have = recipe(yaml.safe_load(path.read_text()) or {})
            diff = {k: (have.get(k), want.get(k)) for k in want if have.get(k) != want.get(k)}
            if diff:
                conflicts.append((path.relative_to(REPO_ROOT), diff))
                continue
            unchanged.append(path.relative_to(REPO_ROOT))
            # Same recipe already — give it the same provenance header as the
            # generated ones, without touching a single value.
            if args.apply and "Reproduces the paper's Table 2 cell" not in path.read_text():
                stamped = (
                    provenance_header(cell, generated=False) + "\n" + path.read_text().lstrip("\n")
                )
                after = recipe(yaml.safe_load(stamped) or {})
                drift = {
                    k: (after.get(k), want.get(k)) for k in want if after.get(k) != want.get(k)
                }
                if drift:
                    raise SystemExit(f"header stamp changed values in {path}: {drift}")
                path.write_text(stamped)
                stamped_paths.append(path.relative_to(REPO_ROOT))
            continue
        written.append(path.relative_to(REPO_ROOT))
        text = render(cell, frozen)
        # Guard: the file we are about to write must parse back to exactly the
        # run's recipe. A formatting slip (an exponent YAML reads as a string, a
        # dropped key) would otherwise ship a config that trains something else.
        round_trip = recipe(yaml.safe_load(text) or {})
        drift = {
            k: (round_trip.get(k), want.get(k)) for k in want if round_trip.get(k) != want.get(k)
        }
        if drift:
            raise SystemExit(
                f"render() does not round-trip for {path.relative_to(REPO_ROOT)}: {drift}"
            )
        if args.apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    print(f"  new:       {len(written)}")
    for p in written:
        print(f"    + {p}")
    print(
        f"  already matching their cell: {len(unchanged)}"
        + (f" ({len(stamped_paths)} newly stamped with provenance)" if stamped_paths else "")
    )
    for p in unchanged:
        print(f"    {'s' if p in stamped_paths else '='} {p}")
    if conflicts:
        print(f"  CONFLICT ({len(conflicts)}): an existing config disagrees with its cell's run")
        for p, diff in conflicts:
            print(f"    ! {p}")
            for k, (have, want) in sorted(diff.items()):
                print(f"        {k}: config={have!r} run={want!r}")
    if missing:
        print(f"  source run not present locally ({len(missing)}): {sorted(set(missing))}")
    if not args.apply:
        print("check only — nothing written. Re-run with --apply.")
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())

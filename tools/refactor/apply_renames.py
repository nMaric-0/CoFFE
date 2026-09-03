#!/usr/bin/env python3
"""Phase-4 rename migration: symbols, module paths and file names.

Reads the rename table below (PAPER_CANON §1 + `docs/refactor/manifest.json`'s
`rename` records), prints a dry-run diff, and — only with ``--apply`` — rewrites
the tree and `git mv`s the files.

What it does
------------
* **file renames** (``git mv``, so history follows), plus a textual rewrite of
  every reference to the old path (imports, ``--config`` paths in docs and
  shell drivers, ``python scripts/...`` invocations);
* **symbol renames**, word-boundary anchored (``MFTCPEACosine`` -> ``CoFFE``);
* **module-path renames** (``models.mft_cpea_cosine`` -> ``models.coffe``, and
  the ``from .mft_cpea_cosine import`` relative form).

What it deliberately does NOT do
--------------------------------
* **Config *values*** (``model.name: "mft_cpea"``, ``objective: "enhanced"``,
  variant ids ``"spatial"/"spectral"/"both"``). Those are frozen-artifact
  vocabulary: readers normalise them via ``coffe_compat``, writers were changed
  by hand, and many occurrences are DO-NOT-RENAME literals that name real
  directories on disk (PAPER_CANON §7.3). A blind word-boundary rewrite cannot
  tell the two apart, so it must not try.
* **Prose, docstrings and labels.** "MFT-CPEA", "prototypical network",
  "5-way", "Cosine" in a title — each needs a judgement about what the sentence
  should now claim. Hand-edited.
* **``nn.Module`` attribute names** — state_dict keys, frozen (PAPER_CANON
  §7.2). No rule here touches ``self.<attr>``; the table contains no attribute
  names, and G4 (the checkpoint fixture) is the proof.

Usage::

    python tools/refactor/apply_renames.py            # dry run (default)
    python tools/refactor/apply_renames.py --apply
    python tools/refactor/apply_renames.py --list     # just print the table
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directories never touched: vendored code, frozen artifacts, the refactor's
#: own paper trail, and everything git does not track anyway.
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "third_party",
    "experiments",
    "results",
    "archive",
    "data",
    "checkpoints",
    "logs",
    "outputs",
    ".pytest_cache",
    "__pycache__",
}

#: Files whose *content* records the legacy vocabulary on purpose.
EXCLUDED_FILES = {
    "PAPER_CANON.md",
    "CHANGES.md",
    "coffe/compat.py",
    "docs/refactor/AUDIT.md",
    "docs/refactor/BASELINE.md",
    "docs/refactor/ENV.md",
    "docs/refactor/LOG.md",
    "docs/refactor/manifest.json",
    "docs/refactor/inventory.json",
    "docs/refactor/closure.json",
    "docs/refactor/packages.txt",
    "docs/refactor/baseline_pytest.txt",
    "tools/refactor/apply_renames.py",
    "tools/refactor/build_manifest.py",
    "tools/refactor/inventory.py",
    "tools/refactor/closure.py",
    "tools/refactor/lambda_probe.py",
}

TEXT_SUFFIXES = {".py", ".ipynb", ".md", ".sh", ".yaml", ".yml", ".txt", ".json", ".cfg", ".toml"}


# ----------------------------------------------------------------------
# The rename table
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FileRename:
    """One ``git mv``. ``refs`` are extra textual spellings of the old path."""

    old: str
    new: str
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SymbolRename:
    """One word-boundary-anchored identifier rewrite."""

    old: str
    new: str
    why: str = ""


#: Files. Phase 4 changes basenames only — the move into a ``coffe/`` package
#: is phase 5 (the manifest's `rename` targets are stated post-restructure).
FILE_RENAMES: tuple[FileRename, ...] = (
    FileRename("models/mft_cpea_cosine.py", "models/coffe.py"),
    FileRename("models/hypersigma/hypersigma_cosine.py", "models/hypersigma/few_shot.py"),
    FileRename("pretrain/masked_modeling_enhanced.py", "pretrain/simmim.py"),
    FileRename("scripts/evaluate_cosine.py", "scripts/evaluate.py"),
    FileRename("scripts/pretrain_enhanced.py", "scripts/pretrain.py"),
    FileRename("scripts/evaluate_hypersigma_cosine.py", "scripts/evaluate_hypersigma.py"),
    FileRename("scripts/run_cosine_eval.sh", "scripts/run_eval.sh"),
    FileRename("scripts/run_trento_cosine_eval.sh", "scripts/run_eval_trento.sh"),
    FileRename("tests/test_pretrain_enhanced.py", "tests/test_pretrain_simmim.py"),
    FileRename("docs/ENHANCED_PRETRAINING.md", "docs/PRETRAINING.md"),
    FileRename("docs/COSINE_VARIANT.md", "docs/EVAL_PROTOCOL.md"),
    # Configs: canonical names per PAPER_CANON §9,
    # `configs/{coffe,mft,hypersigma}/<scene>_<regime>[_hsi].yaml`. The regime in
    # each new name is the one the file's own mask rates encode — no rate is
    # changed by this rename (see LOG.md, phase 4: the three `*_pretrain_enhanced`
    # base configs do not all carry the same regime).
    FileRename(
        "configs/pretrain/houston_pretrain_enhanced.yaml", "configs/coffe/houston_simmim.yaml"
    ),
    FileRename(
        "configs/pretrain/trento_pretrain_enhanced.yaml", "configs/coffe/trento_simmim.yaml"
    ),
    FileRename("configs/pretrain/muufl_pretrain_enhanced.yaml", "configs/coffe/muufl_simmim.yaml"),
    FileRename(
        "configs/pretrain/houston_pretrain_hsi_only.yaml", "configs/coffe/houston_simmim_hsi.yaml"
    ),
    FileRename(
        "configs/pretrain/trento_pretrain_hsi_only.yaml", "configs/coffe/trento_simmim_hsi.yaml"
    ),
    FileRename(
        "configs/pretrain/muufl_pretrain_hsi_only.yaml", "configs/coffe/muufl_simmim_hsi.yaml"
    ),
    FileRename("configs/pretrain/houston_pretrain_mae.yaml", "configs/coffe/houston_mae.yaml"),
    FileRename("configs/pretrain/trento_pretrain_mae.yaml", "configs/coffe/trento_mae.yaml"),
    FileRename("configs/pretrain/muufl_pretrain_mae.yaml", "configs/coffe/muufl_mae.yaml"),
    FileRename(
        "configs/pretrain/houston_pretrain_mae_hsi_only.yaml", "configs/coffe/houston_mae_hsi.yaml"
    ),
    FileRename(
        "configs/pretrain/trento_pretrain_mae_hsi_only.yaml", "configs/coffe/trento_mae_hsi.yaml"
    ),
    FileRename(
        "configs/pretrain/muufl_pretrain_mae_hsi_only.yaml", "configs/coffe/muufl_mae_hsi.yaml"
    ),
    FileRename(
        "configs/pretrain/mft_original_houston_spatial.yaml",
        "configs/mft/houston_simmim_token.yaml",
    ),
    FileRename(
        "configs/pretrain/mft_original_trento_spatial.yaml", "configs/mft/trento_simmim_token.yaml"
    ),
    FileRename(
        "configs/pretrain/mft_original_muufl_spatial.yaml", "configs/mft/muufl_simmim_token.yaml"
    ),
    FileRename("configs/pretrain/mft_original_houston_mae.yaml", "configs/mft/houston_mae.yaml"),
    FileRename("configs/pretrain/mft_original_trento_mae.yaml", "configs/mft/trento_mae.yaml"),
    FileRename("configs/pretrain/mft_original_muufl_mae.yaml", "configs/mft/muufl_mae.yaml"),
    FileRename(
        "configs/pretrain/hypersigma_houston_adapt.yaml",
        "configs/hypersigma/houston_patchnative_joint_sem.yaml",
    ),
    FileRename(
        "configs/pretrain/hypersigma_trento_adapt.yaml",
        "configs/hypersigma/trento_patchnative_joint_sem.yaml",
    ),
    FileRename(
        "configs/pretrain/hypersigma_muufl_adapt.yaml",
        "configs/hypersigma/muufl_patchnative_joint_sem.yaml",
    ),
    FileRename(
        "configs/pretrain/hypersigma_houston_adapt_pca100.yaml",
        "configs/hypersigma/houston_patchnative_pca100_joint_sem.yaml",
    ),
    FileRename(
        "configs/pretrain/hypersigma_houston_adapt_native_sem.yaml",
        "configs/hypersigma/houston_backbonenative_upscale_sem_only.yaml",
    ),
    FileRename(
        "configs/pretrain/hypersigma_houston_adapt_native_sem_pad.yaml",
        "configs/hypersigma/houston_backbonenative_pad_sem_only.yaml",
    ),
    FileRename(
        "configs/pretrain/hypersigma_trento_adapt_native_sem_pad.yaml",
        "configs/hypersigma/trento_backbonenative_pad_sem_only.yaml",
    ),
    FileRename(
        "configs/pretrain/hypersigma_muufl_adapt_native_sem_pad.yaml",
        "configs/hypersigma/muufl_backbonenative_pad_sem_only.yaml",
    ),
)

#: Path *templates* — the same renames as above, in the f-string / shell-variable
#: spellings used by the experiment drivers (``configs/pretrain/{ds}_pretrain_enhanced.yaml``).
#: They name no file on disk, so they cannot be FileRename entries, but every
#: value they can expand to is one.
PATH_TEMPLATES: tuple[SymbolRename, ...] = (
    SymbolRename("configs/pretrain/{ds}_pretrain_enhanced.yaml", "configs/coffe/{ds}_simmim.yaml"),
    SymbolRename(
        "configs/pretrain/{ds}_pretrain_hsi_only.yaml", "configs/coffe/{ds}_simmim_hsi.yaml"
    ),
    SymbolRename("configs/pretrain/{ds}_pretrain_mae.yaml", "configs/coffe/{ds}_mae.yaml"),
    SymbolRename(
        "configs/pretrain/{ds}_pretrain_mae_hsi_only.yaml", "configs/coffe/{ds}_mae_hsi.yaml"
    ),
    SymbolRename(
        "configs/pretrain/mft_original_{ds}_spatial.yaml", "configs/mft/{ds}_simmim_token.yaml"
    ),
    SymbolRename("configs/pretrain/mft_original_{ds}_mae.yaml", "configs/mft/{ds}_mae.yaml"),
    SymbolRename(
        "configs/pretrain/hypersigma_{dataset}_adapt.yaml",
        "configs/hypersigma/{dataset}_patchnative_joint_sem.yaml",
    ),
    SymbolRename(
        "configs/pretrain/hypersigma_${ds}_adapt_native_sem_pad.yaml",
        "configs/hypersigma/${ds}_backbonenative_pad_sem_only.yaml",
    ),
)

#: Classes and other importable symbols (PAPER_CANON §1). Word-boundary anchored.
SYMBOL_RENAMES: tuple[SymbolRename, ...] = (
    SymbolRename("MFTCPEACosine", "CoFFE", "canon §1: the paper's compact encoder"),
    SymbolRename("MFTOriginalCosine", "MFTOriginal", "canon §1: the architectural control"),
    SymbolRename("HyperSIGMACosine", "HyperSIGMAFewShot", "canon §1"),
    SymbolRename(
        "EnhancedMaskedSpectralSpatialModel",
        "SimMIMPretrainModel",
        "canon §1: objective 'enhanced' -> 'simmim'",
    ),
)

#: Dotted and relative module paths implied by FILE_RENAMES.
MODULE_RENAMES: tuple[SymbolRename, ...] = (
    SymbolRename("models.mft_cpea_cosine", "models.coffe"),
    SymbolRename("models.hypersigma.hypersigma_cosine", "models.hypersigma.few_shot"),
    SymbolRename("pretrain.masked_modeling_enhanced", "pretrain.simmim"),
    SymbolRename("scripts.evaluate_cosine", "scripts.evaluate"),
    SymbolRename("scripts.pretrain_enhanced", "scripts.pretrain"),
    SymbolRename("scripts.evaluate_hypersigma_cosine", "scripts.evaluate_hypersigma"),
    # relative imports inside the packages
    SymbolRename("from .mft_cpea_cosine", "from .coffe"),
    SymbolRename("from .hypersigma_cosine", "from .few_shot"),
    SymbolRename("from .masked_modeling_enhanced", "from .simmim"),
)


# ----------------------------------------------------------------------
# Rewriting
# ----------------------------------------------------------------------


def _is_excluded(rel: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return True
    return rel.as_posix() in EXCLUDED_FILES


def tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    files = []
    for line in out:
        rel = Path(line)
        if _is_excluded(rel) or rel.suffix not in TEXT_SUFFIXES:
            continue
        files.append(rel)
    return files


def build_rules() -> list[tuple[re.Pattern, str, str]]:
    """(compiled pattern, replacement, human label), most specific first."""
    rules: list[tuple[re.Pattern, str, str]] = []
    # Paths first: `scripts/evaluate_cosine.py` must not be half-rewritten by
    # the `scripts.evaluate_cosine` module rule.
    for tpl in PATH_TEMPLATES:
        rules.append((re.compile(re.escape(tpl.old)), tpl.new, f"template {tpl.old} -> {tpl.new}"))
    for fr in FILE_RENAMES:
        for spelling in (fr.old, *fr.refs):
            rules.append((re.compile(re.escape(spelling)), fr.new, f"path {spelling} -> {fr.new}"))
    for mr in MODULE_RENAMES:
        rules.append((re.compile(re.escape(mr.old)), mr.new, f"module {mr.old} -> {mr.new}"))
    for sr in SYMBOL_RENAMES:
        rules.append(
            (re.compile(rf"\b{re.escape(sr.old)}\b"), sr.new, f"symbol {sr.old} -> {sr.new}")
        )
    return rules


@dataclass
class Change:
    path: Path
    before: str
    after: str
    hits: dict[str, int] = field(default_factory=dict)


def rewrite_text(
    text: str, rules: Sequence[tuple[re.Pattern, str, str]]
) -> tuple[str, dict[str, int]]:
    hits: dict[str, int] = {}
    for pattern, replacement, label in rules:
        text, n = pattern.subn(replacement, text)
        if n:
            hits[label] = hits.get(label, 0) + n
    return text, hits


def collect_changes(rules: Sequence[tuple[re.Pattern, str, str]]) -> list[Change]:
    changes = []
    for rel in tracked_text_files():
        path = REPO_ROOT / rel
        before = path.read_text(encoding="utf-8")
        after, hits = rewrite_text(before, rules)
        if after != before:
            changes.append(Change(rel, before, after, hits))
    return changes


def print_diff(changes: Sequence[Change]) -> None:
    for change in changes:
        diff = difflib.unified_diff(
            change.before.splitlines(keepends=True),
            change.after.splitlines(keepends=True),
            fromfile=f"a/{change.path}",
            tofile=f"b/{change.path}",
            n=1,
        )
        sys.stdout.writelines(diff)


def print_summary(changes: Sequence[Change]) -> None:
    total = 0
    per_rule: dict[str, int] = {}
    for change in changes:
        for label, n in change.hits.items():
            per_rule[label] = per_rule.get(label, 0) + n
            total += n
    print(f"\n{len(changes)} files, {total} substitutions")
    for label in sorted(per_rule):
        print(f"  {per_rule[label]:4d}  {label}")


def do_moves(apply: bool) -> None:
    for fr in FILE_RENAMES:
        src, dst = REPO_ROOT / fr.old, REPO_ROOT / fr.new
        if not src.exists():
            if dst.exists():
                print(f"  (already moved) {fr.old} -> {fr.new}")
                continue
            raise SystemExit(f"missing source for rename: {fr.old}")
        print(f"  git mv {fr.old} -> {fr.new}")
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-C", str(REPO_ROOT), "mv", fr.old, fr.new], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    ap.add_argument("--list", action="store_true", help="print the rename table and exit")
    ap.add_argument("--quiet", action="store_true", help="summary only, no diff")
    args = ap.parse_args()

    if args.list:
        print("FILE RENAMES")
        for fr in FILE_RENAMES:
            print(f"  {fr.old}\n    -> {fr.new}")
        print("\nSYMBOL RENAMES")
        for sr in SYMBOL_RENAMES:
            print(f"  {sr.old} -> {sr.new}" + (f"   ({sr.why})" if sr.why else ""))
        print("\nPATH TEMPLATES")
        for tpl in PATH_TEMPLATES:
            print(f"  {tpl.old} -> {tpl.new}")
        print("\nMODULE RENAMES")
        for mr in MODULE_RENAMES:
            print(f"  {mr.old} -> {mr.new}")
        return 0

    rules = build_rules()
    changes = collect_changes(rules)

    if not args.quiet:
        print("TEXT CHANGES")
        print_diff(changes)
    print_summary(changes)

    # Content is rewritten at the *old* paths first, then the files move; doing
    # it the other way round would recreate the files git just moved away.
    if args.apply:
        for change in changes:
            (REPO_ROOT / change.path).write_text(change.after, encoding="utf-8")

    print("\nMOVES")
    do_moves(args.apply)

    print("applied." if args.apply else "dry run — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

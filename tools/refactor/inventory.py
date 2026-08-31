#!/usr/bin/env python3
"""Snapshot the repository's Python surface for the phase-0 refactor baseline.

Stdlib only (``ast`` for parsing). Walks the repo, skipping vendored code,
caches, and per-run experiment trees, and writes ``docs/refactor/inventory.json``:

* ``python``    -- one record per ``.py`` file: LOC, top-level defs/classes,
                   and imports (each resolved to an in-repo module path when
                   the import names something that exists in this repo).
* ``other``     -- ``.yaml``/``.yml``/``.sh``/``.ipynb``/``.md``/``.json`` files
                   with their sizes in bytes.
* ``summary``   -- counts and totals.

Usage:  python tools/refactor/inventory.py [--root .] [--out docs/refactor/inventory.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

# Directories never descended into.
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "third_party",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "node_modules",
    ".egg-info",
}

# Per-run experiment output trees are skipped; loose files in experiments/ are kept.
SKIP_GLOB_PARENTS = ("experiments",)

OTHER_SUFFIXES = {".yaml", ".yml", ".sh", ".ipynb", ".md", ".json"}


def should_skip_dir(path: Path, root: Path) -> bool:
    name = path.name
    if name in SKIP_DIRS or name.endswith(".egg-info"):
        return True
    rel = path.relative_to(root)
    # experiments/<run>/ is skipped; experiments/ itself is not.
    if len(rel.parts) >= 2 and rel.parts[0] in SKIP_GLOB_PARENTS:
        return True
    return False


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not should_skip_dir(here / d, root))
        for fn in sorted(filenames):
            yield here / fn


def module_index(root: Path, py_files: list[Path]) -> dict[str, str]:
    """Map dotted in-repo module names -> repo-relative file paths."""
    index: dict[str, str] = {}
    for path in py_files:
        rel = path.relative_to(root)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
            if not parts:
                continue
        else:
            parts[-1] = parts[-1][: -len(".py")]
        index[".".join(parts)] = rel.as_posix()
    return index


def resolve(name: str, index: dict[str, str]) -> str | None:
    """Longest-prefix match of a dotted import against in-repo modules."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        hit = index.get(".".join(parts[:i]))
        if hit:
            return hit
    return None


def absolutise(node: ast.ImportFrom, rel: Path) -> str:
    """Turn a relative ``from . import x`` into a dotted absolute module name."""
    pkg = list(rel.parts[:-1])
    up = node.level - 1
    if up:
        pkg = pkg[:-up] if up <= len(pkg) else []
    tail = node.module.split(".") if node.module else []
    return ".".join(pkg + tail)


def scan_python(path: Path, root: Path, index: dict[str, str]) -> dict:
    rel = path.relative_to(root)
    text = path.read_text(encoding="utf-8", errors="replace")
    record: dict = {
        "path": rel.as_posix(),
        "loc": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
        "bytes": path.stat().st_size,
        "functions": [],
        "classes": [],
        "imports": [],
        "parse_error": None,
    }
    try:
        tree = ast.parse(text, filename=str(rel))
    except SyntaxError as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
        return record

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            record["functions"].append({"name": node.name, "line": node.lineno})
        elif isinstance(node, ast.ClassDef):
            record["classes"].append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [
                        b.name
                        for b in node.body
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                }
            )

    seen: set[tuple] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                entry = (alias.name, node.lineno)
                if entry in seen:
                    continue
                seen.add(entry)
                record["imports"].append(
                    {
                        "module": alias.name,
                        "line": node.lineno,
                        "relative": False,
                        "resolved": resolve(alias.name, index),
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            mod = absolutise(node, rel) if node.level else (node.module or "")
            entry = (mod, node.lineno, node.level)
            if entry in seen:
                continue
            seen.add(entry)
            record["imports"].append(
                {
                    "module": mod,
                    "line": node.lineno,
                    "relative": bool(node.level),
                    "names": [a.name for a in node.names],
                    "resolved": resolve(mod, index) if mod else None,
                }
            )
    record["imports"].sort(key=lambda i: (i["line"], i["module"]))
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repository root (default: cwd)")
    ap.add_argument("--out", default="docs/refactor/inventory.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = list(iter_files(root))
    py_files = [f for f in files if f.suffix == ".py"]
    index = module_index(root, py_files)

    python = [scan_python(f, root, index) for f in py_files]
    other = [
        {
            "path": f.relative_to(root).as_posix(),
            "suffix": f.suffix,
            "bytes": f.stat().st_size,
        }
        for f in files
        if f.suffix in OTHER_SUFFIXES
    ]

    by_suffix: dict[str, int] = {}
    for entry in other:
        by_suffix[entry["suffix"]] = by_suffix.get(entry["suffix"], 0) + 1

    payload = {
        "root": root.name,
        "skipped_dirs": sorted(SKIP_DIRS),
        "skipped_subtrees": [f"{p}/*/" for p in SKIP_GLOB_PARENTS],
        "summary": {
            "python_files": len(python),
            "python_loc": sum(r["loc"] for r in python),
            "top_level_functions": sum(len(r["functions"]) for r in python),
            "top_level_classes": sum(len(r["classes"]) for r in python),
            "parse_errors": [r["path"] for r in python if r["parse_error"]],
            "other_files": len(other),
            "other_by_suffix": dict(sorted(by_suffix.items())),
        },
        "python": python,
        "other": other,
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    s = payload["summary"]
    print(f"wrote {out.relative_to(root)}")
    print(f"  python: {s['python_files']} files, {s['python_loc']} LOC")
    print(f"  top-level: {s['top_level_functions']} defs, {s['top_level_classes']} classes")
    print(f"  other: {s['other_files']} files {s['other_by_suffix']}")
    if s["parse_errors"]:
        print(f"  PARSE ERRORS: {s['parse_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

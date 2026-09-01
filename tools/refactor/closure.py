#!/usr/bin/env python3
"""Static import-closure tracer for the phase-1 refactor audit.

Reuses ``tools/refactor/inventory.py``'s module index and import scanner, then
computes, for each named *root set*, the transitive closure of in-repo Python
modules reachable by ``import`` / ``from ... import``. Notebooks (``.ipynb``)
are parsed for imports in their code cells; shell scripts are scanned for the
Python entry points they invoke.

The closure is static only. Dynamic dispatch (a config string routed to a class
in a lookup table) is *not* followed -- those edges are supplied by hand via
``EXTRA_EDGES`` below, each with a file:line justification.

``ROOTS`` is the phase-1 audit's root list and is kept intact as provenance even
after a phase prunes some of those entry points; roots that no longer exist are
skipped and reported under ``pruned_roots``. ``docs/refactor/closure.json`` is
the frozen phase-1 snapshot that ``AUDIT.md`` cites -- re-run against the
current tree with an explicit ``--out`` rather than overwriting it.

Usage:  python tools/refactor/closure.py [--root .] [--out docs/refactor/closure.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventory import iter_files, module_index, resolve, absolutise  # noqa: E402

# ---------------------------------------------------------------------------
# Root sets. Each maps a name -> list of repo-relative entry-point paths.
# Grouped so the audit can ask "what does *only* the exploratory tree reach?"
# ---------------------------------------------------------------------------

ROOTS: dict[str, list[str]] = {
    # ---- Table 2, CoFFE + MFT rows: the significance pipeline -------------
    "paper_significance": [
        "scripts/run_significance_experiment.py",
        "scripts/sig_pretrain_worker.py",
        "scripts/sig_eval_worker.py",
        "scripts/sig_significance_config.py",
        "scripts/aggregate_significance.py",
    ],
    # ---- Table 2, per-group runners --------------------------------------
    "paper_group_runners": [
        "scripts/run_mae_experiments.py",
        "scripts/run_hsi_only_experiments.py",
        "scripts/run_mft_original_mae_experiments.py",
        "scripts/run_mft_original_spatial_experiments.py",
    ],
    # ---- The two core CLIs the runners shell out to -----------------------
    "paper_core_cli": [
        "scripts/pretrain_enhanced.py",
        "scripts/evaluate_cosine.py",
    ],
    # ---- Tables 2-3, HyperSIGMA rows -------------------------------------
    "paper_hypersigma": [
        "scripts/adapt_hypersigma.py",
        "scripts/evaluate_hypersigma_cosine.py",
        "scripts/fit_pca_hypersigma.py",
        "scripts/run_hypersigma_spatial_pca100.py",
        "scripts/gather_native_pca100_raw.py",
        "scripts/build_native_pca100_report.py",
    ],
    # ---- Result compilation ----------------------------------------------
    "paper_compile": [
        "scripts/compile_results.py",
        "scripts/compile_mft_faithful_results.py",
        "scripts/gather_requested_results.py",
        "scripts/aggregate_experiment_results.py",
        "scripts/build_experiment_metadata.py",
    ],
    # ---- Exploratory: ablation / combo / bestcfg (D9) ---------------------
    "exploratory": [
        "scripts/run_ablation.py",
        "scripts/ablation_config.py",
        "scripts/ablation_pretrain_worker.py",
        "scripts/ablation_eval_worker.py",
        "scripts/aggregate_ablation.py",
        "scripts/run_combo.py",
        "scripts/combo_config.py",
        "scripts/combo_pretrain_worker.py",
        "scripts/combo_eval_worker.py",
        "scripts/aggregate_combo.py",
        "scripts/run_bestcfg.py",
        "scripts/bestcfg_config.py",
        "scripts/bestcfg_pretrain_worker.py",
        "scripts/bestcfg_eval_worker.py",
        "scripts/aggregate_bestcfg.py",
    ],
    # ---- Duplicate / superseded twins (D8) -------------------------------
    "duplicates": [
        "scripts/adapt_hypersigma_houston.py",
    ],
    # ---- Notebooks kept by default (canon gate line) ---------------------
    "notebooks_keep": [
        "notebooks/pretrain.ipynb",
        "notebooks/evaluate.ipynb",
        "notebooks/compare.ipynb",
        "notebooks/adapt_hypersigma_native_sem.ipynb",
    ],
    "notebooks_other": [
        "notebooks/evaluate_hypersigma.ipynb",
        "notebooks/evaluate_hypersigma_native.ipynb",
        "notebooks/evaluate_hypersigma_pca100.ipynb",
        "notebooks/pretrain_run2.ipynb",
        "notebooks/evaluate_hypersigma copy.ipynb",
        "notebooks/evaluate_hypersigma copy 2.ipynb",
        "notebooks/evaluate_hypersigma copy 3.ipynb",
        "notebooks/evaluate_hypersigma_pca100 copy.ipynb",
        "notebooks/evaluate_hypersigma_pca100 copy 3.ipynb",
        "notebooks/pretrain copy.ipynb",
        "notebooks/pretrain copy 2.ipynb",
        "notebooks/pretrain copy 3.ipynb",
    ],
    # ---- Tests -----------------------------------------------------------
    "tests": [
        "tests/test_data.py",
        "tests/test_models.py",
        "tests/test_pretrain_enhanced.py",
        "tests/test_spatial_weights.py",
        "tests/test_mft_original_shapes.py",
        "tests/test_hypersigma_shapes.py",
        "tests/test_hypersigma_native_shapes.py",
        # Phase-2 equivalence harness (pytest collects these directly).
        "tests/equivalence/test_equivalence.py",
        "tests/equivalence/conftest.py",
        "tests/equivalence/make_golden.py",
    ],
    # ---- Refactor tooling ------------------------------------------------
    "tooling": [
        "tools/refactor/inventory.py",
        "tools/refactor/closure.py",
        "tools/refactor/build_manifest.py",
        "tools/refactor/lambda_probe.py",
    ],
}

# Root-set groups that count as PAPER provenance.
PAPER_ROOTS = [k for k in ROOTS if k.startswith("paper_")]

# ---------------------------------------------------------------------------
# Dynamic edges the static tracer cannot see. Each needs a justification.
# Populated after grepping for string dispatch; see AUDIT.md.
# ---------------------------------------------------------------------------

EXTRA_EDGES: list[tuple[str, str, str]] = [
    # (from_path, to_path, evidence)
]


def notebook_source(path: Path) -> str:
    """Concatenate a notebook's code cells into one Python-ish source string."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return ""
    out: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        # Drop IPython magics / shell escapes, which are not valid Python.
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!", "?")):
                continue
            out.append(line)
        out.append("")
    return "\n".join(out)


def imports_of(path: Path, root: Path, index: dict[str, str]) -> list[dict]:
    """In-repo imports of one .py or .ipynb file, as {module, line, resolved}."""
    rel = path.relative_to(root)
    if path.suffix == ".ipynb":
        text = notebook_source(path)
        rel_for_ast = Path(rel.parts[0]) / "<nb>.py"
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel_for_ast = rel
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    found: list[dict] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = absolutise(node, rel_for_ast) if node.level else (node.module or "")
            mods = [(mod, node.lineno)] if mod else []
            # `from pkg import submodule` -- the submodule is the real target.
            if mod:
                for a in node.names:
                    mods.append((f"{mod}.{a.name}", node.lineno))
        else:
            continue
        for name, lineno in mods:
            hit = resolve(name, index)
            if hit and hit not in seen:
                seen.add(hit)
                found.append({"module": name, "line": lineno, "resolved": hit})
    return found


def _display(path: Path, root: Path) -> str:
    """Repo-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="docs/refactor/closure.json")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = list(iter_files(root))
    py_files = [f for f in files if f.suffix == ".py"]
    index = module_index(root, py_files)

    traceable = {f.relative_to(root).as_posix() for f in files if f.suffix in (".py", ".ipynb")}

    def package_inits(rel: str) -> list[str]:
        """Ancestor ``__init__.py`` files Python executes when importing ``rel``."""
        parts = Path(rel).parts[:-1]
        out = []
        for i in range(1, len(parts) + 1):
            cand = Path(*parts[:i]) / "__init__.py"
            if cand.as_posix() in traceable:
                out.append(cand.as_posix())
        return out

    # Edge table: path -> [{to, module, line}]
    edges: dict[str, list[dict]] = {}
    for rel in sorted(traceable):
        seen_to: set[str] = set()
        ed: list[dict] = []
        for i in imports_of(root / rel, root, index):
            for tgt in [i["resolved"], *package_inits(i["resolved"])]:
                if tgt in seen_to or tgt == rel:
                    continue
                seen_to.add(tgt)
                ed.append(
                    {"to": tgt, "module": i["module"], "line": i["line"]}
                    if tgt == i["resolved"]
                    else {"to": tgt, "module": f'<pkg init for {i["module"]}>', "line": i["line"]}
                )
        edges[rel] = ed
    for src, dst, why in EXTRA_EDGES:
        edges.setdefault(src, []).append({"to": dst, "module": "<dynamic>", "line": 0, "why": why})

    # Roots pruned from the tree by a later phase (phase 3 removed the D8
    # duplicate twins and archived the D9 exploratory pipeline) stay listed in
    # ROOTS as audit provenance. A re-run must skip them: seeding the BFS with a
    # path that no longer exists would silently count phantom nodes.
    live_roots: dict[str, list[str]] = {}
    pruned_roots: dict[str, list[str]] = {}
    for name, roots in ROOTS.items():
        live_roots[name] = [r for r in roots if r in traceable]
        gone = [r for r in roots if r not in traceable]
        if gone:
            pruned_roots[name] = gone

    # BFS per root set.
    closures: dict[str, dict[str, list[str]]] = {}
    for name, roots in live_roots.items():
        reached: dict[str, list[str]] = {}
        frontier = [(r, [r]) for r in roots]
        while frontier:
            cur, chain = frontier.pop(0)
            if cur in reached:
                continue
            reached[cur] = chain
            for e in edges.get(cur, []):
                if e["to"] not in reached:
                    frontier.append((e["to"], chain + [f'{cur}:{e["line"]} -> {e["to"]}']))
        closures[name] = reached

    union_paper: set[str] = set()
    for k in PAPER_ROOTS:
        union_paper |= set(closures[k])

    all_reached: set[str] = set()
    for c in closures.values():
        all_reached |= set(c)

    unreached = sorted(traceable - all_reached)

    # Reverse index: which root sets reach each file.
    reached_by: dict[str, list[str]] = {}
    for name, c in closures.items():
        for p in c:
            reached_by.setdefault(p, []).append(name)

    payload = {
        "roots": ROOTS,
        "live_roots": live_roots,
        "pruned_roots": pruned_roots,
        "paper_root_sets": PAPER_ROOTS,
        "extra_edges": [{"from": a, "to": b, "why": w} for a, b, w in EXTRA_EDGES],
        "summary": {
            "traceable_files": len(traceable),
            "paper_closure_size": len(union_paper),
            "reached_by_something": len(all_reached),
            "unreached": len(unreached),
            "pruned_root_paths": sum(len(v) for v in pruned_roots.values()),
            "per_root_set": {k: len(v) for k, v in closures.items()},
        },
        "paper_closure": sorted(union_paper),
        "unreached": unreached,
        "reached_by": dict(sorted(reached_by.items())),
        "closures": {k: {p: v[-1] if len(v) > 1 else "ROOT" for p, v in c.items()} for k, c in closures.items()},
        "edges": edges,
    }

    out = root / args.out
    if out.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {_display(out, root)}: it is the frozen "
            "phase-1 snapshot AUDIT.md cites by line. Re-run with an explicit "
            "--out (or --force if you really mean to re-baseline it)."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    s = payload["summary"]
    print(f"wrote {_display(out, root)}")
    print(f"  traceable: {s['traceable_files']}  paper closure: {s['paper_closure_size']}")
    print(f"  reached by something: {s['reached_by_something']}  unreached: {s['unreached']}")
    for k, v in s["per_root_set"].items():
        print(f"    {k:24s} {v}")
    if pruned_roots:
        print("  PRUNED ROOTS (removed from the tree by a later phase, skipped):")
        for k, v in pruned_roots.items():
            for p in v:
                print(f"    {k:24s} {p}")
    if unreached:
        print("  UNREACHED:")
        for p in unreached:
            print(f"    {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

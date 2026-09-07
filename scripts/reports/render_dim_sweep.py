#!/usr/bin/env python
"""Render the feature-width ablation's aggregate JSON as markdown tables.

Reads ``results/hypersigma_dim_sweep.json`` (written by
``scripts/experiments/run_hypersigma_dim_sweep.py``) and prints the tables that
``docs/feature_width_ablation.md`` carries, so those numbers are transcribed by
a script rather than by hand. Read-only with respect to the sweep; ``--out``
writes the markdown to a file instead of stdout.

    python scripts/reports/render_dim_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO / "results" / "hypersigma_dim_sweep.json"

#: Primary block: the reduced feature re-normalised to unit length, the geometry
#: the native features already have.
PRIMARY = "euclidean_l2"
SECONDARY = "euclidean"
SCENE_ORDER = ("houston", "trento", "muufl")
SCENE_TITLES = {"houston": "Houston", "trento": "Trento", "muufl": "MUUFL"}
CURVE_LABELS = {
    "pca_pool": "PCA (fit: all labelled features)",
    "pca_train": "PCA (fit: MFT train split)",
    "gaussian_rp": "Random projection (mean ± std, 5 seeds)",
}


def _cell(point: dict[str, Any], block: str, *, with_spread: bool) -> str:
    """One curve point as ``OA (Δ)``, with the seed spread where there is one."""
    stats = point[block]
    oa = f"{stats['oa_mean']:.2f}"
    if with_spread and stats["n_runs"] > 1:
        oa += f" ± {stats['oa_std']:.2f}"
    return f"{oa} ({stats['delta_oa_mean']:+.2f})"


def _significance(point: dict[str, Any], block: str) -> str:
    p = point[block]["max_p_value"]
    return "n.s." if p >= 0.05 else f"p<{0.05 if p > 0.001 else 0.001:g}"


def render(payload: dict[str, Any]) -> str:
    scenes = payload["scenes"]
    out: list[str] = []

    # --- control reproduction -----------------------------------------
    out.append("### Control: the cache reproduces the published cell\n")
    out.append(
        "| Scene | Published cell | Feature | Paper (Table 3) | Frozen run | Re-run | Drift |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for scene in (s for s in SCENE_ORDER if s in scenes):
        data = scenes[scene]
        control, published = data["control"], data["control"]["published"]
        check = control["check"]
        out.append(
            f"| {SCENE_TITLES.get(scene, scene)} | {data['cell']['description']} "
            f"| {data['cell']['feature_dim']}-d "
            f"| {published['OA']:.2f} ± {published['ci_95']:.2f} "
            f"| {published['frozen_run_OA']:.4f} "
            f"| {check['reproduced_oa']:.4f} "
            f"| {check['delta_oa']:+.5f} pp |"
        )
    out.append("")
    out.append(
        "`Paper` is Table 3's printed value; the gate compares against the frozen run's "
        "own full-precision mean, so `Drift` is the cache's, not the paper's rounding."
    )
    out.append("")

    # --- headline: matched capacity ------------------------------------
    out.append("### At CoFFE's width (128-d)\n")
    out.append(
        "| Scene | Native OA | PCA-128 | Random-proj-128 | CoFFE 128-d (Table 2) "
        "| CoFFE minus best HyperSIGMA |"
    )
    out.append("|---|---|---|---|---|---|")
    for scene in (s for s in SCENE_ORDER if s in scenes):
        data = scenes[scene]
        curves = data["curves"]
        native = data["control"][f"oa_{PRIMARY}"]["mean"]
        pca = curves.get("pca_pool", {}).get("128")
        rp = curves.get("gaussian_rp", {}).get("128")
        coffe = data["coffe_published_oa"]
        best = max([v for v in (native, pca[PRIMARY]["oa_mean"] if pca else None) if v is not None])
        out.append(
            f"| {SCENE_TITLES.get(scene, scene)} | {native:.2f} "
            f"| {_cell(pca, PRIMARY, with_spread=False) if pca else '—'} "
            f"| {_cell(rp, PRIMARY, with_spread=True) if rp else '—'} "
            f"| {coffe:.2f} | {coffe - best:+.2f} pp |"
        )
    out.append("")
    out.append(
        "Δ in parentheses is paired against that cell's own control on the identical "
        "2000 episodes. The last column is CoFFE's published Table 2 cell minus the best "
        "of its native width and its PCA-128 value, whichever is higher, so the margin is "
        "never flattered by the compression. (The random projection is below native in "
        "every scene, so it never enters.)"
    )
    out.append("")

    # --- per-scene curves ---------------------------------------------
    for scene in (s for s in SCENE_ORDER if s in scenes):
        data = scenes[scene]
        curves = data["curves"]
        widths = sorted({int(d) for curve in curves.values() for d in curve}, reverse=False)
        groups = [g for g in ("pca_pool", "pca_train", "gaussian_rp") if g in curves]

        out.append(f"### {SCENE_TITLES.get(scene, scene)} — OA by feature width\n")
        out.append(
            "| Width | Variance kept (pool fit) | "
            + " | ".join(CURVE_LABELS[g] for g in groups)
            + " |"
        )
        out.append("|---" * (len(groups) + 2) + "|")
        for width in widths:
            cells = []
            for group in groups:
                point = curves[group].get(str(width))
                cells.append(
                    _cell(point, PRIMARY, with_spread=group == "gaussian_rp") if point else "—"
                )
            pool = curves.get("pca_pool", {}).get(str(width), {})
            kept = pool.get("explained_variance_ratio")
            variance = f"{kept * 100:.2f}%" if kept is not None else "—"
            out.append(f"| {width} | {variance} | " + " | ".join(cells) + " |")
        native_dim = data["cell"]["feature_dim"]
        native_oa = data["control"][f"oa_{PRIMARY}"]["mean"]
        out.append(
            f"| **{native_dim} (native)** | 100% | **{native_oa:.2f}** |" + " |" * (len(groups) - 1)
        )
        out.append("")

        stage_groups = sorted(g for g in curves if g.startswith("stage"))
        if stage_groups:
            out.append("SEM stage blocks (the fused feature is 4 x 128 by construction):\n")
            out.append("| Map | OA | Δ vs 512-d | Paired test |")
            out.append("|---|---|---|---|")
            for group in stage_groups:
                for _dim, point in curves[group].items():
                    out.append(
                        f"| `{group}` | {point[PRIMARY]['oa_mean']:.2f} "
                        f"| {point[PRIMARY]['delta_oa_mean']:+.2f} pp "
                        f"| {_significance(point, PRIMARY)} |"
                    )
            out.append("")

    # --- the raw-projection block, for completeness ---------------------
    out.append("### Without re-normalising after projection\n")
    out.append(
        f"The `{SECONDARY}` block: the paper's metric on the raw projected coordinates, "
        "whose norms shrink and vary. Reported at 128-d only."
    )
    out.append("")
    out.append("| Scene | PCA-128 | Random-proj-128 |")
    out.append("|---|---|---|")
    for scene in (s for s in SCENE_ORDER if s in scenes):
        curves = scenes[scene]["curves"]
        pca = curves.get("pca_pool", {}).get("128")
        rp = curves.get("gaussian_rp", {}).get("128")
        out.append(
            f"| {SCENE_TITLES.get(scene, scene)} "
            f"| {_cell(pca, SECONDARY, with_spread=False) if pca else '—'} "
            f"| {_cell(rp, SECONDARY, with_spread=True) if rp else '—'} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=None, help="default: stdout")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(
            f"{args.source} not found — run scripts/experiments/run_hypersigma_dim_sweep.py first"
        )
    markdown = render(json.loads(args.source.read_text()))
    if args.out:
        args.out.write_text(markdown)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()

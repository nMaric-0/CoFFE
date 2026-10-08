"""Feature-width ablations must not be absorbed by the experiment-tree readers.

Three readers walk ``experiments/*/evaluations/*/results.json`` and roll it up
into committed artifacts. A feature-width ablation
(``scripts/experiments/run_hypersigma_dim_sweep.py``,
``docs/feature_width_ablation.md``) re-runs a *published* cell with the frozen
eval feature compressed before the prototypes are formed, so its results carry
that cell's ``model_type``, ``mode``, geometry and adapted checkpoint and differ
only in feature width — and its control run **is** the published configuration.
Nothing but the ``feature_reduction`` marker distinguishes the tree, so a reader
that classifies on those fields will let an ablation stand in for the cell, or
fold it into summary totals, GPU-hours and family counts.

``scripts/compile_results.py``'s exclusion is pinned in
``test_compile_results.py``; this module pins the other two roll-ups.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reports import aggregate_experiment_results, build_experiment_metadata

PUBLISHED = {
    "model_type": "HyperSIGMADual",
    "dataset": "trento",
    "mode": "spat_pool",
    "split": "all",
    "n_way": 6,
    "k_shot": 5,
    "k_query": 100,
    "num_episodes": 2000,
    "distance_metric": "euclidean",
    "euclidean": {"OA": {"mean": 91.07, "std": 1.9, "ci_95": 0.09}},
    "cosine": {"OA": {"mean": 89.51, "std": 2.1, "ci_95": 0.09}},
    "OA": {"mean": 91.07, "std": 1.9, "ci_95": 0.09},
}


def _write_eval(exp_dir: Path, eval_name: str, results: dict) -> None:
    eval_dir = exp_dir / "evaluations" / eval_name
    eval_dir.mkdir(parents=True)
    (eval_dir / "results.json").write_text(json.dumps(results))
    (eval_dir / "eval_config.json").write_text(json.dumps({"dataset": results["dataset"]}))
    (eval_dir / "eval_metadata.json").write_text(
        json.dumps({"status": "complete", "wallclock_seconds": 12.0})
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A published run, an ablation-only tree, and one tree holding both."""
    root = tmp_path / "experiments"

    published = root / "hypersigma_native_ablation_run1"
    published.mkdir(parents=True)
    (published / "pretrain_metadata.json").write_text(json.dumps({"name": "published"}))
    _write_eval(published, "native_trento_spatial_upscale", dict(PUBLISHED))

    reduced = root / "hypersigma_dimreduce_run1"
    reduced.mkdir(parents=True)
    _write_eval(
        reduced,
        "trento_backbonenative_upscale_frozen_spat768",
        {**PUBLISHED, "feature_reduction": {"kind": "identity", "dim": 768}},
    )
    _write_eval(
        reduced,
        "trento_pca128",
        {**PUBLISHED, "feature_reduction": {"kind": "pca", "dim": 128}},
    )

    mixed = root / "hypersigma_mixed_run1"
    mixed.mkdir(parents=True)
    (mixed / "pretrain_metadata.json").write_text(json.dumps({"name": "mixed"}))
    _write_eval(mixed, "real_eval", dict(PUBLISHED))
    _write_eval(
        mixed,
        "ablation_eval",
        {**PUBLISHED, "feature_reduction": {"kind": "gaussian_rp", "dim": 128, "seed": 42}},
    )
    return root


def test_aggregate_drops_the_ablation_tree_and_names_it(tree: Path) -> None:
    data = aggregate_experiment_results.collect(tree)

    assert "hypersigma_dimreduce_run1" not in data["experiments"]
    assert "hypersigma_dimreduce_run1" in data["skipped"]
    dropped = data["skipped_evaluations"]
    assert set(dropped) == {
        "hypersigma_dimreduce_run1/trento_backbonenative_upscale_frozen_spat768",
        "hypersigma_dimreduce_run1/trento_pca128",
        "hypersigma_mixed_run1/ablation_eval",
    }
    assert "pca, dim=128" in dropped["hypersigma_dimreduce_run1/trento_pca128"]


def test_aggregate_keeps_the_published_run(tree: Path) -> None:
    data = aggregate_experiment_results.collect(tree)

    published = data["experiments"]["hypersigma_native_ablation_run1"]
    assert set(published["evaluations"]) == {"native_trento_spatial_upscale"}
    assert data["num_evaluations"] == 2  # the published eval plus the mixed tree's real one


def test_aggregate_keeps_a_mixed_tree_minus_its_ablation_evals(tree: Path) -> None:
    data = aggregate_experiment_results.collect(tree)

    mixed = data["experiments"]["hypersigma_mixed_run1"]
    assert set(mixed["evaluations"]) == {"real_eval"}
    assert mixed["num_evaluations"] == 1


def test_metadata_rollup_drops_the_ablation_tree(tree: Path) -> None:
    data = build_experiment_metadata.build(tree)

    assert "hypersigma_dimreduce_run1" not in data["experiments"]
    assert "hypersigma_dimreduce_run1" in data["summary"]["skipped"]


def test_metadata_rollup_counts_no_ablation_evaluation(tree: Path) -> None:
    """The roll-up's totals are the thing an absorbed ablation would move."""
    data = build_experiment_metadata.build(tree)

    mixed = data["experiments"]["hypersigma_mixed_run1"]
    assert set(mixed["evaluations"]) == {"real_eval"}
    assert data["summary"]["num_evaluations"] == 2
    assert data["summary"]["num_evaluations_with_results"] == 2


def test_a_result_without_the_marker_is_still_rolled_up(tree: Path) -> None:
    """The exclusion must key on the marker alone, not on a directory name."""
    (
        tree / "hypersigma_dimreduce_run1" / "evaluations" / "trento_pca128" / "results.json"
    ).write_text(json.dumps(dict(PUBLISHED)))

    data = aggregate_experiment_results.collect(tree)
    assert "trento_pca128" in data["experiments"]["hypersigma_dimreduce_run1"]["evaluations"]

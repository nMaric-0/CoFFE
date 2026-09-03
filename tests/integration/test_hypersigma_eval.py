"""The HyperSIGMA evaluator end-to-end — the path every Table 3 cell came through.

``coffe.eval.hypersigma`` had **no test coverage at all** before phase 7, even
though PAPER_CANON §6 Table 3 is entirely its output. This runs it over a
synthetic mini-scene with randomly initialised ViT bodies (no released
checkpoint, PAPER_CANON §7.5), so what is verified is the protocol and the
written artifact, never an accuracy figure.

The one behaviour worth knowing about, and pinned here: the loop accumulates
**both** the cosine and the Euclidean confusion matrices in a single pass, and
``distance_metric`` only chooses which block is recorded as the run's primary
metric. That is why PAPER_CANON §8 D18 can say one Table 3 cell's
``eval_config.json`` records ``cosine`` while its published value is the
Euclidean sub-block — both were computed.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA

from tests.conftest import CANON_DATASET_SPECS, spec_for
from tests.equivalence._harness import write_scene

pytestmark = pytest.mark.slow

SCENE = "trento"  # smallest scene: 6 classes, 63 bands
K_SHOT = 5  # PAPER_CANON §4
K_QUERY = 3
EPISODES = 2


@pytest.fixture(scope="module")
def scene_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("hs_eval") / "raw"
    write_scene(spec_for(SCENE), root)
    return root


@pytest.fixture(scope="module")
def pca_path(tmp_path_factory) -> str:
    """A tiny deterministic PCA for the 3-band spatial front end."""
    bands = CANON_DATASET_SPECS[SCENE]["hsi_channels"]
    rng = np.random.default_rng(0)
    fitted = PCA(n_components=3, svd_solver="full").fit(
        rng.standard_normal((2048, bands)).astype(np.float32)
    )
    path = tmp_path_factory.mktemp("hs_pca") / "pca.pkl"
    with path.open("wb") as fh:
        pickle.dump(fitted, fh)
    return str(path)


def _evaluate(scene_root: Path, pca_path: str, output: Path | None = None, **overrides):
    from coffe.eval.hypersigma import run_evaluation

    params = dict(
        data_root=str(scene_root),
        split="all",
        k_shot=K_SHOT,
        k_query=K_QUERY,
        num_episodes=EPISODES,
        device="cpu",
        no_plots=True,
        # No released checkpoints: random-init bodies exercise the plumbing.
        spat_ckpt=None,
        spec_ckpt=None,
        pca_spat_path=pca_path,
        pca_stats_path=None,
        mode="fused",
        num_example_episodes=0,
        output=str(output) if output else None,
    )
    params.update(overrides)
    return run_evaluation(SCENE, **params)


@pytest.fixture(scope="module")
def results(scene_root: Path, pca_path: str):
    return _evaluate(scene_root, pca_path)


def test_both_distance_blocks_are_computed_in_one_pass(results) -> None:
    """PAPER_CANON §8 D18: cosine and Euclidean are both accumulated."""
    assert "cosine" in results and "euclidean" in results

    for block in ("cosine", "euclidean"):
        for metric in ("OA", "AA", "Kappa"):
            assert set(results[block][metric]) >= {"mean", "std", "ci_95"}
            assert 0.0 <= results[block][metric]["mean"] <= 100.0 or metric == "Kappa"


def test_the_evaluated_episodes_follow_the_paper_protocol(results) -> None:
    """N-way = the scene's class count, and the requested episode count is honoured."""
    n_classes = CANON_DATASET_SPECS[SCENE]["num_classes"]

    assert results["euclidean"]["num_episodes"] == EPISODES
    # Class-balanced queries => OA == AA by construction (PAPER_CANON §4).
    assert results["euclidean"]["OA"]["mean"] == pytest.approx(
        results["euclidean"]["AA"]["mean"], abs=1e-9
    )
    assert len(results["euclidean"]["per_class"]) == n_classes


def test_the_distance_metric_only_selects_the_primary_block(
    scene_root: Path, pca_path: str, tmp_path: Path
) -> None:
    """Switching ``distance_metric`` must not change either block's numbers.

    Both are computed either way, so a cosine run and a Euclidean run over the
    same seed must agree on both blocks — the flag is a *label*, not a
    computation switch. This is exactly the property D18 relies on.
    """
    euclidean = _evaluate(
        scene_root, pca_path, output=tmp_path / "euc.json", distance_metric="euclidean"
    )
    cosine = _evaluate(scene_root, pca_path, output=tmp_path / "cos.json", distance_metric="cosine")

    for block in ("cosine", "euclidean"):
        assert euclidean[block]["OA"]["mean"] == pytest.approx(
            cosine[block]["OA"]["mean"], rel=1e-9
        )

    # ...but the written artifact records which one was asked for.
    assert json.loads((tmp_path / "euc.json").read_text())["distance_metric"] == "euclidean"
    assert json.loads((tmp_path / "cos.json").read_text())["distance_metric"] == "cosine"


def test_the_written_results_json_records_the_route_and_its_geometry(
    scene_root: Path, pca_path: str, tmp_path: Path
) -> None:
    """``model_type: "HyperSIGMADual"`` is canonical already (PAPER_CANON §8 D16).

    The readers (``compile_results.py``, ``build_experiment_metadata.py``) match
    on that string, so it is part of the artifact contract — as is recording the
    input regime, which is what distinguishes the Table 3 rows from each other.
    """
    output = tmp_path / "results.json"
    _evaluate(scene_root, pca_path, output=output)

    written = json.loads(output.read_text())

    assert written["model_type"] == "HyperSIGMADual"
    assert written["dataset"] == SCENE
    assert written["k_shot"] == K_SHOT
    assert written["num_episodes"] == EPISODES
    assert written["mode"] == "fused"
    # Input regime (PAPER_CANON §1): patch-native here, so native_geometry off.
    assert written["native_geometry"] is False
    assert written["spat_patch_k"] == 3
    # Both blocks are persisted, not only the primary one.
    assert "cosine" in written and "euclidean" in written


@pytest.mark.parametrize("mode", ["spat_pool", "spec_pool", "fused"])
def test_every_table_3_feature_column_evaluates(scene_root: Path, pca_path: str, mode: str) -> None:
    """PAPER_CANON §6: the three feature columns — sp., sc., fu. — all run.

    ``spat_pool``/``spec_pool`` take a single-branch shortcut through
    ``forward_features``; ``fused`` needs both branches plus the SEM. All three
    must reach a full episode result.
    """
    results = _evaluate(scene_root, pca_path, mode=mode)

    assert results["euclidean"]["num_episodes"] == EPISODES
    assert 0.0 <= results["euclidean"]["OA"]["mean"] <= 100.0


def test_the_unadapted_ablation_needs_no_adapted_checkpoint(
    scene_root: Path, pca_path: str
) -> None:
    """``adapted_checkpoint="none"`` is the frozen route (PAPER_CANON §1)."""
    results = _evaluate(scene_root, pca_path, adapted_checkpoint="none")

    assert results["euclidean"]["num_episodes"] == EPISODES

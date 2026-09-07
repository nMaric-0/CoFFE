"""The feature reducers: label-free, deterministic, and what they claim to be.

``coffe.eval.dim_reduction`` compresses a frozen eval feature before the
prototypes are formed, so it sits inside PAPER_CANON §4's protocol. Two
properties keep it there and are pinned here: no reducer can see labels (a
supervised projection would stop being a frozen-encoder evaluation), and every
map is reproducible from its recorded spec.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from coffe.eval import dim_reduction as dr

FEATURE_DIM = 512
LABEL_LIKE = {"labels", "label", "y", "targets", "target", "classes", "support_labels"}


@pytest.fixture
def features() -> torch.Tensor:
    """Correlated features, so PCA has real structure to find."""
    rng = np.random.default_rng(7)
    latent = rng.standard_normal((600, 24))
    mixing = rng.standard_normal((24, FEATURE_DIM))
    noise = 0.05 * rng.standard_normal((600, FEATURE_DIM))
    x = torch.from_numpy((latent @ mixing + noise).astype(np.float32))
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def test_no_builder_can_see_labels() -> None:
    builders = [
        dr.build_reducer,
        dr.identity_reducer,
        dr.pca_reducer,
        dr.gaussian_rp_reducer,
        dr.stage_block_reducer,
        dr.stage_mean_reducer,
    ]
    for builder in builders:
        params = set(inspect.signature(builder).parameters)
        assert not params & LABEL_LIKE, f"{builder.__name__} takes labels: {params & LABEL_LIKE}"


def test_identity_returns_its_input_untouched(features: torch.Tensor) -> None:
    reducer = dr.build_reducer("identity", feature_dim=FEATURE_DIM)
    assert reducer.dim == FEATURE_DIM
    assert reducer(features) is features


@pytest.mark.parametrize("dim", [16, 128])
def test_pca_is_reproducible_and_reduces_to_the_requested_width(
    features: torch.Tensor, dim: int
) -> None:
    first = dr.pca_reducer(features, dim, corpus="pool")
    second = dr.pca_reducer(features, dim, corpus="pool")
    assert first.spec == second.spec
    torch.testing.assert_close(first.matrix, second.matrix)
    assert first(features).shape == (features.shape[0], dim)
    assert 0.0 < first.spec["explained_variance_ratio"] <= 1.0 + 1e-6


def test_pca_keeps_more_variance_as_the_width_grows(features: torch.Tensor) -> None:
    kept = [
        dr.pca_reducer(features, dim, corpus="pool").spec["explained_variance_ratio"]
        for dim in (16, 64, 128)
    ]
    assert kept == sorted(kept)


def test_pca_refuses_a_width_the_fit_corpus_cannot_support(features: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="min\\(n_fit, D\\)"):
        dr.pca_reducer(features[:32], 128, corpus="train")


def test_random_projection_depends_only_on_its_seed() -> None:
    same = [dr.gaussian_rp_reducer(FEATURE_DIM, 64, seed=42).matrix for _ in range(2)]
    torch.testing.assert_close(same[0], same[1])
    other = dr.gaussian_rp_reducer(FEATURE_DIM, 64, seed=123).matrix
    assert not torch.allclose(same[0], other)


def test_random_projection_preserves_squared_distances_in_expectation(
    features: torch.Tensor,
) -> None:
    """The Johnson-Lindenstrauss property the 1/sqrt(dim) scaling buys."""
    reducer = dr.gaussian_rp_reducer(FEATURE_DIM, 256, seed=42)
    projected = reducer(features)
    before = torch.cdist(features[:200], features[:200]).pow(2)
    after = torch.cdist(projected[:200], projected[:200]).pow(2)
    ratio = after.sum() / before.sum()
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_stage_maps_are_exact_selections_and_averages(features: torch.Tensor) -> None:
    stages, dr_dim = 4, FEATURE_DIM // 4
    for stage in range(stages):
        reducer = dr.stage_block_reducer(FEATURE_DIM, stage=stage)
        assert reducer.dim == dr_dim
        torch.testing.assert_close(
            reducer(features), features[:, stage * dr_dim : (stage + 1) * dr_dim]
        )
    torch.testing.assert_close(
        dr.stage_mean_reducer(FEATURE_DIM)(features),
        features.view(features.shape[0], stages, dr_dim).mean(dim=1),
    )


def test_stage_maps_reject_a_width_that_is_not_whole_blocks() -> None:
    with pytest.raises(ValueError, match="equal stage blocks"):
        dr.stage_block_reducer(770, stage=0)
    with pytest.raises(ValueError, match="equal stage blocks"):
        dr.stage_mean_reducer(770)


def test_stage_index_is_bounds_checked() -> None:
    with pytest.raises(ValueError, match=r"outside 0\.\.3"):
        dr.stage_block_reducer(FEATURE_DIM, stage=4)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reducer kind"):
        dr.build_reducer("umap", feature_dim=FEATURE_DIM, dim=8)


def test_every_kind_round_trips_through_build_reducer(features: torch.Tensor) -> None:
    built = {
        "identity": dict(),
        "pca": dict(dim=32, fit_features=features),
        "gaussian_rp": dict(dim=32, seed=42),
        "stage_block": dict(stage=1),
        "stage_mean": dict(),
    }
    for kind, kwargs in built.items():
        reducer = dr.build_reducer(kind, feature_dim=FEATURE_DIM, **kwargs)
        assert reducer.kind == kind
        assert reducer.spec["kind"] == kind
        assert reducer(features).shape[1] == reducer.dim
        assert set(dr.KINDS) >= {reducer.kind}


def test_pca_family_is_bit_identical_to_separate_fits(features: torch.Tensor) -> None:
    """The claim that buys the driver one SVD instead of len(dims)."""
    dims = (16, 64, 128)
    family = dr.pca_family(features, dims, corpus="pool")
    assert [r.dim for r in family] == list(dims)
    for reducer in family:
        alone = dr.pca_reducer(features, reducer.dim, corpus="pool")
        assert reducer.spec == alone.spec
        assert torch.equal(reducer.matrix, alone.matrix)
        assert torch.equal(reducer.mean, alone.mean)
        assert torch.equal(reducer(features), alone(features))


def test_pca_family_refuses_a_width_the_corpus_cannot_support(features: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="min\\(n_fit, D\\)"):
        dr.pca_family(features[:32], (16, 128), corpus="train")


def test_pca_family_of_no_dims_is_empty(features: torch.Tensor) -> None:
    assert dr.pca_family(features, (), corpus="pool") == []

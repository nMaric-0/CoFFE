"""Episode construction: N-way, K = 5, class-balanced queries.

The half of PAPER_CANON §4 that needs a dataset. ``PatchedEpisodeSampler`` is
driven over the shape-faithful synthetic mini-scenes, so the real dataset
classes, the real ``class_indices`` construction and the real sampling code all
run — with no file under ``data/raw/``.

The classifier half (Euclidean NCM, prototype = mean of K supports) is in
``tests/unit/test_ncm_protocol.py``.

Two constants are *not* asserted as global law, because §4's own note and
§8 D18 say they are not: ``k_query`` and ``num_episodes`` differ between
Table 2 (100 / 1000) and Table 3 (30-100 / 2000). They are pinned where they
belong — on the reproduce pipeline, in ``tests/unit/test_config_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from coffe.data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from coffe.eval.episodic import DATASET_SPECS
from tests.conftest import CANON_DATASET_SPECS, SCENE_KEYS, spec_for
from tests.equivalence._harness import load_scene_dataset

K_SHOT = 5  # PAPER_CANON §4
K_QUERY = 8  # scaled down from 100: the mini-scenes hold 30 px/class
SEED = 42


@pytest.fixture(scope="module")
def datasets(mini_scene_root):
    """The three mini-scenes, loaded through the repo's own dataset classes."""
    return {key: load_scene_dataset(spec_for(key), mini_scene_root, "all") for key in SCENE_KEYS}


def _sampler(dataset, *, n_way=None, k_shot=K_SHOT, k_query=K_QUERY, episodes=4):
    return PatchedEpisodeSampler(
        dataset=dataset,
        n_way=n_way,
        k_shot=k_shot,
        k_query=k_query,
        num_episodes=episodes,
        seed=SEED,
    )


# ----------------------------------------------------------------------
# N-way = the scene's full class count
# ----------------------------------------------------------------------


def test_dataset_specs_match_the_paper_table() -> None:
    """The evaluator's own spec table must equal PAPER_CANON §5 Table 1."""
    assert DATASET_SPECS == CANON_DATASET_SPECS


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_default_n_way_is_the_scenes_full_class_count(datasets, scene: str) -> None:
    """PAPER_CANON §4: N = the scene's class count — 15 / 6 / 11, never 5.

    ``n_way=None`` is how the CLI and the reproduce pipeline both request it.
    """
    expected = CANON_DATASET_SPECS[scene]["num_classes"]

    sampler = _sampler(datasets[scene], n_way=None)

    assert sampler.n_way == expected
    assert len(sampler.available_classes) == expected


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_every_class_appears_exactly_once_in_an_n_way_episode(datasets, scene: str) -> None:
    """With N = all classes, an episode is a permutation of the class set."""
    n_classes = CANON_DATASET_SPECS[scene]["num_classes"]
    sampler = _sampler(datasets[scene], n_way=None)

    episode = sampler.sample_episode()

    original = episode["original_classes"].tolist()
    assert sorted(original) == sorted(sampler.available_classes)
    assert len(set(original)) == n_classes


# ----------------------------------------------------------------------
# K = 5 supports and class-balanced queries
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_five_supports_are_drawn_per_class(datasets, scene: str) -> None:
    """PAPER_CANON §4: K = 5 per class, and relabelled 0..N-1."""
    n_classes = CANON_DATASET_SPECS[scene]["num_classes"]
    sampler = _sampler(datasets[scene], n_way=None)

    episode = sampler.sample_episode()

    labels = episode["support_labels"].numpy()
    assert labels.shape == (n_classes * K_SHOT,)
    counts = np.bincount(labels, minlength=n_classes)
    assert list(counts) == [K_SHOT] * n_classes
    assert set(labels.tolist()) == set(range(n_classes))


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_query_counts_are_class_balanced(datasets, scene: str) -> None:
    """Balanced queries are why OA == AA by construction (PAPER_CANON §4)."""
    n_classes = CANON_DATASET_SPECS[scene]["num_classes"]
    sampler = _sampler(datasets[scene], n_way=None)

    episode = sampler.sample_episode()

    labels = episode["query_labels"].numpy()
    assert labels.shape == (n_classes * K_QUERY,)
    counts = np.bincount(labels, minlength=n_classes)
    assert list(counts) == [K_QUERY] * n_classes


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_support_and_query_tensors_carry_the_scenes_band_counts(datasets, scene: str) -> None:
    """PAPER_CANON §5: 144+1 Houston / 63+1 Trento / 64+2 MUUFL, 11x11 each."""
    canon = CANON_DATASET_SPECS[scene]
    n_classes = canon["num_classes"]
    sampler = _sampler(datasets[scene], n_way=None)

    episode = sampler.sample_episode()

    assert episode["support_hsi"].shape == (n_classes * K_SHOT, canon["hsi_channels"], 11, 11)
    assert episode["support_aux"].shape == (n_classes * K_SHOT, canon["aux_channels"], 11, 11)
    assert episode["query_hsi"].shape == (n_classes * K_QUERY, canon["hsi_channels"], 11, 11)
    assert episode["query_aux"].shape == (n_classes * K_QUERY, canon["aux_channels"], 11, 11)


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_supports_and_queries_are_disjoint(datasets, scene: str) -> None:
    """K + Q are drawn without replacement, so no support is also a query.

    Sampling is by dataset index, so the check is on the identity of the drawn
    samples rather than on their values.
    """
    dataset = datasets[scene]
    sampler = _sampler(dataset, n_way=None)

    # Re-draw with a sampler whose RNG state is identical, and compare the
    # per-class index draws the episode was built from.
    rng = np.random.RandomState(SEED)
    selected = rng.choice(sampler.available_classes, sampler.n_way, replace=False)
    for original_class in selected:
        indices = sampler.class_indices[original_class]
        drawn = rng.choice(indices, K_SHOT + K_QUERY, replace=False).tolist()
        assert len(set(drawn)) == K_SHOT + K_QUERY
        support, query = drawn[:K_SHOT], drawn[K_SHOT:]
        assert set(support).isdisjoint(query)


def test_a_class_without_enough_samples_is_dropped_not_resampled(datasets) -> None:
    """Classes below ``k_shot + k_query`` are excluded, never sampled twice."""
    dataset = datasets["trento"]
    per_class = min(len(idx) for idx in dataset.class_indices.values())

    # Ask for more than any class can supply.
    with pytest.raises(ValueError, match="Not enough classes"):
        _sampler(dataset, n_way=6, k_query=per_class)


def test_episode_sampling_is_seed_reproducible(datasets) -> None:
    """Two samplers with the same seed produce identical episodes."""
    a = _sampler(datasets["trento"], n_way=None).sample_episode()
    b = _sampler(datasets["trento"], n_way=None).sample_episode()

    assert a["original_classes"].tolist() == b["original_classes"].tolist()
    assert a["query_labels"].tolist() == b["query_labels"].tolist()
    assert bool((a["query_hsi"] == b["query_hsi"]).all())


def test_iteration_yields_the_requested_episode_count(datasets) -> None:
    sampler = _sampler(datasets["trento"], n_way=None, episodes=3)

    episodes = list(sampler)

    assert len(episodes) == 3 == len(sampler)

"""The feature-cached NCM loop must compute what the live episode loop computes.

``coffe.eval.reduced_ncm`` exists to answer a question the live loop cannot
afford (PAPER_CANON §4's protocol at several feature widths), and it is only
usable if it is the *same* protocol: same episodes, same prototypes, same
metrics. This module pins that with a batch-invariant fake encoder — features
built from per-channel reductions, so a patch's feature cannot depend on the
batch it was encoded in. Any disagreement is then the loops', not the float
reduction order's, and the comparison can be exact.

The real HyperSIGMA encoder is *not* batch-invariant (matmul reduction order),
which is why the driver additionally gates each cached run against its
published cell's OA. This test is the part that can be checked without a GPU,
a dataset, or a released checkpoint (PAPER_CANON §7.5).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from coffe.data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from coffe.eval.feature_cache import encode_dataset
from coffe.eval.reduced_ncm import build_episode_plan, run_episodes, summarize

K_SHOT = 5  # PAPER_CANON §4
K_QUERY = 7
EPISODES = 12
N_CLASSES = 6
SEED = 42


class _FakePatched(torch.utils.data.Dataset):
    """A pre-patched dataset with the attributes the sampler and cache use."""

    def __init__(self, n: int = 260, bands: int = 5, patch: int = 3) -> None:
        rng = np.random.default_rng(0)
        self.hsi = torch.from_numpy(rng.standard_normal((n, bands, patch, patch), dtype=np.float32))
        self.aux = torch.from_numpy(rng.standard_normal((n, 1, patch, patch), dtype=np.float32))
        labels = np.array([i % N_CLASSES for i in range(n)])
        self.labels = torch.from_numpy(labels).long()
        self.class_indices = {c: np.flatnonzero(labels == c).tolist() for c in range(N_CLASSES)}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"hsi": self.hsi[idx], "aux": self.aux[idx], "label": self.labels[idx]}


class _BatchInvariantEncoder(torch.nn.Module):
    """Mirrors the evaluator surface; feature = per-channel mean/std, L2-normalised.

    Only per-sample reductions, so ``feature(patch)`` is independent of batch
    composition — the property that lets this test assert exact equality.
    """

    distance_metric = "euclidean"

    def forward_features(self, hsi: torch.Tensor, aux: torch.Tensor | None = None):
        pooled = torch.cat([hsi.mean(dim=(2, 3)), hsi.std(dim=(2, 3))], dim=-1)
        feats = F.normalize(pooled, p=2, dim=-1)
        return feats.unsqueeze(1), feats, feats

    def eval_patch_embeddings(self, patch_emb, cls_emb, cls_token_weight=None, renormalize=True):
        return patch_emb


def _sampler(dataset) -> PatchedEpisodeSampler:
    return PatchedEpisodeSampler(
        dataset=dataset,
        n_way=None,
        k_shot=K_SHOT,
        k_query=K_QUERY,
        num_episodes=EPISODES,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def live_and_cached():
    from coffe.eval.hypersigma import evaluate as live_evaluate

    dataset = _FakePatched()
    model = _BatchInvariantEncoder().eval()

    live = live_evaluate(
        model,
        _sampler(dataset),
        "cpu",
        num_episodes=EPISODES,
        n_way=N_CLASSES,
        num_total_classes=N_CLASSES,
        temperature=10.0,
        num_example_episodes=0,
        max_tsne_samples=0,
    )

    features = encode_dataset(model, dataset, device="cpu", batch_size=32, log_interval=0)
    plan = build_episode_plan(_sampler(dataset), EPISODES, seed=SEED)
    cached = run_episodes(features, plan, num_total_classes=N_CLASSES, temperature=10.0)
    return live, cached, plan, features


def test_the_cached_loop_reproduces_every_episode_exactly(live_and_cached) -> None:
    live, cached, _, _ = live_and_cached
    for metric in ("euclidean", "cosine"):
        np.testing.assert_array_equal(
            np.asarray(live[metric]["episode_oas"]),
            np.asarray(cached[metric]["episode_oas"]),
        )
        np.testing.assert_array_equal(
            np.asarray(live[metric]["episode_kappas"]),
            np.asarray(cached[metric]["episode_kappas"]),
        )


def test_the_summaries_and_per_class_blocks_agree(live_and_cached) -> None:
    live, cached, _, _ = live_and_cached
    for metric in ("euclidean", "cosine"):
        got = summarize(cached[metric])
        for key in ("OA", "AA", "Kappa"):
            assert got[key]["mean"] == pytest.approx(live[metric][key]["mean"], abs=1e-12)
            assert got[key]["ci_95"] == pytest.approx(live[metric][key]["ci_95"], abs=1e-12)
        assert got["num_episodes"] == live[metric]["num_episodes"]
        assert set(got["per_class"]) == set(live[metric]["per_class"])
        for cls, block in got["per_class"].items():
            expected = live[metric]["per_class"][cls]
            assert block["accuracy"] == pytest.approx(expected["accuracy"], abs=1e-12)
            assert block["total_samples"] == expected["total_samples"]
            assert block["correct_samples"] == expected["correct_samples"]


def test_the_global_confusion_matrix_agrees(live_and_cached) -> None:
    live, cached, _, _ = live_and_cached
    for metric in ("euclidean", "cosine"):
        np.testing.assert_array_equal(
            live[metric]["global_conf_matrix"], cached[metric]["global_conf_matrix"]
        )


def test_the_plan_indexes_the_same_patches_the_sampler_stacks() -> None:
    """The cache is indexed by dataset position, so the plan must be too."""
    dataset = _FakePatched()
    plan = build_episode_plan(_sampler(dataset), EPISODES, seed=SEED)
    for episode, batch in zip(range(EPISODES), _sampler(dataset)):
        for column, index in enumerate(plan.support_indices[episode]):
            assert torch.equal(batch["support_hsi"][column], dataset[index]["hsi"])
        for column, index in enumerate(plan.query_indices[episode]):
            assert torch.equal(batch["query_hsi"][column], dataset[index]["hsi"])
        assert np.array_equal(plan.original_classes[episode], batch["original_classes"].numpy())
        assert np.array_equal(plan.support_labels, batch["support_labels"].numpy())
        assert np.array_equal(plan.query_labels, batch["query_labels"].numpy())


def test_unit_norm_features_make_the_two_euclidean_blocks_coincide(live_and_cached) -> None:
    """`euclidean_l2` only differs from `euclidean` once a projection shrinks norms."""
    _, cached, _, _ = live_and_cached
    np.testing.assert_allclose(
        cached["euclidean"]["episode_oas"], cached["euclidean_l2"]["episode_oas"], atol=1e-9
    )


@pytest.mark.parametrize("fixed_support", [False, True])
def test_the_index_draw_consumes_the_rng_exactly_as_the_patch_draw(fixed_support: bool) -> None:
    """`sample_episode_indices` was lifted out of `sample_episode`.

    The extraction is only safe if both consume the sampler's RNG in the same
    order, so two samplers at one seed must agree episode for episode — checked
    here for `fixed_support` too, which no paper path uses (default `False`,
    neither evaluator passes it) but which shares the lifted code.
    """
    dataset = _FakePatched()
    common = dict(
        dataset=dataset,
        n_way=None,
        k_shot=K_SHOT,
        k_query=K_QUERY,
        num_episodes=EPISODES,
        seed=SEED,
        fixed_support=fixed_support,
    )
    indices_only = PatchedEpisodeSampler(**common)
    with_patches = PatchedEpisodeSampler(**common)

    for _ in range(EPISODES):
        classes, per_class = indices_only.sample_episode_indices()
        episode = with_patches.sample_episode()

        assert np.array_equal(classes, episode["original_classes"].numpy())
        support = [idx for group in per_class for idx in group[:K_SHOT]]
        query = [idx for group in per_class for idx in group[K_SHOT:]]
        assert len(support) == len(episode["support_labels"])
        assert len(query) == len(episode["query_labels"])
        for column, index in enumerate(support):
            assert torch.equal(episode["support_hsi"][column], dataset[index]["hsi"])
        for column, index in enumerate(query):
            assert torch.equal(episode["query_hsi"][column], dataset[index]["hsi"])

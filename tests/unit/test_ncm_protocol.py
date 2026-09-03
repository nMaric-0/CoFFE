"""The evaluation protocol is a Euclidean nearest-class-mean classifier.

PAPER_CANON §4 makes these law: prototype = mean of the K = 5 support features,
queries assigned by Euclidean nearest class mean, no learnable similarity
parameters. Two things are checked here:

1. **The classifier is the textbook one.** On random features the evaluator's
   assignments must equal ``sklearn.neighbors.NearestCentroid``'s — an
   independent implementation, not a second copy of ours.
2. **Euclidean is the default everywhere.** ``cosine`` survives as an option
   value only (PAPER_CANON §1, phase-4 gate). Note this is about *defaults*:
   every paper run passes ``distance_metric`` explicitly, and one Table 3 cell
   passes ``cosine`` (§8 D18).

Prose note: this classifier is nearest-class-mean on fixed features (Mensink et
al.), not a prototypical network (Snell et al.) — nothing here is trained.

The episode *sampling* side of §4 (N-way = full class count, K = 5,
class-balanced queries) is in
``tests/integration/test_eval_protocol.py``, which needs a dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.neighbors import NearestCentroid

from coffe.eval.episodic import _DEFAULT_ARGS
from coffe.models import CoFFE, MFTOriginal

K_SHOT = 5  # PAPER_CANON §4
DIM = 128


def _ncm() -> CoFFE:
    """A CoFFE instance used only for its prototype/distance maths.

    The classifier has no learnable similarity parameters, so the encoder
    weights are irrelevant here — the features are supplied directly.
    """
    return CoFFE(
        hsi_channels=8,
        aux_channels=1,
        embed_dim=DIM,
        num_heads=2,
        num_layers=2,
        use_projection=False,
    ).eval()


def _support_and_queries(n_way: int, dim: int = DIM, seed: int = 0, n_query: int = 40):
    gen = torch.Generator().manual_seed(seed)
    labels = torch.arange(n_way).repeat_interleave(K_SHOT)
    # Class-separated support clouds so the assignment problem is non-degenerate
    # (all-random features would leave many near-ties).
    centres = torch.randn(n_way, dim, generator=gen) * 3.0
    support = centres[labels] + torch.randn(n_way * K_SHOT, dim, generator=gen)
    queries = torch.randn(n_query, dim, generator=gen) * 3.0
    return support, labels, queries


# ----------------------------------------------------------------------
# The classifier itself
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_way", [15, 6, 11], ids=["houston", "trento", "muufl"])
def test_assignments_equal_sklearn_nearest_centroid(n_way: int) -> None:
    """Our argmax over ``-d²`` must agree with sklearn's NearestCentroid.

    Run at each scene's class count (PAPER_CANON §5) so the N-way sizes the
    paper actually evaluates are the ones compared.
    """
    model = _ncm()
    support, labels, queries = _support_and_queries(n_way)

    with torch.no_grad():
        logits = model._forward_mean_features(support, labels, queries)
    ours = logits.argmax(dim=1).numpy()

    reference = NearestCentroid(metric="euclidean").fit(
        support.numpy().astype(np.float64), labels.numpy()
    )
    theirs = reference.predict(queries.numpy().astype(np.float64))

    assert np.array_equal(ours, theirs)


def test_prototype_is_the_mean_of_the_k_supports() -> None:
    """PAPER_CANON §4: "prototype = mean of the 5 support features"."""
    model = _ncm()
    support, labels, _queries = _support_and_queries(n_way=6)

    prototypes = model.compute_prototypes(support, labels)

    assert prototypes.shape == (6, DIM)
    for c in range(6):
        members = support[labels == c]
        assert members.shape[0] == K_SHOT
        assert torch.allclose(prototypes[c], members.mean(dim=0), atol=1e-6)


def test_euclidean_logits_are_the_negated_squared_distance() -> None:
    """The score is ``-||q - p||²``, so argmax == nearest prototype."""
    model = _ncm()
    support, labels, queries = _support_and_queries(n_way=6)

    with torch.no_grad():
        prototypes = model.compute_prototypes(support, labels)
        logits = model._forward_mean_features(support, labels, queries)

    expected = -torch.cdist(queries, prototypes, p=2).pow(2)
    assert torch.allclose(logits, expected, atol=1e-5)


def test_no_learnable_similarity_parameters() -> None:
    """PAPER_CANON §4: the classifier is parameter-free.

    Nothing named like a similarity/classifier head may exist on the encoder —
    old checkpoints' ``DenseSimilarity`` weights are dropped on load.
    """
    model = _ncm()
    offenders = [
        name
        for name, _ in model.named_parameters()
        if any(tag in name.lower() for tag in ("similarity", "classifier", "logit_scale"))
    ]
    assert offenders == []


def test_mean_distances_mode_is_selectable_and_differs() -> None:
    """``mean_distances`` exists as an option; ``mean_features`` is the protocol."""
    model = _ncm()
    support, labels, queries = _support_and_queries(n_way=6)

    with torch.no_grad():
        mean_features = model._forward_mean_features(support, labels, queries)
        mean_distances = model._forward_mean_distances(support, labels, queries)

    assert model.prototype_mode == "mean_features"
    assert _DEFAULT_ARGS["prototype_mode"] == "mean_features"
    assert not torch.allclose(mean_features, mean_distances)


# ----------------------------------------------------------------------
# Euclidean is the default; cosine is a non-default option
# ----------------------------------------------------------------------


def test_coffe_defaults_to_euclidean() -> None:
    assert CoFFE(hsi_channels=8, aux_channels=1).distance_metric == "euclidean"


def test_mft_control_defaults_to_euclidean() -> None:
    # 63 bands (Trento): MFT's valid 9-wide spectral conv needs > 8.
    assert MFTOriginal(hsi_channels=63, aux_channels=1).distance_metric == "euclidean"


def test_hypersigma_few_shot_defaults_to_euclidean() -> None:
    """Checked on the signature, so no ViT body has to be built."""
    import inspect

    from coffe.models.hypersigma import HyperSIGMAFewShot

    default = inspect.signature(HyperSIGMAFewShot.__init__).parameters["distance_metric"].default
    assert default == "euclidean"


def test_the_evaluator_defaults_that_are_the_paper_protocol() -> None:
    """The subset of ``_DEFAULT_ARGS`` that does match PAPER_CANON §4.

    ``n_way=None`` means "all classes in the scene" — the N-way protocol.
    The defaults that *don't* match are pinned separately below, so this test
    cannot be read as a clean bill of health for the whole table.
    """
    assert _DEFAULT_ARGS["distance_metric"] == "euclidean"
    assert _DEFAULT_ARGS["k_shot"] == K_SHOT
    assert _DEFAULT_ARGS["n_way"] is None
    assert _DEFAULT_ARGS["pool_sigma"] is None
    assert _DEFAULT_ARGS["temperature"] == 10.0
    assert _DEFAULT_ARGS["prototype_mode"] == "mean_features"


def test_the_evaluator_defaults_that_do_not_match_the_paper() -> None:
    """The rest of ``_DEFAULT_ARGS``: stale values, pinned as the gap they are.

    Five of these defaults contradict the canon (reported as S4 at the phase-7
    gate, and the same class of staleness as PAPER_CANON §8 D4):

    * ``num_heads``/``num_layers`` are 8/4 where §2 says **2/2**;
    * ``lambda_factor`` is 2.0 where every paper CoFFE run records **0.5**
      (§8 D3);
    * ``use_projection`` is ``True`` where §4 says the head is **off** at eval;
    * ``k_query`` is 15, which matches no table (Table 2 used 100, Table 3
      used 30-100) — so unlike ``num_episodes`` this is not a per-table
      difference, just a default nothing published used.

    Harmless for every published number, because the reproduce pipeline and
    ``coffe.runners.eval_runner`` supply all of them explicitly — the frozen
    ``eval_config.json``/``pretrain_config.yaml`` are the source of truth on
    the paper path, and no default is ever consulted there. It bites only a
    caller who invokes ``run_evaluation`` with nothing but a checkpoint.

    Pinned rather than corrected: changing a default is a behaviour change,
    which phase 7 may not make. If Nikola flips any of them at the gate,
    update this test in that same commit.
    """
    assert _DEFAULT_ARGS["num_heads"] == 8  # canon §2: 2
    assert _DEFAULT_ARGS["num_layers"] == 4  # canon §2: 2
    assert _DEFAULT_ARGS["lambda_factor"] == 2.0  # canon §8 D3: 0.5
    assert _DEFAULT_ARGS["use_projection"] is True  # canon §4: False
    assert _DEFAULT_ARGS["k_query"] == 15  # canon §4: 100 (Table 2)
    # num_episodes genuinely is per-table (1000 Table 2 / 2000 Table 3, §8 D18),
    # so its default is a choice rather than a contradiction.
    assert _DEFAULT_ARGS["num_episodes"] == 2000


def test_cosine_is_still_selectable() -> None:
    """Cosine must keep working — it produced one Table 3 cell (§8 D18)."""
    model = CoFFE(
        hsi_channels=8,
        aux_channels=1,
        embed_dim=DIM,
        num_heads=2,
        num_layers=2,
        use_projection=False,
        distance_metric="cosine",
    ).eval()
    support, labels, queries = _support_and_queries(n_way=6)

    with torch.no_grad():
        support_n = torch.nn.functional.normalize(support, p=2, dim=-1)
        queries_n = torch.nn.functional.normalize(queries, p=2, dim=-1)
        logits = model._forward_mean_features(support_n, labels, queries_n)

    prototypes = model.compute_prototypes(support_n, labels)
    expected = queries_n @ prototypes.T * model.temperature
    assert torch.allclose(logits, expected, atol=1e-5)


def test_cli_default_distance_metric_is_euclidean() -> None:
    """``scripts/evaluate.py --help`` must not offer cosine as the default."""
    import runpy
    import sys
    from unittest.mock import patch

    # The parser is built under ``__main__``; parse an argv that supplies only
    # the two required flags and read the resulting namespace.
    argv = ["evaluate.py", "--checkpoint", "random", "--dataset", "houston"]
    captured: dict = {}

    def fake_main(args):
        captured["args"] = args

    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
    with patch.object(sys, "argv", argv), patch("coffe.eval.episodic.main", fake_main):
        runpy.run_path(str(script), run_name="__main__")

    assert captured["args"].distance_metric == "euclidean"
    assert captured["args"].n_way is None
    assert captured["args"].k_shot == K_SHOT

"""One decision per labelled pixel: the shape a classification map needs.

``coffe.eval.prediction_map`` is not a second protocol — it is PAPER_CANON §4's
classifier run once per sample instead of once per episode, so that a pixel has
a single decision to paint. These tests hold it to exactly that:

* the pass **partitions** the labelled pool into ``C * K`` supports and every
  other sample as a query, classified exactly once;
* the supports are the episode sampler's own ``fixed_support`` draw at that
  seed, not a second sampling implementation;
* the assignments agree with ``sklearn.neighbors.NearestCentroid`` — an
  independent nearest-class-mean, as ``tests/unit/test_ncm_protocol.py`` also
  insists;
* the painted canvases say what they claim about the pixels they cover.

The scenes are the synthetic mini-scenes; nothing under ``data/raw/`` is read.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.neighbors import NearestCentroid

from coffe.data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from coffe.data.scene_coords import recover
from coffe.eval.prediction_map import predict_all
from tests.conftest import spec_for
from tests.equivalence._harness import load_scene_dataset

K_SHOT = 5  # PAPER_CANON §4
DIM = 32
SEED = 42


@pytest.fixture(scope="module")
def dataset(mini_scene_root):
    return load_scene_dataset(spec_for("trento"), mini_scene_root, "all")


@pytest.fixture(scope="module")
def features(dataset):
    """Stand-in features: the module's contract is over ``[n, D]``, not an encoder.

    ``test_the_encoder_path_runs_end_to_end`` covers the real one.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randn(len(dataset), DIM, generator=generator)


@pytest.fixture(scope="module")
def prediction(features, dataset):
    return predict_all(features, dataset, scene="trento", k_shot=K_SHOT, seed=SEED)


# ----------------------------------------------------------------------
# The partition
# ----------------------------------------------------------------------


def test_every_labelled_sample_is_either_support_or_classified_once(prediction, dataset):
    covered = np.concatenate([prediction.support_indices, prediction.query_indices])

    assert len(np.unique(covered)) == len(covered)  # nothing twice
    assert sorted(covered.tolist()) == list(range(len(dataset)))


def test_the_support_is_k_shot_per_class(prediction):
    assert len(prediction.support_indices) == prediction.num_classes * K_SHOT
    per_class = np.bincount(prediction.support_classes)[prediction.classes]
    assert set(per_class.tolist()) == {K_SHOT}


def test_the_support_is_the_samplers_own_fixed_draw(features, dataset):
    """Not a second sampling implementation: the same indices the protocol's
    sampler picks for ``fixed_support`` at this seed."""
    sampler = PatchedEpisodeSampler(
        dataset,
        n_way=None,
        k_shot=K_SHOT,
        k_query=1,
        num_episodes=1,
        seed=SEED,
        fixed_support=True,
    )
    expected = sorted(
        int(i) for indices in sampler.support_indices_by_class.values() for i in indices
    )

    prediction = predict_all(features, dataset, k_shot=K_SHOT, seed=SEED)

    assert sorted(prediction.support_indices.tolist()) == expected


def test_queries_keep_their_true_class(prediction, dataset):
    labels = dataset.labels.numpy()
    assert np.array_equal(prediction.query_classes, labels[prediction.query_indices])


def test_predictions_are_original_class_labels(prediction):
    assert set(np.unique(prediction.predictions)).issubset(set(prediction.classes.tolist()))
    assert len(prediction.predictions) == len(prediction.query_indices)


# ----------------------------------------------------------------------
# The classifier
# ----------------------------------------------------------------------


def test_assignments_match_an_independent_nearest_class_mean(prediction, features):
    reference = NearestCentroid(metric="euclidean").fit(
        features[prediction.support_indices].numpy(), prediction.support_classes
    )
    expected = reference.predict(features[prediction.query_indices].numpy())

    assert np.array_equal(prediction.predictions, expected)


def test_the_reported_oa_is_the_pass_accuracy(prediction):
    correct = (prediction.predictions == prediction.query_classes).mean() * 100

    assert prediction.metrics()["OA"] == pytest.approx(correct)


def test_per_class_accuracy_matches_the_confusion_diagonal(prediction):
    conf = prediction.confusion()
    per_class = prediction.per_class_accuracy()

    for j, original in enumerate(prediction.classes):
        expected = conf[j, j] / conf[j, :].sum() * 100
        assert per_class[int(original)] == pytest.approx(expected)


def test_cosine_is_selectable_and_scale_invariant(features, dataset):
    """``cosine`` stays an option value (PAPER_CANON §1); the temperature
    scales the logits, never the argmax."""
    hot = predict_all(features, dataset, metric="cosine", temperature=1.0, seed=SEED)
    cold = predict_all(features, dataset, metric="cosine", temperature=25.0, seed=SEED)

    assert np.array_equal(hot.predictions, cold.predictions)


def test_the_seed_fixes_the_pass(features, dataset):
    same = predict_all(features, dataset, seed=SEED)
    other = predict_all(features, dataset, seed=SEED + 1)

    assert np.array_equal(
        predict_all(features, dataset, seed=SEED).support_indices, same.support_indices
    )
    assert not np.array_equal(same.support_indices, other.support_indices)


def test_a_class_too_small_to_classify_is_reported(caplog):
    """A class with fewer than K+1 samples cannot be classified, and its pixels
    would be blank under a ground truth that shows them — so it is announced.
    No paper scene comes close; the guard is for the surprise."""

    class TooSmall:
        def __init__(self):
            self.labels = torch.tensor([1] * 10 + [2] * 10 + [3] * 3)
            self.class_indices = {
                1: list(range(10)),
                2: list(range(10, 20)),
                3: [20, 21, 22],
            }

        def __len__(self):
            return 23

    with caplog.at_level("WARNING"):
        prediction = predict_all(torch.randn(23, DIM), TooSmall(), k_shot=K_SHOT, seed=1)

    assert prediction.classes.tolist() == [1, 2]
    assert len(prediction.query_indices) == 10  # class 3's three samples are not queried
    assert "left unclassified" in caplog.text


def test_a_feature_matrix_of_the_wrong_length_is_rejected(dataset):
    with pytest.raises(ValueError, match="share an index space"):
        predict_all(torch.randn(len(dataset) - 1, DIM), dataset)


# ----------------------------------------------------------------------
# The painted canvases
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def coords(dataset):
    """A coordinate table for the mini scene, built the way the real one is.

    The samples are laid out row by row on a canvas, the mask is what that
    layout labels, and ``recover`` is asked to find the order back — so the
    fixture exercises the recovery rather than hand-building the table.
    """
    width = 32
    labels = dataset.labels.numpy()
    height = int(np.ceil(len(labels) / width))
    mask = np.zeros((height, width), dtype=np.int64)
    for i, label in enumerate(labels):
        mask[i // width, i % width] = label
    return recover("trento_mini", mask, labels, order_train="row_major")


def test_the_canvases_cover_the_pixels_they_claim(prediction, coords, dataset):
    maps = prediction.to_maps(coords)

    assert np.array_equal(maps["truth"], coords.gt_map())
    assert maps["support"].sum() == prediction.num_classes * K_SHOT
    # Every labelled pixel carries a class in the prediction panel: a decision
    # for the queries, the known class for the supports.
    assert np.count_nonzero(maps["prediction"]) == len(dataset)


def test_the_prediction_canvas_holds_the_decisions(prediction, coords):
    maps = prediction.to_maps(coords)

    at_queries = maps["prediction"][
        coords.rows[prediction.query_indices], coords.cols[prediction.query_indices]
    ]
    at_supports = maps["prediction"][
        coords.rows[prediction.support_indices], coords.cols[prediction.support_indices]
    ]

    assert np.array_equal(at_queries, prediction.predictions)
    assert np.array_equal(at_supports, prediction.support_classes)


def test_the_agreement_canvas_marks_right_wrong_and_support(prediction, coords):
    maps = prediction.to_maps(coords)
    correct = prediction.predictions == prediction.query_classes

    assert (maps["agreement"] == 1).sum() == correct.sum()
    assert (maps["agreement"] == 2).sum() == (~correct).sum()
    assert (maps["agreement"] == 3).sum() == len(prediction.support_indices)
    # Unlabelled scene pixels stay 0.
    assert (maps["agreement"] > 0).sum() == len(coords)


def test_the_summary_blob_is_json_serialisable(prediction):
    import json

    blob = json.loads(json.dumps(prediction.to_dict()))

    assert blob["num_support"] + blob["num_queries"] == len(prediction.classes) * K_SHOT + len(
        prediction.query_indices
    )
    assert 0.0 <= blob["metrics"]["OA"] <= 100.0


# ----------------------------------------------------------------------
# The real encoder, once
# ----------------------------------------------------------------------


def test_the_encoder_path_runs_end_to_end(mini_scene_root):
    """A frozen CoFFE encoded once, then one decision per sample — the wiring
    ``scripts/predict_map.py`` uses."""
    from coffe.eval.feature_cache import encode_dataset
    from coffe.models import CoFFE

    dataset = load_scene_dataset(spec_for("trento"), mini_scene_root, "all")
    model = CoFFE(
        hsi_channels=63,
        aux_channels=1,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        patch_size=11,
        use_projection=False,
    )
    features = encode_dataset(model, dataset, device="cpu", batch_size=64)

    prediction = predict_all(features, dataset, scene="trento", seed=SEED)

    assert len(features) == len(dataset)
    assert len(prediction.query_indices) == len(dataset) - prediction.num_classes * K_SHOT
    assert 0.0 <= prediction.metrics()["OA"] <= 100.0

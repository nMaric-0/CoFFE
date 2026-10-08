"""One nearest-class-mean decision per labelled pixel, for a classification map.

PAPER_CANON §4's protocol reports a *distribution*: 5-shot supports are redrawn
per episode and each query pixel is classified many times, under many different
support draws. That is the right thing to average into an OA, and the wrong
thing to paint: a map needs exactly one decision per pixel.

This module runs the same classifier over the same features in the one shape
that yields a map — support drawn **once** at a seed, N = the scene's full
class count, and then *every remaining labelled sample* classified exactly
once:

    features = encode_dataset(model, dataset)          # coffe.eval.feature_cache
    prediction = predict_all(features, dataset, seed=42)
    canvas = prediction.to_maps(coords)                 # coffe.data.scene_coords

Nothing here is a second protocol. The prototypes and distances come from
``coffe.eval.hypersigma._compute_logits`` (imported lazily, so a CoFFE map does
not drag in the vendored HyperSIGMA stack) and the metrics from
``coffe.eval.episodic.compute_metrics_from_confusion_matrix`` — the repository's
one NCM and its one metric block, as ``coffe.eval.reduced_ncm`` also does — and
the support draw is
:class:`~coffe.data.samplers.patched_episode_sampler.PatchedEpisodeSampler`'s
own ``fixed_support`` code path at the given seed, not a second sampler.

**What the number here is and is not.** ``MapPrediction.metrics()`` is the OA of
*one* support draw over the whole labelled pool. It is not a paper cell: Table 2
and Table 3 average 1000-2000 episodes of 100 queries per class, and their ±
column is the spread across those episodes, which a single pass cannot have. Use
it to caption the map it belongs to, and expect it to sit inside the published
CI rather than to reproduce the mean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from coffe.data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from coffe.eval.episodic import compute_metrics_from_confusion_matrix

logger = logging.getLogger(__name__)

__all__ = ["MapPrediction", "predict_all"]

#: PAPER_CANON §4: the protocol's shot count and its cosine temperature.
K_SHOT = 5
TEMPERATURE = 10.0


@dataclass(frozen=True)
class MapPrediction:
    """Every labelled sample's single decision, in dataset-index space."""

    scene: str
    #: The original class labels the pass ran over, ascending. Position ``j``
    #: here is the local label ``j`` the classifier worked in.
    classes: np.ndarray
    #: Dataset indices used as support (excluded from the queries).
    support_indices: np.ndarray
    #: The original class each support sample belongs to.
    support_classes: np.ndarray
    #: Dataset indices classified, and their truth / prediction as *original*
    #: class labels.
    query_indices: np.ndarray
    query_classes: np.ndarray
    predictions: np.ndarray
    k_shot: int
    seed: int
    metric: str
    temperature: float

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def confusion(self) -> np.ndarray:
        """``[C, C]`` counts in local-label space, truth by row."""
        local = {int(c): j for j, c in enumerate(self.classes)}
        truth = np.array([local[int(c)] for c in self.query_classes])
        pred = np.array([local[int(c)] for c in self.predictions])
        n = self.num_classes
        return np.bincount(truth * n + pred, minlength=n * n).reshape(n, n).astype(np.int64)

    def metrics(self) -> dict[str, float]:
        """OA / AA / Kappa of this single pass (see the module docstring)."""
        return compute_metrics_from_confusion_matrix(self.confusion())

    def per_class_accuracy(self) -> dict[int, float]:
        """Recall per *original* class, in percent."""
        conf = self.confusion()
        totals = conf.sum(axis=1)
        return {
            int(c): float(conf[j, j] / totals[j] * 100) if totals[j] else 0.0
            for j, c in enumerate(self.classes)
        }

    def to_maps(self, coords: Any) -> dict[str, np.ndarray]:
        """Scatter the pass onto the scene, given a :class:`SceneCoords`.

        Returns four ``[H, W]`` canvases, all 0 where the scene is unlabelled:

        ``truth``
            the ground truth as the packs record it.
        ``prediction``
            the predicted class per query pixel; support pixels carry their
            own (known) class, since nothing was decided for them.
        ``support``
            1 at the ``C * k_shot`` support pixels, else 0.
        ``agreement``
            1 correct, 2 wrong, 3 support — the panel that shows *where* the
            errors are rather than how many.
        """
        truth = coords.to_map(coords.labels)
        prediction = coords.to_map(self.predictions, indices=self.query_indices)
        prediction[coords.rows[self.support_indices], coords.cols[self.support_indices]] = (
            self.support_classes
        )

        support = np.zeros(coords.shape, dtype=np.int16)
        support[coords.rows[self.support_indices], coords.cols[self.support_indices]] = 1

        agreement = np.zeros(coords.shape, dtype=np.int16)
        correct = self.predictions == self.query_classes
        agreement[coords.rows[self.query_indices], coords.cols[self.query_indices]] = np.where(
            correct, 1, 2
        )
        agreement[coords.rows[self.support_indices], coords.cols[self.support_indices]] = 3

        return {
            "truth": truth,
            "prediction": prediction,
            "support": support,
            "agreement": agreement,
        }

    def to_dict(self) -> dict[str, Any]:
        """The summary blob, for a results JSON next to the map."""
        return {
            "scene": self.scene,
            "k_shot": self.k_shot,
            "seed": self.seed,
            "metric": self.metric,
            "temperature": self.temperature,
            "classes": [int(c) for c in self.classes],
            "num_support": int(len(self.support_indices)),
            "num_queries": int(len(self.query_indices)),
            # Plain floats: the metric helper returns numpy scalars, which
            # json.dumps will not serialise.
            "metrics": {
                key: float(value) for key, value in self.metrics().items() if key != "per_class_acc"
            },
            "per_class_accuracy": self.per_class_accuracy(),
        }


@torch.no_grad()
def predict_all(
    features: torch.Tensor,
    dataset: Any,
    *,
    scene: str = "",
    k_shot: int = K_SHOT,
    seed: int = 42,
    metric: str = "euclidean",
    temperature: float = TEMPERATURE,
    device: str = "cpu",
) -> MapPrediction:
    """Classify every labelled sample once against one 5-shot support draw.

    Args:
        features: ``[n, D]`` frozen features, row ``i`` being ``dataset[i]``'s —
            i.e. what :func:`coffe.eval.feature_cache.encode_dataset` returns.
        dataset: the dataset those features were encoded from; only its
            ``class_indices`` are read.
        scene: dataset key, recorded on the result.
        k_shot: supports per class (PAPER_CANON §4: 5).
        seed: fixes the support draw.
        metric: ``"euclidean"`` (the paper's) or ``"cosine"``.
        temperature: only affects the cosine logits' scale, not its argmax.
        device: where to run the distance computation.

    Returns:
        A :class:`MapPrediction` over every class with at least ``k_shot + 1``
        samples — the same availability rule the episode sampler applies, at
        the smallest query count that leaves a query at all.
    """
    features = torch.as_tensor(features)
    if features.ndim != 2:
        raise ValueError(f"features must be [n, D]; got {tuple(features.shape)}")
    if len(features) != len(dataset):
        raise ValueError(
            f"{len(features)} features for {len(dataset)} samples — the cache and "
            f"the dataset must share an index space"
        )

    # The sampler's own fixed-support draw, so the map's supports come from the
    # code path the protocol uses rather than a second implementation. k_query=1
    # is the loosest availability filter that still leaves something to classify;
    # the sampler's episode drawing is never invoked.
    sampler = PatchedEpisodeSampler(
        dataset,
        n_way=None,
        k_shot=k_shot,
        k_query=1,
        num_episodes=1,
        seed=seed,
        fixed_support=True,
    )
    classes = np.array(sorted(sampler.available_classes), dtype=np.int64)
    # A class too small to give K supports and one query is not classified at
    # all, and its pixels would then be blank on a map that shows them in the
    # ground truth. No scene in the paper is anywhere near that, so say it out
    # loud if it ever happens.
    dropped = sorted(set(sampler.class_indices) - set(sampler.available_classes))
    if dropped:
        logger.warning(
            "[map] class(es) %s have fewer than %d samples: left unclassified",
            dropped,
            k_shot + 1,
        )

    support_indices, support_classes, query_indices, query_classes = [], [], [], []
    for original_class in classes:
        chosen = sampler.support_indices_by_class[original_class]
        support_indices.extend(int(i) for i in chosen)
        support_classes.extend([int(original_class)] * len(chosen))

        held = set(int(i) for i in chosen)
        rest = [int(i) for i in sampler.class_indices[original_class] if int(i) not in held]
        query_indices.extend(rest)
        query_classes.extend([int(original_class)] * len(rest))

    support_indices_arr = np.asarray(support_indices, dtype=np.int64)
    query_indices_arr = np.asarray(query_indices, dtype=np.int64)

    # Imported here, not at module scope: `coffe.eval.hypersigma` pulls in the
    # vendored HyperSIGMA stack (timm, einops, third_party), which a CoFFE map
    # has no use for. The import is still that module's `_compute_logits` --
    # there is one NCM in this repository and this is not a second one.
    from coffe.eval.hypersigma import _compute_logits

    z = features.to(device).float()
    s_feat = z.index_select(0, torch.as_tensor(support_indices_arr, device=device))
    q_feat = z.index_select(0, torch.as_tensor(query_indices_arr, device=device))
    # Local labels are class-major over `classes`, matching how the supports
    # were appended above.
    support_labels = torch.as_tensor(
        np.repeat(np.arange(len(classes)), k_shot), device=device, dtype=torch.long
    )

    logits = _compute_logits(s_feat, support_labels, q_feat, metric=metric, temperature=temperature)
    predictions = classes[logits.argmax(dim=1).cpu().numpy()]

    return MapPrediction(
        scene=scene,
        classes=classes,
        support_indices=support_indices_arr,
        support_classes=np.asarray(support_classes, dtype=np.int64),
        query_indices=query_indices_arr,
        query_classes=np.asarray(query_classes, dtype=np.int64),
        predictions=predictions.astype(np.int64),
        k_shot=k_shot,
        seed=seed,
        metric=metric,
        temperature=temperature,
    )

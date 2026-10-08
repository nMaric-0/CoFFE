"""One-pass feature cache for frozen-encoder episodic evaluation.

The episode loops in :mod:`coffe.eval.hypersigma` and :mod:`coffe.eval.episodic`
re-encode every support and query patch *inside* every episode. That is the
right shape while an encoder is training, but for a frozen one it is pure
waste: PAPER_CANON §4's protocol draws 2000 episodes of N x (5 + 100) patches,
so one Table 3 Houston run pushes 3.15M patches through a ViT-Base pair in
order to encode 15,029 distinct ones. With frozen weights in ``eval()`` mode
each patch has exactly one feature vector.

:func:`encode_dataset` walks the dataset once and returns that ``[n, D]``
matrix; :mod:`coffe.eval.reduced_ncm` then indexes it per episode. Against the
frozen runs' own logs the pass costs ~13 s where the live loop took 46 min
(Houston fused), ~18 s vs 12 min (Trento spat_pool), ~29 s vs 21 min (MUUFL
spat_pool).

**The composition is the contract.** The feature cached here is what the live
loop feeds to the prototypes, i.e. ``coffe.eval.hypersigma.evaluate``'s

    s_patch, s_cls, _ = model.forward_features(support_hsi, support_aux)
    s_feat = model.eval_patch_embeddings(s_patch, s_cls).mean(dim=1)

reproduced verbatim below. ``aux`` is passed even though the HyperSIGMA route
ignores it (HSI-only, PAPER_CANON §1), so a CoFFE-shaped encoder can use the
same path unchanged.

**Float caveat.** A cached feature is computed in a batch of ``batch_size``
rather than the live loop's per-episode batches of ``N*K`` and ``N*Q``. Neither
ViT branch carries batch-coupled state (LayerNorm is per-sample), so the
difference is float reduction order only — but it is not bit-exact, which is
why callers must gate on the full-dimension control reproducing the published
OA. ``scripts/experiments/run_hypersigma_dim_sweep.py`` gates at 0.05 pp.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: str = "cuda",
    batch_size: int = 256,
    log_interval: int = 20,
) -> torch.Tensor:
    """Encode every sample of ``dataset`` into its frozen eval feature.

    Args:
        model: a frozen evaluator exposing ``forward_features`` and
            ``eval_patch_embeddings`` (``HyperSIGMAFewShot``, ``CoFFE``, ...).
        dataset: pre-patched dataset indexed the way the episode sampler
            indexes it, so row ``i`` of the result is ``dataset[i]``'s feature.
        device: device to encode on; the returned matrix is always on CPU.
        batch_size: patches per forward pass.
        log_interval: log progress every this many batches.

    Returns:
        ``[len(dataset), D]`` float32 CPU tensor.
    """
    model.eval()
    n = len(dataset)  # type: ignore[arg-type]
    if n == 0:
        raise ValueError("cannot encode an empty dataset")

    chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    for batch_idx, start in enumerate(range(0, n, batch_size)):
        stop = min(start + batch_size, n)
        samples = [dataset[i] for i in range(start, stop)]  # type: ignore[index]
        hsi = torch.stack([s["hsi"] for s in samples]).to(device)
        aux = torch.stack([s["aux"] for s in samples]).to(device)

        # The live loop's composition, verbatim (see module docstring).
        patch_emb, cls_emb, _ = model.forward_features(hsi, aux)
        feats = model.eval_patch_embeddings(patch_emb, cls_emb).mean(dim=1)

        chunks.append(feats.float().cpu())
        if log_interval and batch_idx % log_interval == 0:
            done = stop
            rate = done / max(1e-9, time.perf_counter() - started)
            logger.info(
                "[cache] %d/%d patches (%.0f patches/s, dim=%d)",
                done,
                n,
                rate,
                chunks[-1].shape[-1],
            )

    features = torch.cat(chunks, dim=0)
    logger.info(
        "[cache] encoded %d patches -> [%d, %d] in %.1f s",
        n,
        *features.shape,
        time.perf_counter() - started,
    )
    return features


def save_features(path: str | Path, features: torch.Tensor, meta: dict[str, Any]) -> Path:
    """Write a feature cache plus the config that produced it.

    ``<path>.npz`` holds the matrix, ``<path>.json`` the metadata; the sidecar
    is what :func:`load_features` checks a cache against, so a cache can never
    be silently reused for a different encoder or geometry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    npz = path.with_suffix(".npz")
    np.savez(npz, features=features.numpy())
    record = {**meta, "shape": list(features.shape), "dtype": "float32"}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
    logger.info("[cache] saved %s (%.1f MB)", npz, npz.stat().st_size / 1e6)
    return npz


def load_features(
    path: str | Path,
    *,
    expect: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]] | None:
    """Load a cache written by :func:`save_features`, or ``None`` if unusable.

    Returns ``None`` (rather than raising) when the files are absent or the
    sidecar disagrees with ``expect`` on any key, so a driver can treat a stale
    cache as a cache miss and re-encode.
    """
    path = Path(path)
    npz, sidecar = path.with_suffix(".npz"), path.with_suffix(".json")
    if not (npz.exists() and sidecar.exists()):
        return None

    meta = json.loads(sidecar.read_text())
    for key, value in (expect or {}).items():
        if meta.get(key) != value:
            logger.warning(
                "[cache] %s stale: %s is %r, expected %r -> re-encoding",
                npz.name,
                key,
                meta.get(key),
                value,
            )
            return None

    with np.load(npz) as handle:
        features = torch.from_numpy(handle["features"])
    logger.info("[cache] reusing %s -> [%d, %d]", npz.name, *features.shape)
    return features, meta

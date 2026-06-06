"""Spatial PCA preprocessor for the HyperSIGMA SpatViT branch.

The SpatViT branch was pretrained on 3-channel RGB-like inputs. We
reduce the dataset's HSI bands to 3 components using PCA fitted on the
already-normalized patches (per-channel [0,1] from
``PatchedMultimodalDataset._normalize``). The PCA is wrapped as a
frozen 1x1 ``nn.Conv2d`` so it can sit inside the encoder graph and
autograd handles backprop correctly.

NOTE: There is **no spectral PCA** in this pipeline. The SpecViT
branch ingests raw bands and uses its built-in
``AdaptiveAvgPool1d(NUM_TOKENS)`` to convert arbitrary band counts into
a fixed number of spectral tokens.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# Derived from the central dataset registry so band counts can never
# drift from data/datasets/registry.py.
from data.datasets.registry import DATASET_REGISTRY  # noqa: E402

DATASET_PCA_CONFIG = {
    name: {"in_bands": spec.hsi_channels, "spat_components": spec.spat_components}
    for name, spec in DATASET_REGISTRY.items()
}


def fit_dataset_pca(
    dataset,
    n_components: int,
    save_path: Optional[str] = None,
) -> PCA:
    """Fit a PCA to the per-pixel spectral vectors of ``dataset``.

    Args:
        dataset: A ``PatchedMultimodalDataset`` (already normalized).
        n_components: Number of PCA components.
        save_path: Optional path to pickle the fitted PCA.

    Returns:
        The fitted sklearn ``PCA`` instance.
    """
    hsi = dataset.hsi  # [N, C, H, W], torch.FloatTensor in [0,1]
    if isinstance(hsi, torch.Tensor):
        hsi_np = hsi.detach().cpu().numpy()
    else:
        hsi_np = np.asarray(hsi)
    N, C, H, W = hsi_np.shape
    pixels = hsi_np.transpose(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(pixels)

    cum_var = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    logger.info(
        "[HyperSIGMA] PCA fit: %d -> %d, cumulative explained variance = %.4f",
        C, n_components, cum_var,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(pca, f)
        logger.info("[HyperSIGMA] PCA saved to %s", save_path)

    return pca


def load_pca(path: str) -> PCA:
    with open(path, "rb") as f:
        return pickle.load(f)


def cumulative_explained_variance(pca: PCA) -> float:
    return float(np.cumsum(pca.explained_variance_ratio_)[-1])


def fit_pca_output_stats(
    dataset,
    pca: PCA,
    save_path: Optional[str] = None,
) -> dict:
    """Compute per-channel mean/std of ``pca(dataset.hsi)``.

    These are the statistics needed to standardize the PCA output so the
    SpatViT branch sees inputs with mean=0/std=1 per channel. The result
    is a dict ``{"mean": [k], "std": [k], "n_pixels": int}`` saved as a
    pickle alongside the PCA itself.
    """
    hsi = dataset.hsi
    if isinstance(hsi, torch.Tensor):
        hsi_np = hsi.detach().cpu().numpy()
    else:
        hsi_np = np.asarray(hsi)
    N, C, H, W = hsi_np.shape
    pixels = hsi_np.transpose(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]

    # Apply the PCA in numpy so we don't have to materialize a torch module here.
    reduced = (pixels - pca.mean_) @ pca.components_.T  # [N*H*W, k]
    mean = reduced.mean(axis=0).astype(np.float32)
    std = reduced.std(axis=0).astype(np.float32)
    # Guard against zero variance (degenerate component).
    std = np.where(std < 1e-8, np.float32(1.0), std)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_pixels": int(reduced.shape[0]),
    }
    logger.info(
        "[HyperSIGMA] PCA output stats over %d pixels: mean=%s, std=%s",
        stats["n_pixels"], np.array2string(mean, precision=4),
        np.array2string(std, precision=4),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(stats, f)
        logger.info("[HyperSIGMA] PCA output stats saved to %s", save_path)

    return stats


def load_pca_output_stats(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


class PCAStandardize(nn.Module):
    """Per-channel standardization: ``(x - mean) / std``.

    Mean and std are buffers (move with ``.to(device)``, saved in
    state_dict). Use with the PCA output to give SpatViT inputs with
    mean=0/std=1 per channel — the cleanest distribution match for the
    HyperSIGMA pretraining target.
    """

    def __init__(self, mean, std):
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std_t = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        if (std_t < 1e-8).any():
            raise ValueError(f"PCAStandardize received a near-zero std: {std_t.flatten().tolist()}")
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)
        self.num_channels = mean_t.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, "
            f"mean={self.mean.flatten().tolist()}, "
            f"std={self.std.flatten().tolist()}"
        )


class PCAPreprocessor(nn.Module):
    """Frozen 1x1 Conv2d wrapping a fitted sklearn PCA.

    Maps ``[B, C_in, H, W] -> [B, n_components, H, W]`` via
    ``out_c = sum_c W[out_c, c] * (in_c - mean_c)``. The weight is the
    PCA components (shape ``[n_components, C_in]``); the bias absorbs
    ``-W @ mean``. All parameters are registered as buffers so they are
    moved with ``.to(device)`` and saved with ``state_dict()``, but they
    are not trainable.
    """

    def __init__(self, pca: PCA):
        super().__init__()
        components = pca.components_.astype(np.float32)  # [k, C]
        mean = pca.mean_.astype(np.float32)              # [C]
        k, c = components.shape
        # Conv2d weight shape: [k, C, 1, 1]
        weight = torch.from_numpy(components).view(k, c, 1, 1)
        bias = torch.from_numpy(-components @ mean)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.in_channels = c
        self.out_channels = k
        self.cumulative_explained_variance = cumulative_explained_variance(pca)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, self.weight, self.bias)

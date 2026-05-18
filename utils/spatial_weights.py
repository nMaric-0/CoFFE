"""
Center-weighted spatial kernels for patch-based HSI classification.

Motivated by the label-context mismatch: only the center pixel of each
patch carries the ground-truth label, but uniform pooling treats all
pixels equally, contaminating class prototypes at boundaries.

References:
- SQSFormer (Chen et al., IEEE TGRS 2024): center pixel as query
- CPFENet (Int. J. Remote Sensing, 2025): center pixel enhancement
- PixMIM (Liu et al., 2023): weighted reconstruction targets in MAE
"""

import torch


def make_center_weights(
    patch_size: int = 11,
    sigma: float = 2.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Create a 2D Gaussian weight map centered on the middle pixel,
    flattened to [patch_size * patch_size].

    Args:
        patch_size: spatial dimension of the patch (assumed square)
        sigma: std dev of the Gaussian; larger = more uniform
               sigma=2.0 is a good default for prototype pooling
               sigma=3.0 is a good default for reconstruction loss
        normalize: if True, weights sum to 1 (for pooling)
                   if False, weights are unnormalized (for loss weighting)

    Returns:
        weights: [patch_size^2] tensor
    """
    center = patch_size // 2
    coords = torch.arange(patch_size, dtype=torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    weights = torch.exp(-dist_sq / (2 * sigma ** 2))

    if normalize:
        weights = weights / weights.sum()

    return weights.flatten()  # [patch_size^2]


def center_weighted_pool(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted spatial pooling over patch token embeddings.

    Args:
        embeddings: [B, N, D] patch embeddings (N = patch_size^2)
        weights: [N] spatial weights (should sum to 1)

    Returns:
        pooled: [B, D]
    """
    # weights: [N] -> [1, N, 1] for broadcasting
    w = weights.to(embeddings.device).unsqueeze(0).unsqueeze(-1)
    return (embeddings * w).sum(dim=1)

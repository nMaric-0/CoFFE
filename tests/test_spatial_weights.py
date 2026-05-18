"""Tests for center-weighted spatial kernels and pooling."""

import torch
import pytest

from utils.spatial_weights import make_center_weights, center_weighted_pool


class TestMakeCenterWeights:
    """Tests for the Gaussian weight map generation."""

    def test_normalized_weights_sum_to_one(self):
        w = make_center_weights(patch_size=11, sigma=2.0, normalize=True)
        assert w.shape == (121,)
        assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-6)

    def test_unnormalized_weights_do_not_sum_to_one(self):
        w = make_center_weights(patch_size=11, sigma=2.0, normalize=False)
        assert w.sum() != pytest.approx(1.0, abs=1e-3)

    def test_center_pixel_has_max_weight(self):
        w = make_center_weights(patch_size=11, sigma=2.0, normalize=True)
        center_idx = 11 * 5 + 5  # (5, 5) in 11x11 grid, 0-indexed
        assert w[center_idx] == w.max()

    def test_large_sigma_approximates_uniform(self):
        w = make_center_weights(patch_size=11, sigma=1000.0, normalize=True)
        uniform = torch.ones(121) / 121.0
        assert torch.allclose(w, uniform, atol=1e-5)

    def test_small_sigma_concentrates_on_center(self):
        w = make_center_weights(patch_size=11, sigma=0.01, normalize=True)
        center_idx = 11 * 5 + 5
        # Center pixel should hold almost all weight
        assert w[center_idx].item() > 0.99

    def test_different_patch_sizes(self):
        for ps in [3, 5, 7, 11, 15]:
            w = make_center_weights(patch_size=ps, sigma=2.0, normalize=True)
            assert w.shape == (ps * ps,)
            assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-6)

    def test_weights_are_symmetric(self):
        w = make_center_weights(patch_size=11, sigma=2.0, normalize=True)
        w_2d = w.reshape(11, 11)
        # Horizontal symmetry
        assert torch.allclose(w_2d, w_2d.flip(1), atol=1e-7)
        # Vertical symmetry
        assert torch.allclose(w_2d, w_2d.flip(0), atol=1e-7)


class TestCenterWeightedPool:
    """Tests for the weighted spatial pooling function."""

    def test_output_shape(self):
        embeddings = torch.randn(4, 121, 64)
        weights = make_center_weights(patch_size=11, sigma=2.0, normalize=True)
        pooled = center_weighted_pool(embeddings, weights)
        assert pooled.shape == (4, 64)

    def test_uniform_weights_match_mean(self):
        embeddings = torch.randn(8, 121, 128)
        uniform_weights = torch.ones(121) / 121.0
        pooled = center_weighted_pool(embeddings, uniform_weights)
        mean_pooled = embeddings.mean(dim=1)
        assert torch.allclose(pooled, mean_pooled, atol=1e-5)

    def test_single_pixel_weight(self):
        """With weight concentrated on one pixel, output should be that pixel's embedding."""
        B, N, D = 2, 121, 32
        embeddings = torch.randn(B, N, D)
        weights = torch.zeros(N)
        center_idx = 60  # center of 11x11
        weights[center_idx] = 1.0
        pooled = center_weighted_pool(embeddings, weights)
        assert torch.allclose(pooled, embeddings[:, center_idx, :], atol=1e-6)

    def test_device_handling(self):
        """Weights are moved to embedding device automatically."""
        embeddings = torch.randn(2, 121, 64)  # CPU
        weights = torch.ones(121) / 121.0  # CPU
        pooled = center_weighted_pool(embeddings, weights)
        assert pooled.device == embeddings.device

    def test_gradient_flows(self):
        embeddings = torch.randn(2, 121, 64, requires_grad=True)
        weights = make_center_weights(patch_size=11, sigma=2.0, normalize=True)
        pooled = center_weighted_pool(embeddings, weights)
        loss = pooled.sum()
        loss.backward()
        assert embeddings.grad is not None
        assert embeddings.grad.shape == embeddings.shape

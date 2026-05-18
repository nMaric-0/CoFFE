"""
Tests for the unified masked pretraining model.

Verifies:
- UnifiedBandMasking masks roughly the requested fraction of (pixel, band) entries
- EnhancedMaskedSpectralSpatialModel supports band-only, spatial-only, and
  combined masking; each forward pass produces the expected shapes
  (one token per pixel carrying both HSI and aux bands)
- Backward pass populates gradients on all trainable parameters
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from models import MFTCPEACosine
from pretrain.masked_modeling import UnifiedBandMasking
from pretrain.masked_modeling_enhanced import EnhancedMaskedSpectralSpatialModel


def _make_encoder(embed_dim: int = 64):
    return MFTCPEACosine(
        hsi_channels=144,
        aux_channels=1,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        patch_size=11,
        lambda_factor=2.0,
        dropout=0.1,
        use_projection=True,
        proj_hidden_dim=256,
        proj_num_layers=2,
        proj_l2_normalize=True,
    )


def _run_and_check(pretrain_model, batch_size: int = 4):
    hsi = torch.randn(batch_size, 144, 11, 11)
    aux = torch.randn(batch_size, 1, 11, 11)

    pretrain_model.train()
    loss, info = pretrain_model(hsi, aux)

    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert torch.isfinite(loss)

    assert info["pred"].shape == (batch_size, 121, 145)
    assert info["target"].shape == (batch_size, 121, 145)
    assert info["mask"].shape == (batch_size, 121, 145)
    assert info["band_recon"] is loss

    loss.backward()

    missing = [
        n for n, p in pretrain_model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, f"Missing gradients: {missing}"

    return loss, info


def test_unified_band_masking():
    """UnifiedBandMasking should hide ~mask_ratio of (pixel, band) entries."""
    print("Testing UnifiedBandMasking...")

    masking = UnifiedBandMasking(num_channels=145, mask_ratio=0.9)

    x = torch.randn(4, 145, 11, 11)
    x_masked, mask = masking(x)

    assert x_masked.shape == x.shape
    assert mask.shape == x.shape

    ratio = mask.mean().item()
    print(f"  Mask ratio: {ratio:.3f} (target: 0.900)")
    assert 0.85 < ratio < 0.95, f"Unexpected mask ratio {ratio}"

    # Masked entries should equal the learnable fill value, visible entries
    # should equal the original input.
    fill = masking.mask_value.detach().expand_as(x)
    diff_masked = (x_masked - fill)[mask.bool()].abs().max().item()
    diff_visible = (x_masked - x)[(1 - mask).bool()].abs().max().item()
    assert diff_masked < 1e-6, f"Masked entries were not replaced ({diff_masked})"
    assert diff_visible < 1e-6, f"Visible entries were altered ({diff_visible})"

    print("  ✓ UnifiedBandMasking passed")


def test_enhanced_model_band_only():
    """Band-only masking path (default configuration)."""
    print("Testing band-only masking...")

    encoder = _make_encoder()
    pretrain_model = EnhancedMaskedSpectralSpatialModel(
        encoder=encoder,
        hsi_channels=144,
        aux_channels=1,
        patch_size=11,
        embed_dim=64,
        decoder_hidden_dim=128,
        band_mask_ratio=0.9,
        spatial_mask_ratio=0.0,
    )

    loss, info = _run_and_check(pretrain_model)
    # Encoder token count: 1 (CLS) + 121 (patches), not 1 + 2*121 as before.
    assert encoder.pos_embed.shape == (1, 1 + 121, 64)
    # Every (pixel, band) entry has an independent Bernoulli sample, so mask
    # coverage should be around 0.9.
    ratio = info["mask"].mean().item()
    assert 0.85 < ratio < 0.95, f"Unexpected mask ratio {ratio}"
    print(f"  Loss: {loss.item():.4f}, mask ratio: {ratio:.3f}")
    print("  ✓ Band-only masking passed")


def test_enhanced_model_spatial_only():
    """Spatial (MAE-style) masking only — no per-band masking."""
    print("Testing spatial-only masking...")

    encoder = _make_encoder()
    pretrain_model = EnhancedMaskedSpectralSpatialModel(
        encoder=encoder,
        hsi_channels=144,
        aux_channels=1,
        patch_size=11,
        embed_dim=64,
        decoder_hidden_dim=128,
        band_mask_ratio=0.0,
        spatial_mask_ratio=0.75,
    )

    loss, info = _run_and_check(pretrain_model)
    # Spatial masking masks whole tokens — each row of info["mask"] is either
    # all-0 or all-1 across the 145 bands.
    row_sums = info["mask"].sum(dim=-1)
    bands = info["mask"].shape[-1]
    is_all_or_none = (row_sums == 0) | (row_sums == bands)
    assert bool(is_all_or_none.all()), "Spatial mask should be per-token all-or-nothing"
    # 0.75 * 121 = 90.75 → exactly 91 tokens per batch element are masked.
    per_batch_tokens_masked = (row_sums > 0).sum(dim=-1)
    assert (per_batch_tokens_masked == 91).all(), per_batch_tokens_masked
    print(f"  Loss: {loss.item():.4f}, masked tokens/batch: 91/121")
    print("  ✓ Spatial-only masking passed")


def test_enhanced_model_combined():
    """Both masks active — loss mask should be union of band and spatial."""
    print("Testing combined masking...")

    encoder = _make_encoder()
    pretrain_model = EnhancedMaskedSpectralSpatialModel(
        encoder=encoder,
        hsi_channels=144,
        aux_channels=1,
        patch_size=11,
        embed_dim=64,
        decoder_hidden_dim=128,
        band_mask_ratio=0.5,
        spatial_mask_ratio=0.5,
    )

    loss, info = _run_and_check(pretrain_model)
    # Combined coverage must exceed band-only (0.5) in expectation.
    ratio = info["mask"].mean().item()
    assert ratio > 0.55, f"Combined mask coverage too low: {ratio}"
    print(f"  Loss: {loss.item():.4f}, combined mask ratio: {ratio:.3f}")
    print("  ✓ Combined masking passed")


def main():
    print("=" * 60)
    print("Testing unified masked pretraining")
    print("=" * 60)

    test_unified_band_masking()
    print()
    test_enhanced_model_band_only()
    print()
    test_enhanced_model_spatial_only()
    print()
    test_enhanced_model_combined()
    print()

    print("=" * 60)
    print("All tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()

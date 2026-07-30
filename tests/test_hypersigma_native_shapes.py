"""Shape + clean-load tests for the native-geometry HyperSIGMA ablation.

The native-geometry ablation keeps each encoder at its pretrained input
size (SpatViT 64x64/patch-8/100ch, SpecViT 64x64) and resizes the 11x11
patch up to it, instead of shrinking the encoder. These tests:

* exercise the construction + forward-shape contract WITHOUT the released
  checkpoints (random body, spectral-resample 144->100 front-end), for
  both ``input_fit`` strategies and both branches;
* a checkpoint-gated test (skipped if the ``.pth`` are absent) confirming
  ``patch_embed.proj`` / ``spat_map`` / ``pos_embed`` actually LOAD.

Run with::

    pytest tests/test_hypersigma_native_shapes.py -q
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.decomposition import PCA

from models.hypersigma import HyperSIGMACosine, HyperSIGMADual
from models.hypersigma._input_fit import fit_input
from models.hypersigma.preprocessing import SpectralResample

SPAT_CKPT = Path("checkpoints/hypersigma/spat-vit-base.pth")
SPEC_CKPT = Path("checkpoints/hypersigma/spec-vit-base.pth")


def _make_dummy_pca(tmp_path: Path, in_bands: int, n_components: int) -> str:
    rng = np.random.default_rng(0)
    pixels = rng.standard_normal(size=(2048, in_bands)).astype(np.float32)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(pixels)
    out = tmp_path / f"pca_{in_bands}_{n_components}.pkl"
    with open(out, "wb") as f:
        pickle.dump(pca, f)
    return str(out)


# ----------------------------------------------------------------------
# Pure helper / front-end shape contracts
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fit", ["upscale", "pad"])
def test_fit_input_shapes(fit):
    x = torch.randn(2, 7, 11, 11)
    out = fit_input(x, 64, fit)
    assert out.shape == (2, 7, 64, 64), out.shape


def test_fit_input_pad_centered_zero():
    x = torch.ones(1, 1, 11, 11)
    out = fit_input(x, 64, "pad", pad_anchor="center")
    # The 11x11 block of ones sits centered; corners are zero-padded.
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 32, 32] == 1.0
    assert out[..., :].sum() == 11 * 11


def test_fit_input_pad_reflect_rejected():
    # Reflect is impossible at 11->64; helper only does constant padding.
    with pytest.raises(ValueError):
        fit_input(torch.randn(1, 1, 11, 11), 64, "nope")


def test_spectral_resample_to_100():
    for bands in (63, 64, 144):
        x = torch.randn(2, bands, 11, 11)
        out = SpectralResample(100)(x)
        assert out.shape == (2, 100, 11, 11), (bands, out.shape)
    # No-op when already at target.
    x = torch.randn(2, 100, 5, 5)
    assert SpectralResample(100)(x).shape == (2, 100, 5, 5)


# ----------------------------------------------------------------------
# Checkpoint-free native forward shapes (random body, resample front-end)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fit", ["upscale", "pad"])
@pytest.mark.parametrize("bands", [144, 63, 64])
def test_native_spatial_forward(fit, bands):
    dual = HyperSIGMADual(
        pca_spat_path=None,        # native spatial uses native_pca_spat_path / resample
        spat_ckpt=None,            # random body; skips the clean-load assert
        spec_ckpt=None,
        hsi_channels=bands,
        native_geometry=True,
        input_fit=fit,
        native_pca_spat_path=None,  # -> SpectralResample(100)
        build_spat=True,
        build_spec=False,
        build_sem=False,
    )
    assert isinstance(dual.pca_spat, SpectralResample)
    assert dual.spat.in_chans == 100
    model = HyperSIGMACosine(dual=dual, mode="spat_pool", distance_metric="cosine")
    x = torch.randn(2, bands, 11, 11)
    with torch.no_grad():
        spat_features = dual.spat(dual.pca_spat(x))
        patch, cls, _ = model.forward_features(x, None)
    # patch_size=8 over a 64x64 input -> 8x8 grid; fpn4 = MaxPool(4,4) -> 2x2.
    assert spat_features[-1].shape == (2, 768, 2, 2), spat_features[-1].shape
    assert patch.shape == (2, 1, 768)
    norms = torch.linalg.vector_norm(cls, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


@pytest.mark.parametrize("fit", ["upscale", "pad"])
@pytest.mark.parametrize("bands", [144, 63, 64])
def test_native_spectral_forward(fit, bands):
    dual = HyperSIGMADual(
        pca_spat_path=None,
        spat_ckpt=None,
        spec_ckpt=None,
        hsi_channels=bands,
        native_geometry=True,
        input_fit=fit,
        build_spat=False,
        build_spec=True,
        build_sem=False,
    )
    assert dual.spec.img_size == 64
    model = HyperSIGMACosine(dual=dual, mode="spec_pool", distance_metric="cosine")
    x = torch.randn(2, bands, 11, 11)
    with torch.no_grad():
        spec_features = dual.spec(x)
        patch, cls, _ = model.forward_features(x, None)
    assert spec_features[-1].shape == (2, 100, 768), spec_features[-1].shape
    assert patch.shape == (2, 1, 768)


def test_native_pca_front_end_selected(tmp_path):
    pca_path = _make_dummy_pca(tmp_path, in_bands=144, n_components=100)
    dual = HyperSIGMADual(
        pca_spat_path=None, spat_ckpt=None, spec_ckpt=None,
        hsi_channels=144, native_geometry=True,
        native_pca_spat_path=pca_path,
        build_spat=True, build_spec=False, build_sem=False,
    )
    from models.hypersigma.preprocessing import PCAPreprocessor
    assert isinstance(dual.pca_spat, PCAPreprocessor)
    assert dual.pca_spat.out_channels == 100


# ----------------------------------------------------------------------
# Checkpoint-gated: confirm the pretrained projections actually load
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Adapted-geometry PCA-100 variant: always 100 spatial channels
# (PCA where bands >= 100, spectral resample up to 100 otherwise)
# ----------------------------------------------------------------------


def test_adapted_pca100_resample_when_under_100_bands(tmp_path):
    # Trento-like: 63 bands, adapted geometry, target 100ch, no PCA pickle -> resample.
    missing = str(tmp_path / "pca_trento_100band.pkl")  # never created
    dual = HyperSIGMADual(
        pca_spat_path=missing, spat_ckpt=None, spec_ckpt=None, hsi_channels=63,
        spat_patch_k=3, spat_resample_to=100,
        build_spat=True, build_spec=False, build_sem=False,
    )
    assert isinstance(dual.pca_spat, SpectralResample)
    assert dual.spat.in_chans == 100
    x = torch.randn(2, 63, 11, 11)
    with torch.no_grad():
        feats = dual.spat(dual.pca_spat(x))
    # Adapted geometry (patch-3, pad 11->12) -> 4x4 token grid.
    assert feats[-1].shape == (2, 768, 4, 4), feats[-1].shape


def test_adapted_pca100_uses_pca_when_available(tmp_path):
    from models.hypersigma.preprocessing import PCAPreprocessor
    pca_path = _make_dummy_pca(tmp_path, in_bands=144, n_components=100)
    dual = HyperSIGMADual(
        pca_spat_path=pca_path, spat_ckpt=None, spec_ckpt=None, hsi_channels=144,
        spat_patch_k=3, spat_resample_to=100,
        build_spat=True, build_spec=False, build_sem=False,
    )
    assert isinstance(dual.pca_spat, PCAPreprocessor)
    assert dual.pca_spat.out_channels == 100
    assert dual.spat.in_chans == 100


# ----------------------------------------------------------------------
# Ablation 2: native-geometry MAE SEM-tuning (sem_only adapt mode)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fit", ["upscale", "pad"])
def test_native_sem_only_adaptation(fit):
    from pretrain.hypersigma_mae import HyperSIGMAMaskedAdaptation

    # Native FULL dual (both branches + SEM), random body, resample 144->100.
    dual = HyperSIGMADual(
        pca_spat_path=None, spat_ckpt=None, spec_ckpt=None,
        hsi_channels=144, native_geometry=True, input_fit=fit,
        native_pca_spat_path=None,            # -> SpectralResample(100)
        build_spat=True, build_spec=True, build_sem=True,
    )
    model = HyperSIGMAMaskedAdaptation(
        dual=dual, adapt_mode="sem_only", hsi_channels=144, patch_size=11, mask_ratio=0.75,
    )
    # Native spatial patch grid is 8x8 = 64 tokens (not the pad_to//k = 1 of the old formula).
    assert model.num_spat_tokens == 64, model.num_spat_tokens
    assert model.num_spec_tokens == 100

    x = torch.randn(2, 144, 11, 11)
    out = model(x)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"]), out["loss"]
    # sem_only uses only the fused loss.
    assert out["pred_fused"].shape == (2, 100, 11, 11), out["pred_fused"].shape

    # Only the fusion path trains: SEM + fused_decoder + spectral l1 + mask tokens.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    allowed = ("dual.sem.", "fused_decoder.", "dual.spec.model.l1.",
               "spat_mask_token", "spec_mask_token")
    bad = [n for n in trainable if not any(n.startswith(p) for p in allowed)]
    assert not bad, f"unexpected trainable params: {bad[:10]}"
    # The pretrained encoder bodies + input projections are frozen.
    assert not any(n.startswith("dual.spat.model.blocks.") for n in trainable)
    assert not any(n.startswith("dual.spec.model.blocks.") for n in trainable)
    assert not any(n.startswith("dual.spat.model.patch_embed.") for n in trainable)
    assert not any(n.startswith("dual.spec.model.spat_map.") for n in trainable)
    # SEM and l1 are actually present + trainable (sanity).
    assert any(n.startswith("dual.sem.") for n in trainable)
    assert any(n.startswith("dual.spec.model.l1.") for n in trainable)


def test_native_fused_forward_runs():
    # The dual must build + forward fused at native geometry (guard relaxed).
    dual = HyperSIGMADual(
        pca_spat_path=None, spat_ckpt=None, spec_ckpt=None,
        hsi_channels=144, native_geometry=True, input_fit="upscale",
        native_pca_spat_path=None,
        build_spat=True, build_spec=True, build_sem=True,
    )
    model = HyperSIGMACosine(dual=dual, mode="fused", distance_metric="cosine")
    x = torch.randn(2, 144, 11, 11)
    with torch.no_grad():
        patch, cls, _ = model.forward_features(x, None)
    assert patch.shape == (2, 1, 512), patch.shape  # SEM fused = num_stages*dr_dim = 512


@pytest.mark.skipif(not SPAT_CKPT.exists(), reason="spat-vit-base.pth not present")
def test_native_spatial_loads_pretrained():
    dual = HyperSIGMADual(
        pca_spat_path=None, spat_ckpt=str(SPAT_CKPT), spec_ckpt=None,
        hsi_channels=144, native_geometry=True, input_fit="upscale",
        native_pca_spat_path=None,  # resample -> 100ch matches the 100ch patch_embed
        build_spat=True, build_spec=False, build_sem=False,
    )
    assert dual.spat.pos_embed_source == "loaded"
    assert dual.spat.patch_embed_source == "loaded"
    assert tuple(dual.spat.model.patch_embed.proj.weight.shape) == (768, 100, 8, 8)
    assert tuple(dual.spat.model.pos_embed.shape) == (1, 64, 768)


@pytest.mark.skipif(not SPEC_CKPT.exists(), reason="spec-vit-base.pth not present")
def test_native_spectral_loads_pretrained():
    dual = HyperSIGMADual(
        pca_spat_path=None, spat_ckpt=None, spec_ckpt=str(SPEC_CKPT),
        hsi_channels=144, native_geometry=True, input_fit="upscale",
        build_spat=False, build_spec=True, build_sem=False,
    )
    assert dual.spec.pos_embed_source == "loaded"
    assert dual.spec.spat_map_source == "loaded"
    assert tuple(dual.spec.model.spat_map.weight.shape) == (768, 4096)
    assert tuple(dual.spec.model.pos_embed.shape) == (1, 100, 768)

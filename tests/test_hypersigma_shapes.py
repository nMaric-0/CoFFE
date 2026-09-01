"""Shape smoke tests for the HyperSIGMA wrappers.

These tests exercise the construction + forward-shape contract of the
dual encoder without requiring the released SpatViT/SpecViT checkpoints
or the Houston dataset. They synthesize a tiny PCA on-the-fly and run a
single dummy batch through ``HyperSIGMADual`` / ``HyperSIGMAFewShot``.

Run with::

    pytest tests/test_hypersigma_shapes.py -q
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.decomposition import PCA

from models.hypersigma import (
    HyperSIGMAFewShot,
    HyperSIGMADual,
    SEM,
)


def _make_dummy_pca(tmp_path: Path, in_bands: int = 144, n_components: int = 3) -> str:
    rng = np.random.default_rng(0)
    pixels = rng.standard_normal(size=(2048, in_bands)).astype(np.float32)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(pixels)
    out = tmp_path / "pca.pkl"
    with open(out, "wb") as f:
        pickle.dump(pca, f)
    return str(out)


def test_sem_shape():
    sem = SEM(spat_dim=768, num_tokens=100, dr_dim=128, num_stages=4)
    spat_feats = [torch.randn(2, 768, 4, 4) for _ in range(4)]
    spec_pooled = torch.randn(2, 100)
    out = sem(spat_feats, spec_pooled)
    assert out.shape == (2, 512), out.shape


def test_dual_forward_shapes_random_init(tmp_path):
    pca_path = _make_dummy_pca(tmp_path)
    dual = HyperSIGMADual(
        pca_spat_path=pca_path,
        spat_ckpt=None,   # skip checkpoint loading; transformer body random
        spec_ckpt=None,
        hsi_channels=144,
        spat_patch_k=3,
        freeze_body=True,
    )
    dual.eval()
    x = torch.randn(2, 144, 11, 11)
    with torch.no_grad():
        out = dual(x)
    assert out["fused"].shape == (2, 512)
    assert len(out["spat_features"]) == 4
    assert all(f.shape == (2, 768, 4, 4) for f in out["spat_features"])
    assert out["spec_first"].shape == (2, 100, 128)
    assert out["spec_features"][-1].shape == (2, 100, 768)


def test_fewshot_wrapper_modes(tmp_path):
    pca_path = _make_dummy_pca(tmp_path)
    dual = HyperSIGMADual(
        pca_spat_path=pca_path,
        spat_ckpt=None,
        spec_ckpt=None,
        hsi_channels=144,
        spat_patch_k=3,
        freeze_body=True,
    )
    x = torch.randn(3, 144, 11, 11)
    for mode in HyperSIGMAFewShot.SUPPORTED_MODES:
        model = HyperSIGMAFewShot(dual=dual, mode=mode, distance_metric="cosine")
        with torch.no_grad():
            patch, cls, aux = model.forward_features(x, None)
        assert patch.shape[0] == 3 and patch.shape[1] == 1
        assert patch.shape[2] in {128 * 4, 768}
        assert cls.shape == (3, patch.shape[2])
        # L2 norm sanity check.
        norms = torch.linalg.vector_norm(cls, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_trainable_components_present(tmp_path):
    pca_path = _make_dummy_pca(tmp_path)
    dual = HyperSIGMADual(
        pca_spat_path=pca_path,
        spat_ckpt=None,
        spec_ckpt=None,
        hsi_channels=144,
        spat_patch_k=3,
        freeze_body=True,
    )
    trainable = {n for n, p in dual.named_parameters() if p.requires_grad}
    # Spatial side.
    assert any(n.startswith("spat.model.patch_embed.proj.") for n in trainable)
    assert "spat.model.pos_embed" in trainable
    # Spectral side.
    assert any(n.startswith("spec.model.spat_map.") for n in trainable)
    assert "spec.model.pos_embed" in trainable
    # SEM.
    assert any(n.startswith("sem.dr.") for n in trainable)
    assert any(n.startswith("sem.fc_spec.") for n in trainable)
    # Transformer bodies are frozen.
    assert not any(n.startswith("spat.model.blocks.") for n in trainable)
    assert not any(n.startswith("spec.model.blocks.") for n in trainable)

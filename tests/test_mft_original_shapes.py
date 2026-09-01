"""Shape + MAE-gradient tests for the original-MFT baseline.

The original-MFT baseline (``models.mft_original.MFTOriginal``) is the
faithful MFT — Conv3D+HetConv HSI front-end -> 121 spatial tokens, LiDAR-derived
external CLS via channel tokenization, mCrossPA fusion — pretrained with standard
MAE (``pretrain.mft_mae.MFTMAEPretrainModel``). These checkpoint-free tests
exercise:

* the tokenization / forward-feature shape contract across all three datasets'
  band counts (144/63/64 HSI, 1/1/2 aux);
* the few-shot eval contract (single-token CLS packing, no class-token folding);
* the standard-MAE forward (scalar loss, ``len_keep`` from ``mask_ratio``);
* the key consequence of combining mCrossPA with MAE — that the decoder's
  cross-attention to the encoded CLS actually routes a reconstruction gradient
  into the mCrossPA attention projections (otherwise the fusion transformer
  would never train).

Run with::

    pytest tests/test_mft_original_shapes.py -q
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.mft_original import MFTOriginal
from pretrain.mft_mae import MFTMAEPretrainModel
from pretrain.mft_spatial_mae import MFTSpatialMaskPretrainModel


# Faithful MFT dims (Roy et al. / srinadh99): FM=16 -> dim=64, 8 heads, depth 2,
# mlp_dim 512 (a fixed 512). head_dim = 64/8 = 8.
DIMS = dict(embed_dim=64, num_heads=8, num_layers=2, mlp_dim=512)
DATASETS = [("houston", 144, 1), ("trento", 63, 1), ("muufl", 64, 2)]


def _encoder(hsi_channels, aux_channels):
    return MFTOriginal(
        hsi_channels=hsi_channels,
        aux_channels=aux_channels,
        use_aux=True,
        patch_size=11,
        attention_type="mcross",
        **DIMS,
    )


@pytest.mark.parametrize("name,hsi_c,aux_c", DATASETS)
def test_tokenize_and_cls_shapes(name, hsi_c, aux_c):
    model = _encoder(hsi_c, aux_c)
    hsi = torch.randn(2, hsi_c, 11, 11)
    aux = torch.randn(2, aux_c, 11, 11)
    with torch.no_grad():
        hsi_tokens = model.tokenize(hsi)
        cls = model.make_cls(aux)
    assert hsi_tokens.shape == (2, 121, DIMS["embed_dim"]), hsi_tokens.shape
    assert cls.shape == (2, 1, DIMS["embed_dim"]), cls.shape


@pytest.mark.parametrize("name,hsi_c,aux_c", DATASETS)
def test_forward_features_contract(name, hsi_c, aux_c):
    model = _encoder(hsi_c, aux_c)
    hsi = torch.randn(3, hsi_c, 11, 11)
    aux = torch.randn(3, aux_c, 11, 11)
    with torch.no_grad():
        patch, cls, aux_emb = model.forward_features(hsi, aux)
    # CLS packed as a single patch token (mean over dim=1 recovers it in eval).
    assert patch.shape == (3, 1, DIMS["embed_dim"]), patch.shape
    assert cls.shape == (3, DIMS["embed_dim"]), cls.shape
    assert torch.allclose(patch.mean(dim=1), cls)
    assert aux_emb is cls or torch.allclose(aux_emb, cls)
    # No class-token folding: eval_patch_embeddings is a no-op pass-through.
    assert torch.allclose(model.eval_patch_embeddings(patch, cls), patch)


def test_hsi_only_rejected():
    with pytest.raises(ValueError):
        _encoder(144, 1).__class__(hsi_channels=144, aux_channels=1, use_aux=False, **DIMS)


def test_mae_forward_loss_and_len_keep():
    model = _encoder(144, 1)
    mae = MFTMAEPretrainModel(
        encoder=model, hsi_channels=144, aux_channels=1, use_aux=True,
        patch_size=11, embed_dim=DIMS["embed_dim"], mask_ratio=0.75,
        decoder_dim=32, decoder_depth=2, decoder_heads=4, norm_pix_loss=True,
    )
    # len_keep = round(121 * 0.25) = 30.
    assert mae.num_tokens == 121 and mae.len_keep == 30, (mae.num_tokens, mae.len_keep)
    assert mae.total_channels == 145

    hsi = torch.randn(2, 144, 11, 11)
    aux = torch.randn(2, 1, 11, 11)
    loss, info = mae(hsi, aux)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    assert info["pred"].shape == (2, 121, 145), info["pred"].shape
    assert info["mask"].shape == (2, 121)
    # 75% of tokens masked per sample.
    assert int(info["mask"][0].sum().item()) == 121 - 30


def test_mae_trains_mcrosspa_attention():
    """The decoder must cross-attend to the encoded CLS so the mCrossPA
    attention projections receive a reconstruction gradient. Without that, the
    CLS (the only thing mCrossPA's attention updates) is dropped before decoding
    and the fusion transformer never trains."""
    model = _encoder(144, 1)
    mae = MFTMAEPretrainModel(
        encoder=model, hsi_channels=144, aux_channels=1, use_aux=True,
        patch_size=11, embed_dim=DIMS["embed_dim"], mask_ratio=0.75,
        decoder_dim=32, decoder_depth=2, decoder_heads=4, norm_pix_loss=True,
    )
    hsi = torch.randn(2, 144, 11, 11)
    aux = torch.randn(2, 1, 11, 11)
    loss, _ = mae(hsi, aux)
    loss.backward()

    # mCrossPA query/key/value projections in the first encoder block must get
    # a non-trivial gradient (proves the cross-attention path trains them).
    for pname in ("q_proj", "k_proj", "v_proj"):
        w = getattr(model.encoder.blocks[0].attn, pname).weight
        assert w.grad is not None, f"{pname}.weight got no grad"
        assert torch.isfinite(w.grad).all(), f"{pname}.weight grad not finite"
        assert w.grad.abs().sum().item() > 0, f"{pname}.weight grad is all-zero"

    # The aux/LiDAR CLS tokenizer must also receive gradient (the CLS is built
    # from aux and feeds the reconstruction via cross-attention).
    assert model.aux_tokenizer.token_wV.grad is not None
    assert model.aux_tokenizer.token_wV.grad.abs().sum().item() > 0


def test_checkpoint_keys_round_trip():
    """The pretrain wrapper stores the encoder under ``encoder.`` and the MAE-only
    modules under ``decoder.`` / ``enc_to_dec.`` — the convention the evaluator's
    fix_state_dict_keys relies on (strip ``encoder.``, skip decoder/enc_to_dec)."""
    model = _encoder(144, 1)
    mae = MFTMAEPretrainModel(
        encoder=model, hsi_channels=144, aux_channels=1, use_aux=True,
        patch_size=11, embed_dim=DIMS["embed_dim"], mask_ratio=0.75,
        decoder_dim=32, decoder_depth=2, decoder_heads=4,
    )
    sd = mae.state_dict()
    assert any(k.startswith("encoder.") for k in sd)
    assert any(k.startswith("decoder.") for k in sd)
    assert any(k.startswith("enc_to_dec.") for k in sd)

    # Stripping the ``encoder.`` prefix yields keys that load into the bare encoder.
    bare = model.state_dict()
    stripped = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = MFTOriginal(
        hsi_channels=144, aux_channels=1, use_aux=True, patch_size=11,
        attention_type="mcross", **DIMS,
    ).load_state_dict(stripped, strict=False)
    assert not missing, f"unexpected missing encoder keys: {missing[:5]}"


# ----------------------------------------------------------------------
# Ablation 2: "Spatial" masking + loss (MFTSpatialMaskPretrainModel)
# ----------------------------------------------------------------------


def _spatial_mae(hsi_c=144, aux_c=1):
    enc = _encoder(hsi_c, aux_c)
    return enc, MFTSpatialMaskPretrainModel(
        encoder=enc, hsi_channels=hsi_c, aux_channels=aux_c, use_aux=True,
        patch_size=11, embed_dim=DIMS["embed_dim"], decoder_hidden_dim=64,
        spatial_mask_ratio=0.75, recon_sigma=1.0,
    )


def test_spatial_forward_loss_and_mask():
    _, mae = _spatial_mae(144, 1)
    assert mae.num_tokens == 121 and mae.total_channels == 145
    hsi = torch.randn(2, 144, 11, 11)
    aux = torch.randn(2, 1, 11, 11)
    loss, info = mae(hsi, aux)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    # MLP decoder reconstructs the full HSI+aux band vector per pixel token.
    assert info["pred"].shape == (2, 121, 145), info["pred"].shape
    # In-place spatial masking: ~75% of the 121 tokens masked (round(0.75*121)=91),
    # broadcast across all 145 bands in the per-entry mask.
    assert mae.spatial_masking.mask_token.shape == (1, 1, DIMS["embed_dim"])
    num_masked_tokens = int(round(0.75 * 121))
    assert int(info["mask"][0].sum().item()) == num_masked_tokens * 145


def test_spatial_trains_mcrosspa_attention_via_cls_injection():
    """CLS injection (patch_enc += encoded CLS) before the MLP decoder must route a
    reconstruction gradient into the mCrossPA attention projections and the LiDAR
    CLS tokenizer; otherwise the fusion transformer never trains."""
    enc, mae = _spatial_mae(144, 1)
    hsi = torch.randn(2, 144, 11, 11)
    aux = torch.randn(2, 1, 11, 11)
    loss, _ = mae(hsi, aux)
    loss.backward()

    for pname in ("q_proj", "k_proj", "v_proj"):
        w = getattr(enc.encoder.blocks[0].attn, pname).weight
        assert w.grad is not None, f"{pname}.weight got no grad"
        assert w.grad.abs().sum().item() > 0, f"{pname}.weight grad is all-zero"

    # The LiDAR CLS tokenizer receives gradient (CLS feeds reconstruction).
    assert enc.aux_tokenizer.token_wV.grad is not None
    assert enc.aux_tokenizer.token_wV.grad.abs().sum().item() > 0


def test_spatial_checkpoint_keys_round_trip():
    """spatial_masking / decoder modules are skipped at eval; encoder.* loads into
    the bare MFTOriginal."""
    enc, mae = _spatial_mae(144, 1)
    sd = mae.state_dict()
    assert any(k.startswith("encoder.") for k in sd)
    assert any(k.startswith("spatial_masking.") for k in sd)
    assert any(k.startswith("decoder.") for k in sd)

    stripped = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, _ = MFTOriginal(
        hsi_channels=144, aux_channels=1, use_aux=True, patch_size=11,
        attention_type="mcross", **DIMS,
    ).load_state_dict(stripped, strict=False)
    assert not missing, f"unexpected missing encoder keys: {missing[:5]}"

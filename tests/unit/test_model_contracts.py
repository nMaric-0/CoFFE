"""Shape, size and frozen-parameter contracts for the three routes.

PAPER_CANON §2 fixes CoFFE's geometry (11×11 patch, 121 pixel tokens + 1
class-agnostic token, D = 128, 2 heads, 2 layers) and its Houston parameter
count (**579,328**, with the ±1 % test this file provides). §1 fixes the two
HyperSIGMA input regimes and the label-free adaptations; §5 fixes the per-scene
band counts the encoders have to accept.

Everything runs on random weights and synthetic tensors: no dataset, no
released HyperSIGMA checkpoint (PAPER_CANON §7.5). The HyperSIGMA ViT
bodies are randomly initialised, so what is under test is the wrapper's
plumbing and its ``requires_grad`` partition, never a pretrained number.
"""

from __future__ import annotations

import pytest
import torch

from coffe.models import CoFFE, MFTOriginal
from tests.conftest import CANON_DATASET_SPECS

PATCH = 11
NUM_TOKENS = PATCH * PATCH  # 121 pixel tokens
EMBED_DIM = 128  # PAPER_CANON §2
NUM_HEADS = 2
NUM_LAYERS = 2

SCENE_IDS = list(CANON_DATASET_SPECS)


def _coffe(scene: str, *, use_aux: bool = True, use_projection: bool = False) -> CoFFE:
    """A CoFFE sized for ``scene`` with the paper's encoder hyperparameters."""
    spec = CANON_DATASET_SPECS[scene]
    return CoFFE(
        hsi_channels=spec["hsi_channels"],
        aux_channels=spec["aux_channels"],
        use_aux=use_aux,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        patch_size=PATCH,
        cls_token_weight=0.5,  # the paper's lambda (PAPER_CANON §8 D3)
        dropout=0.1,
        use_projection=use_projection,
    ).eval()


def _inputs(scene: str, batch: int = 3, seed: int = 0):
    spec = CANON_DATASET_SPECS[scene]
    gen = torch.Generator().manual_seed(seed)
    hsi = torch.rand(batch, spec["hsi_channels"], PATCH, PATCH, generator=gen)
    aux = torch.rand(batch, spec["aux_channels"], PATCH, PATCH, generator=gen)
    return hsi, aux


# ----------------------------------------------------------------------
# CoFFE
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scene", SCENE_IDS)
def test_coffe_forward_shapes_for_every_dataset_spec(scene: str) -> None:
    """144+1 / 63+1 / 64+2 input bands all reduce to 121 tokens of width 128."""
    model = _coffe(scene)
    hsi, aux = _inputs(scene)

    with torch.no_grad():
        patch_emb, cls_emb, aux_emb = model.forward_features(hsi, aux)

    batch = hsi.shape[0]
    assert patch_emb.shape == (batch, NUM_TOKENS, EMBED_DIM)
    assert cls_emb.shape == (batch, EMBED_DIM)
    # Documented compatibility alias: there is no separate aux token under
    # input-level fusion, so aux_emb *is* cls_emb.
    assert aux_emb is cls_emb


@pytest.mark.parametrize("scene", SCENE_IDS)
def test_coffe_token_budget_is_one_class_token_plus_one_per_pixel(scene: str) -> None:
    """PAPER_CANON §2: 121 pixel tokens + 1 prepended class-agnostic token."""
    model = _coffe(scene)
    assert model.pos_embed.shape == (1, 1 + NUM_TOKENS, EMBED_DIM)
    assert model.class_agnostic_emb.shape == (1, 1, EMBED_DIM)


@pytest.mark.parametrize("scene", SCENE_IDS)
def test_input_level_fusion_sizes_the_tokenizer_to_hsi_plus_aux(scene: str) -> None:
    """The fusion claim, read off the front end: one conv over all bands.

    HSI and LiDAR are concatenated *at the input*, so the channel tokenizer's
    input width is ``C_h + C_a`` — there is no separate aux front end and no
    external fusion token (contrast ``MFTOriginal``).
    """
    spec = CANON_DATASET_SPECS[scene]
    model = _coffe(scene)

    total = spec["hsi_channels"] + spec["aux_channels"]
    assert model.total_channels == total
    assert model.channel_tokenizer.conv[0].weight.shape[1] == total
    assert not hasattr(model, "aux_conv")


@pytest.mark.parametrize("scene", SCENE_IDS)
def test_hsi_only_sizing_drops_the_aux_bands(scene: str) -> None:
    """``use_aux=False`` sizes the tokenizer to HSI alone and ignores aux."""
    spec = CANON_DATASET_SPECS[scene]
    model = _coffe(scene, use_aux=False)
    hsi, aux = _inputs(scene)

    assert model.total_channels == spec["hsi_channels"]
    assert model.channel_tokenizer.conv[0].weight.shape[1] == spec["hsi_channels"]

    with torch.no_grad():
        without_aux = model.forward_features(hsi, None)[0]
        with_aux = model.forward_features(hsi, aux)[0]

    assert without_aux.shape == (hsi.shape[0], NUM_TOKENS, EMBED_DIM)
    # Passing aux to an HSI-only model must be a no-op, not a silent effect.
    assert torch.equal(without_aux, with_aux)


@pytest.mark.parametrize("scene", SCENE_IDS)
def test_pooled_eval_feature_is_128_dimensional(scene: str) -> None:
    """The eval feature is ``z = mean_j(patch_emb_j) + λ·cls_emb`` in R^128.

    Uses the harness's mirror of the live evaluator lines (PAPER_CANON §8 D3:
    ``forward_episode`` is dead code and is not the paper path).
    """
    from tests.equivalence._harness import live_eval_feature

    model = _coffe(scene)
    hsi, aux = _inputs(scene)

    feature = live_eval_feature(model, hsi, aux)

    assert feature.shape == (hsi.shape[0], EMBED_DIM)


def test_eval_feature_folds_in_the_class_token_with_weight_lambda() -> None:
    """λ is live: the pooled feature is the patch mean *plus* 0.5·cls_emb."""
    model = _coffe("houston")
    hsi, aux = _inputs("houston")

    with torch.no_grad():
        patch_emb, cls_emb, _ = model.forward_features(hsi, aux)
        adapted = model.eval_patch_embeddings(patch_emb, cls_emb)

    assert model.cls_token_weight == 0.5
    expected = patch_emb + 0.5 * cls_emb.unsqueeze(1)
    assert torch.allclose(adapted, expected, atol=0)
    # And the pooled feature differs from a plain patch-token pool, which is
    # what the paper's §3 prose describes (the D3 gap).
    assert not torch.allclose(adapted.mean(dim=1), patch_emb.mean(dim=1), atol=1e-6)


def test_houston_eval_encoder_parameter_count() -> None:
    """PAPER_CANON §2: **579,328** parameters for the Houston eval encoder.

    Asserted within ±1 % as the canon requires, plus an exact check — the
    count is a deterministic function of the architecture, so if it ever moves
    at all, that is a shape change worth failing on.
    """
    model = _coffe("houston")

    total = sum(p.numel() for p in model.parameters())

    assert total == 579_328
    assert abs(total - 579_328) / 579_328 <= 0.01


@pytest.mark.parametrize("scene", ["trento", "muufl"])
def test_other_scenes_have_marginally_fewer_parameters(scene: str) -> None:
    """PAPER_CANON §2: "marginally fewer on Trento/MUUFL"."""
    houston = sum(p.numel() for p in _coffe("houston").parameters())
    other = sum(p.numel() for p in _coffe(scene).parameters())

    assert other < houston
    assert (houston - other) / houston < 0.05  # marginal, not a different model


def test_projection_head_is_absent_at_eval() -> None:
    """PAPER_CANON §2/§4: the projection head is pretraining-only.

    With ``use_projection=False`` it is an ``nn.Identity``, which is also why
    the ``renormalize`` branch of ``eval_patch_embeddings`` is inert (§8 D3).
    """
    model = _coffe("houston", use_projection=False)

    assert isinstance(model.projection, torch.nn.Identity)
    assert not hasattr(model.projection, "l2_normalize")
    assert not any(name.startswith("projection.") for name, _ in model.named_parameters())


def test_projection_head_is_present_when_pretraining() -> None:
    model = _coffe("houston", use_projection=True)
    assert not isinstance(model.projection, torch.nn.Identity)
    assert any(name.startswith("projection.") for name, _ in model.named_parameters())


def test_gaussian_pool_weights_sum_to_one_when_enabled() -> None:
    """Centre-weighted *pooling* normalises to sum one (unlike the loss weight).

    ``pool_sigma`` is ``None`` in every paper run; the option is pinned so the
    two different normalisations cannot be conflated (see
    ``test_masking_loss.py`` for the mean-one reconstruction weight).
    """
    model = CoFFE(
        hsi_channels=144,
        aux_channels=1,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        patch_size=PATCH,
        use_projection=False,
        pool_sigma=2.0,
    )
    weights = model._center_pool_weights
    assert weights.shape == (NUM_TOKENS,)
    assert weights.sum().item() == pytest.approx(1.0, rel=1e-6)


# ----------------------------------------------------------------------
# MFTOriginal — the architectural control
# ----------------------------------------------------------------------


def _mft(scene: str) -> MFTOriginal:
    spec = CANON_DATASET_SPECS[scene]
    return MFTOriginal(
        hsi_channels=spec["hsi_channels"],
        aux_channels=spec["aux_channels"],
        embed_dim=64,
        num_heads=8,
        num_layers=NUM_LAYERS,
        mlp_dim=512,
        patch_size=PATCH,
        dropout=0.1,
    ).eval()


@pytest.mark.parametrize("scene", SCENE_IDS)
def test_mft_control_keeps_an_external_fusion_token(scene: str) -> None:
    """The control's contrast with CoFFE: a separate aux front end.

    MFT tokenises HSI and aux through *different* front ends and fuses through
    an external token, which is the architecture CoFFE's input-level fusion is
    compared against (PAPER_CANON §1).
    """
    spec = CANON_DATASET_SPECS[scene]
    model = _mft(scene)

    assert hasattr(model, "aux_conv")
    assert model.aux_conv[0].weight.shape[1] == spec["aux_channels"]
    assert not hasattr(model, "channel_tokenizer")


def test_mft_control_ignores_lambda() -> None:
    """PAPER_CANON §8 D3.4: λ is CoFFE-only; the control's hook is a no-op."""
    model = _mft("houston")
    hsi, aux = _inputs("houston")

    with torch.no_grad():
        patch_emb, cls_emb, _ = model.forward_features(hsi, aux)
        adapted = model.eval_patch_embeddings(patch_emb, cls_emb)

    assert adapted is patch_emb or torch.equal(adapted, patch_emb)

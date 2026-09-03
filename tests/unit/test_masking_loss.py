"""Eq. 1 semantics: what gets masked, and what the loss averages over.

PAPER_CANON §3 states the objective as

* SimMIM-style **in-place** masked reconstruction over the token space, with
  two composable masks — band ``(pixel, band)`` entries and whole-token
  spatial masks;
* loss = masked MSE over the **union** of the two masks;
* an optional Gaussian centre weight ``w_j`` with **mean one**.

Each clause is pinned below. The masks come out of the model (captured with
forward hooks on the two masking submodules, so the union is checked against
its actual inputs rather than against a second copy of the same arithmetic).

``tests/integration/test_pretrain_simmim.py`` covers the shape/gradient
contract of the same model; this file is only about the masking and loss
*semantics*.
"""

from __future__ import annotations

import pytest
import torch

from coffe.models import CoFFE
from coffe.pretrain.masked_modeling import SpatialTokenMasking, UnifiedBandMasking
from coffe.pretrain.simmim import SimMIMPretrainModel
from coffe.utils.spatial_weights import make_center_weights

PATCH = 11
NUM_TOKENS = PATCH * PATCH  # 121
HSI = 8  # small band count: masking semantics do not depend on it
AUX = 1


def _encoder(embed_dim: int = 32, hsi_channels: int = HSI, use_aux: bool = True) -> CoFFE:
    return CoFFE(
        hsi_channels=hsi_channels,
        aux_channels=AUX,
        use_aux=use_aux,
        embed_dim=embed_dim,
        num_heads=2,
        num_layers=2,
        patch_size=PATCH,
        use_projection=False,
    )


def _simmim(
    band: float,
    token: float,
    *,
    recon_sigma: float | None = None,
    embed_dim: int = 32,
) -> SimMIMPretrainModel:
    return SimMIMPretrainModel(
        encoder=_encoder(embed_dim),
        hsi_channels=HSI,
        aux_channels=AUX,
        patch_size=PATCH,
        embed_dim=embed_dim,
        decoder_hidden_dim=64,
        band_mask_ratio=band,
        spatial_mask_ratio=token,
        recon_sigma=recon_sigma,
    )


def _batch(batch: int = 4, hsi_channels: int = HSI, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    hsi = torch.rand(batch, hsi_channels, PATCH, PATCH, generator=gen)
    aux = torch.rand(batch, AUX, PATCH, PATCH, generator=gen)
    return hsi, aux


# ----------------------------------------------------------------------
# Realized mask ratios vs. the configured ones
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.75, 0.85])
def test_token_mask_realized_count_is_exact(ratio: float) -> None:
    """Spatial masking takes a fixed *count*, so the realized ratio is exact.

    ``num_mask = round(ratio * N)``: 91/121 at 0.75, 103/121 at 0.85. Every
    sample in the batch gets exactly that many masked tokens — this is the one
    mask rate that carries no sampling noise.
    """
    torch.manual_seed(0)
    masking = SpatialTokenMasking(embed_dim=16, mask_ratio=ratio)
    tokens = torch.randn(8, NUM_TOKENS, 16)

    _out, token_mask = masking(tokens)

    expected = round(ratio * NUM_TOKENS)
    assert torch.equal(token_mask.sum(dim=1), torch.full((8,), float(expected))), (
        f"expected exactly {expected} masked tokens per sample"
    )


@pytest.mark.parametrize("ratio", [0.75, 0.85, 0.9])
def test_band_mask_realized_ratio_tracks_the_configured_ratio(ratio: float) -> None:
    """Band masking is per-entry Bernoulli, so the realized ratio is only ~exact.

    Deliberately *not* asserted exact: ``UnifiedBandMasking`` draws one
    independent sample per ``(pixel, band)`` entry (``noise < mask_ratio``),
    which is the frozen behaviour. With a fixed seed and 8*32*121 entries the
    realized fraction sits within 0.005 of the configured one; the tolerance is
    sampling noise, not slack.
    """
    torch.manual_seed(0)
    masking = UnifiedBandMasking(num_channels=32, mask_ratio=ratio)
    x = torch.rand(8, 32, PATCH, PATCH)

    _x_masked, mask = masking(x)

    assert mask.mean().item() == pytest.approx(ratio, abs=0.005)


def test_band_masking_replaces_only_masked_entries() -> None:
    """Masked entries carry the learnable per-channel fill; visible ones survive."""
    torch.manual_seed(0)
    masking = UnifiedBandMasking(num_channels=6, mask_ratio=0.5)
    x = torch.rand(4, 6, PATCH, PATCH)

    x_masked, mask = masking(x)

    fill = masking.mask_value.detach().expand_as(x)
    assert torch.allclose(x_masked[mask.bool()], fill[mask.bool()], atol=0)
    visible = ~mask.bool()
    assert torch.allclose(x_masked[visible], x[visible], atol=0)


def test_at_least_one_mask_is_required() -> None:
    with pytest.raises(ValueError, match="band_mask_ratio / spatial_mask_ratio"):
        _simmim(0.0, 0.0)


# ----------------------------------------------------------------------
# Composition: the loss mask is the union of the two masks
# ----------------------------------------------------------------------


def _capture_masks(model: SimMIMPretrainModel, hsi, aux):
    """Run a forward pass, returning ``(loss, info, band_mask, token_mask)``.

    The two component masks are captured with forward hooks on the masking
    submodules, so the union assertion below compares the model's loss mask
    against the masks the model actually drew.
    """
    captured: dict[str, torch.Tensor] = {}

    def band_hook(_module, _inputs, output):
        captured["band"] = output[1].detach()

    def token_hook(_module, _inputs, output):
        captured["token"] = output[1].detach()

    handles = []
    if model.use_band_mask:
        handles.append(model.band_masking.register_forward_hook(band_hook))
    if model.use_spatial_mask:
        handles.append(model.spatial_masking.register_forward_hook(token_hook))
    try:
        with torch.no_grad():
            loss, info = model(hsi, aux)
    finally:
        for h in handles:
            h.remove()
    return loss, info, captured.get("band"), captured.get("token")


def test_composed_mask_is_the_union_of_band_and_token_masks() -> None:
    """Band + token masking compose as an elementwise union (PAPER_CANON §3).

    The loss mask must be ``max(band_entry, token_entry)`` over
    ``[B, N, C_total]``: a ``(pixel, band)`` entry counts if *either* its band
    was masked or its whole token was.
    """
    torch.manual_seed(0)
    model = _simmim(0.5, 0.5).eval()
    hsi, aux = _batch()

    _loss, info, band_mask, token_mask = _capture_masks(model, hsi, aux)

    total_channels = HSI + AUX
    band_entry = band_mask.flatten(2).transpose(1, 2)  # [B, N, C_total]
    token_entry = token_mask.unsqueeze(-1).expand(-1, -1, total_channels)
    expected_union = torch.maximum(band_entry, token_entry)

    assert torch.equal(info["mask"], expected_union)
    # Guard against a degenerate batch making the union assertion vacuous:
    # both masks must actually contribute something the other does not.
    assert (token_entry > band_entry).any() and (band_entry > token_entry).any()


def test_token_masking_alone_is_all_or_nothing_per_token() -> None:
    """With band masking off, a token is masked across every band or not at all."""
    torch.manual_seed(0)
    model = _simmim(0.0, 0.75).eval()
    hsi, aux = _batch()

    _loss, info, band_mask, _token_mask = _capture_masks(model, hsi, aux)

    assert band_mask is None, "band masking should not run at band_mask_ratio=0"
    bands = HSI + AUX
    row_sums = info["mask"].sum(dim=-1)
    assert bool(((row_sums == 0) | (row_sums == bands)).all())
    assert torch.equal(
        (row_sums > 0).sum(dim=-1), torch.full((hsi.shape[0],), round(0.75 * NUM_TOKENS))
    )


# ----------------------------------------------------------------------
# The loss averages over the masked entries only
# ----------------------------------------------------------------------


def _masked_mean(sq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Eq. 1's reduction, written out: sum of masked error / number masked."""
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)


@pytest.mark.parametrize(
    "band,token",
    [(0.85, 0.0), (0.0, 0.75), (0.85, 0.75), (0.75, 0.75)],
    ids=["simmim_band", "simmim_token", "simmim_band_token", "simmim_band_token_houston"],
)
def test_loss_is_the_masked_mean_squared_error(band: float, token: float) -> None:
    """The returned scalar is exactly the MSE over the masked entries.

    Covers all four mask-rate pairs that produced paper numbers, including the
    ``(0.75, 0.75)`` Houston band+token cell (PAPER_CANON §8 D19).
    """
    torch.manual_seed(0)
    model = _simmim(band, token).eval()
    hsi, aux = _batch()

    loss, info, _band_mask, _token_mask = _capture_masks(model, hsi, aux)

    sq = (info["pred"] - info["target"]) ** 2
    assert loss.item() == pytest.approx(_masked_mean(sq, info["mask"]).item(), rel=1e-6)


def test_unmasked_entries_do_not_enter_the_loss() -> None:
    """Perturbing an unmasked entry leaves the loss unchanged; a masked one moves it.

    The perturbation is applied to the *captured* ``(pred, target, mask)``
    triple rather than to the input tensor. Perturbing an input entry would
    also change every prediction through the encoder's attention, so it could
    never isolate the reduction — whereas Eq. 1's claim is precisely about
    which entries of that triple the average runs over.
    """
    torch.manual_seed(0)
    model = _simmim(0.5, 0.0).eval()
    hsi, aux = _batch()

    loss, info, _band_mask, _token_mask = _capture_masks(model, hsi, aux)
    pred, target, mask = info["pred"], info["target"], info["mask"]

    baseline = _masked_mean((pred - target) ** 2, mask)
    assert loss.item() == pytest.approx(baseline.item(), rel=1e-6)

    visible_idx = (mask == 0).nonzero()[0].tolist()
    masked_idx = mask.nonzero()[0].tolist()

    perturbed_visible = target.clone()
    perturbed_visible[tuple(visible_idx)] += 100.0
    assert _masked_mean((pred - perturbed_visible) ** 2, mask).item() == pytest.approx(
        baseline.item(), rel=0, abs=0
    )

    perturbed_masked = target.clone()
    perturbed_masked[tuple(masked_idx)] += 100.0
    assert _masked_mean((pred - perturbed_masked) ** 2, mask).item() > baseline.item()


@pytest.mark.parametrize("recon_loss", ["mse", "l1", "smooth_l1"])
def test_selectable_reconstruction_losses_use_the_same_reduction(recon_loss: str) -> None:
    """``l1``/``smooth_l1`` are selectable; only ``mse`` produced paper numbers."""
    torch.manual_seed(0)
    model = SimMIMPretrainModel(
        encoder=_encoder(),
        hsi_channels=HSI,
        aux_channels=AUX,
        patch_size=PATCH,
        embed_dim=32,
        decoder_hidden_dim=64,
        band_mask_ratio=0.5,
        spatial_mask_ratio=0.0,
        recon_loss=recon_loss,
    ).eval()
    hsi, aux = _batch()

    loss, info, _b, _t = _capture_masks(model, hsi, aux)

    diff = info["pred"] - info["target"]
    if recon_loss == "mse":
        sq = diff**2
    elif recon_loss == "l1":
        sq = diff.abs()
    else:
        sq = torch.nn.functional.smooth_l1_loss(
            info["pred"], info["target"], reduction="none", beta=1.0
        )
    assert loss.item() == pytest.approx(_masked_mean(sq, info["mask"]).item(), rel=1e-6)


def test_unknown_reconstruction_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="recon_loss must be one of"):
        SimMIMPretrainModel(
            encoder=_encoder(),
            hsi_channels=HSI,
            aux_channels=AUX,
            patch_size=PATCH,
            embed_dim=32,
            band_mask_ratio=0.5,
            spatial_mask_ratio=0.0,
            recon_loss="cosine",
        )


# ----------------------------------------------------------------------
# The optional Gaussian centre weight has mean one
# ----------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0])
def test_reconstruction_centre_weights_have_mean_one(sigma: float) -> None:
    """PAPER_CANON §3: ``w_j`` has **mean one**, so the loss scale is preserved.

    Note the two normalisations of ``make_center_weights`` are different and
    both are load-bearing: ``normalize=True`` gives weights that *sum* to one
    (pooling), while the reconstruction weight divides by the mean instead
    (``coffe/pretrain/simmim.py``), giving mean one over the 121 tokens.
    """
    model = _simmim(0.85, 0.0, recon_sigma=sigma)

    weights = model._recon_center_weights
    assert weights.shape == (NUM_TOKENS,)
    assert weights.mean().item() == pytest.approx(1.0, rel=1e-6)
    # The classified pixel is the patch centre, and it is up-weighted.
    centre = PATCH * (PATCH // 2) + (PATCH // 2)
    assert weights[centre] == weights.max()
    assert weights[centre].item() > 1.0


def test_no_centre_weight_buffer_when_recon_sigma_is_none() -> None:
    """Uniform weighting must not silently register a weight buffer."""
    model = _simmim(0.85, 0.0, recon_sigma=None)
    assert not hasattr(model, "_recon_center_weights")
    assert "_recon_center_weights" not in dict(model.named_buffers())


def test_centre_weighted_loss_is_the_weighted_masked_mean() -> None:
    """With ``recon_sigma`` on, the error is scaled by ``w_j`` before the average."""
    torch.manual_seed(0)
    model = _simmim(0.85, 0.0, recon_sigma=1.0).eval()
    hsi, aux = _batch()

    loss, info, _b, _t = _capture_masks(model, hsi, aux)

    weights = make_center_weights(PATCH, 1.0, normalize=False)
    weights = weights / weights.mean()
    sq = ((info["pred"] - info["target"]) ** 2) * weights.view(1, NUM_TOKENS, 1)
    assert loss.item() == pytest.approx(_masked_mean(sq, info["mask"]).item(), rel=1e-6)


def test_hsi_only_masking_covers_hsi_bands_only() -> None:
    """``use_aux=False`` masks and reconstructs HSI bands alone (no aux column)."""
    torch.manual_seed(0)
    model = SimMIMPretrainModel(
        encoder=_encoder(use_aux=False),
        hsi_channels=HSI,
        aux_channels=AUX,
        use_aux=False,
        patch_size=PATCH,
        embed_dim=32,
        decoder_hidden_dim=64,
        band_mask_ratio=0.85,
        spatial_mask_ratio=0.0,
    ).eval()
    hsi, aux = _batch()

    _loss, info, _b, _t = _capture_masks(model, hsi, aux)

    assert model.total_channels == HSI
    assert info["pred"].shape == (hsi.shape[0], NUM_TOKENS, HSI)
    assert info["target"].shape == (hsi.shape[0], NUM_TOKENS, HSI)

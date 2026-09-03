"""HyperSIGMA route contracts: input regimes, feature widths, frozen bodies.

PAPER_CANON §1 names the two input regimes — ``backbone_native`` (64×64, by
``pad`` or ``upscale``) and ``patch_native`` (11×11) — and the five
adaptations: ``frozen``, ``sem_only``, ``spatial``, ``spectral``,
``joint_sem``. §6 Table 3 fixes the three feature widths (spatial 768-d,
spectral 768-d, fused SEM 512-d) and records the route's size: ~180 M
parameters with both ViT-Base bodies plus SEM, of which label-free adaptation
trains 1.13 M-8.45 M.

The point of this file is the **``requires_grad`` partition**: the released
transformer bodies must stay frozen in every adaptation, and only the
random-initialised pieces plus that adaptation's decoders may train. It is the
property that makes the route "label-free adaptation from released
checkpoints" rather than fine-tuning.

The ViT bodies are randomly initialised — no released checkpoint is needed
(CLAUDE.md hard rule 5), and no number produced here is comparable to a paper
cell. ``tests/unit/test_hypersigma_shapes.py`` and
``test_hypersigma_native_shapes.py`` cover the forward shapes in more detail;
this file does not repeat them.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from coffe.models.hypersigma import HyperSIGMADual, HyperSIGMAFewShot
from coffe.pretrain.hypersigma_mae import HyperSIGMAMaskedAdaptation

#: PAPER_CANON §1 adaptation ids -> the code's ``adapt_mode`` value.
#: ``frozen`` has no adaptation module at all, so it maps to ``None``.
ADAPTATIONS: dict[str, str | None] = {
    "frozen": None,
    "sem_only": "sem_only",
    "spatial": "spatial_only",
    "spectral": "spectral_only",
    "joint_sem": "joint_sem",
}

#: Prefixes of the released ViT-Base bodies. Nothing under these may ever
#: carry ``requires_grad=True``.
FROZEN_BODY_PREFIXES = ("dual.spat.model.blocks.", "dual.spec.model.blocks.")

#: Trainable-parameter counts of the adaptations behind Table 3, all in the
#: 11x11 / 100-band spatial geometry every published ``11x11`` cell used
#: (``configs/hypersigma/houston_patchnative_pca100_joint_sem.yaml``).
#: Measured in phase 7.
#:
#: PAPER_CANON §6 says "label-free adaptation trains 1.13M-8.45M". Both quoted
#: endpoints land on a config exactly — ``spatial`` = 1,133,188 is the "1.13M"
#: and ``joint_sem`` = 8,446,785 is the "8.45M" — so those are plainly the two
#: numbers the paper reports. But the range does **not** cover every adapted
#: Table 3 cell: the published ``11x11 / spectral`` cell trains **333,689**,
#: below the stated floor. Reported as a paper<->code mismatch (phase-7 gate
#: S5); nothing is changed here, and ``spectral`` is asserted alongside the
#: other two precisely so the gap stays visible.
PUBLISHED_ADAPT_TRAINABLE = {
    "spatial": 1_133_188,  # canon's "1.13M"
    "spectral": 333_689,  # BELOW canon's stated floor (S5)
    "joint_sem": 8_446_785,  # canon's "8.45M"
}


def _pca(path: Path, in_bands: int, n_components: int) -> str:
    """A tiny deterministic PCA, so ``HyperSIGMADual`` can be constructed."""
    rng = np.random.default_rng(0)
    pixels = rng.standard_normal(size=(2048, in_bands)).astype(np.float32)
    fitted = PCA(n_components=n_components, svd_solver="full").fit(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(fitted, fh)
    return str(path)


def _dual_patch_native(tmp_path: Path, hsi_channels: int = 144) -> HyperSIGMADual:
    """11×11 straight into the encoders (PAPER_CANON §1 ``patch_native``)."""
    return HyperSIGMADual(
        pca_spat_path=_pca(tmp_path / "pca3.pkl", hsi_channels, 3),
        spat_ckpt=None,
        spec_ckpt=None,
        hsi_channels=hsi_channels,
        spat_patch_k=3,
        freeze_body=True,
    )


def _dual_backbone_native(
    tmp_path: Path, input_fit: str, hsi_channels: int = 144
) -> HyperSIGMADual:
    """64×64 with the encoders at their pretrained input size."""
    pca_path = _pca(tmp_path / "pca100.pkl", hsi_channels, 100)
    return HyperSIGMADual(
        pca_spat_path=pca_path,
        spat_ckpt=None,
        spec_ckpt=None,
        hsi_channels=hsi_channels,
        freeze_body=True,
        native_geometry=True,
        native_pca_spat_path=pca_path,
        native_spat_in_chans=100,
        input_fit=input_fit,
    )


@pytest.fixture(scope="module")
def patch_native(tmp_path_factory) -> HyperSIGMADual:
    return _dual_patch_native(tmp_path_factory.mktemp("hs_patch"))


# ----------------------------------------------------------------------
# Feature widths and the two input regimes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,width",
    [("spat_pool", 768), ("spec_pool", 768), ("fused", 512)],
    ids=["spatial", "spectral", "fused_sem"],
)
def test_patch_native_feature_widths_match_table_3(
    patch_native: HyperSIGMADual, mode: str, width: int
) -> None:
    """PAPER_CANON §6: sp. = 768-d, sc. = 768-d, fu. = 512-d."""
    model = HyperSIGMAFewShot(dual=patch_native, mode=mode, distance_metric="euclidean")
    x = torch.rand(2, 144, 11, 11)

    with torch.no_grad():
        patch, cls, aux = model.forward_features(x, None)

    assert patch.shape == (2, 1, width)
    assert cls.shape == (2, width)
    assert aux is cls


@pytest.mark.parametrize("input_fit", ["pad", "upscale"])
@pytest.mark.slow
def test_backbone_native_regime_accepts_11x11_and_fits_to_64(
    tmp_path: Path, input_fit: str
) -> None:
    """The 64×64 regime takes the same 11×11 patch and fits it (pad | upscale)."""
    dual = _dual_backbone_native(tmp_path, input_fit)
    model = HyperSIGMAFewShot(dual=dual, mode="spat_pool", distance_metric="euclidean")
    x = torch.rand(2, 144, 11, 11)

    with torch.no_grad():
        _patch, cls, _aux = model.forward_features(x, None)

    assert cls.shape == (2, 768)
    assert dual.input_fit == input_fit


@pytest.mark.parametrize("hsi_channels", [144, 63, 64], ids=["houston", "trento", "muufl"])
def test_every_scene_band_count_reaches_the_spectral_branch(
    tmp_path: Path, hsi_channels: int
) -> None:
    """The spectral branch takes raw bands and lands on its 100 tokens.

    PAPER_CANON §5: Houston's 144 bands go through PCA 144→100; Trento (63) and
    MUUFL (64) are linearly resampled up to 100. Either way the branch sees 100
    spectral tokens.
    """
    dual = _dual_patch_native(tmp_path, hsi_channels)
    x = torch.rand(2, hsi_channels, 11, 11)

    with torch.no_grad():
        features = dual.spec(x)

    assert features[-1].shape == (2, 100, 768)


def test_few_shot_wrapper_freezes_everything() -> None:
    """The ``frozen`` adaptation: nothing in the eval wrapper trains.

    Asserted on the wrapper's own construction contract rather than by building
    a dual, so it costs nothing.
    """
    import inspect

    source = inspect.getsource(HyperSIGMAFewShot.__init__)
    assert "requires_grad_(False)" in source


def test_frozen_route_has_no_trainable_parameters(patch_native: HyperSIGMADual) -> None:
    model = HyperSIGMAFewShot(dual=patch_native, mode="fused", distance_metric="euclidean")

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]

    assert trainable == []
    assert not model.training


def test_route_size_is_two_orders_above_coffe(patch_native: HyperSIGMADual) -> None:
    """PAPER_CANON §6: ~180 M parameters vs CoFFE's ~579 K."""
    total = sum(p.numel() for p in patch_native.parameters())

    assert 150_000_000 < total < 220_000_000
    assert total / 579_328 > 100  # ">2 orders of magnitude fewer" for CoFFE


# ----------------------------------------------------------------------
# The requires_grad partition, per adaptation
# ----------------------------------------------------------------------


def _dual_patch_native_pca100(tmp_path: Path) -> HyperSIGMADual:
    """11x11 with the 100-band spatial front end — the Table 3 ``11x11`` geometry.

    Every published ``11x11`` cell used 100 spatial bands, not the 3 that
    ``configs/hypersigma/houston_patchnative_joint_sem.yaml`` carries (that
    file says so itself: "No published cell as written").
    """
    pca_path = _pca(tmp_path / "pca100.pkl", 144, 100)
    return HyperSIGMADual(
        pca_spat_path=pca_path,
        spat_ckpt=None,
        spec_ckpt=None,
        hsi_channels=144,
        spat_patch_k=3,
        freeze_body=True,
    )


def _adaptation(dual: HyperSIGMADual, adapt_mode: str) -> HyperSIGMAMaskedAdaptation:
    return HyperSIGMAMaskedAdaptation(
        dual=dual,
        adapt_mode=adapt_mode,
        hsi_channels=144,
        patch_size=11,
        mask_ratio=0.75,  # PAPER_CANON §1: 75% token masking
    )


@pytest.mark.parametrize("adaptation", [a for a in ADAPTATIONS if ADAPTATIONS[a] is not None])
def test_released_transformer_bodies_stay_frozen(tmp_path: Path, adaptation: str) -> None:
    """No adaptation may train a released ViT-Base block. This is the invariant.

    Built fresh per adaptation: ``HyperSIGMAMaskedAdaptation.__init__`` mutates
    ``requires_grad`` on the dual it is given, so sharing one dual across modes
    would let an earlier mode's freezing leak into a later one.
    """
    adapt_mode = ADAPTATIONS[adaptation]
    assert adapt_mode is not None
    model = _adaptation(_dual_patch_native(tmp_path, 144), adapt_mode)

    leaked = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and name.startswith(FROZEN_BODY_PREFIXES)
    ]

    assert leaked == [], f"{adaptation} unfroze released body weights: {leaked[:5]}"


@pytest.mark.parametrize(
    "adaptation,expected_trainable_prefixes,expected_frozen_prefixes",
    [
        (
            "spatial",
            ("spat_mask_token", "spat_decoder.", "dual.spat.model.patch_embed."),
            ("dual.spec.", "dual.sem."),
        ),
        (
            "spectral",
            ("spec_mask_token", "spec_decoder.", "dual.spec.model.spat_map."),
            ("dual.spat.", "dual.sem."),
        ),
        (
            "sem_only",
            ("spat_mask_token", "spec_mask_token", "fused_decoder.", "dual.spec.model.l1."),
            ("dual.spat.", "dual.spec.model.blocks."),
        ),
        (
            "joint_sem",
            ("spat_decoder.", "spec_decoder.", "fused_decoder."),
            ("dual.spat.model.blocks.", "dual.spec.model.blocks."),
        ),
    ],
)
def test_each_adaptation_trains_exactly_its_own_pieces(
    tmp_path: Path,
    adaptation: str,
    expected_trainable_prefixes: tuple[str, ...],
    expected_frozen_prefixes: tuple[str, ...],
) -> None:
    """Every adaptation's trainable set is the one PAPER_CANON §1 describes.

    ``spatial``/``spectral`` train one branch's random-init pieces and its
    decoder; ``sem_only`` trains only the fusion path (SEM + fused decoder +
    the spectral ``l1`` projection + the two mask tokens); ``joint_sem`` trains
    both branches plus the fusion path.
    """
    adapt_mode = ADAPTATIONS[adaptation]
    assert adapt_mode is not None
    model = _adaptation(_dual_patch_native(tmp_path, 144), adapt_mode)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable, f"{adaptation} trains nothing"

    for prefix in expected_trainable_prefixes:
        assert any(n.startswith(prefix) for n in trainable), f"{adaptation} should train {prefix!r}"
    for prefix in expected_frozen_prefixes:
        offenders = sorted(n for n in trainable if n.startswith(prefix))
        assert offenders == [], f"{adaptation} should freeze {prefix!r}, trains {offenders[:5]}"


@pytest.mark.parametrize("adaptation", sorted(PUBLISHED_ADAPT_TRAINABLE))
def test_published_adaptation_trainable_counts(tmp_path: Path, adaptation: str) -> None:
    """The trainable count of each published ``11x11`` adaptation, exactly.

    All three in the 11x11 / 100-band spatial geometry — the one every published
    ``11x11`` cell used. Asserted exactly rather than within a tolerance: the
    count is a deterministic function of the architecture, so any drift is a
    shape change.

    Two of the three are PAPER_CANON §6's quoted endpoints (1.13 M, 8.45 M); the
    third, ``spectral`` at 333,689, sits below the stated floor and is the
    reason §6's range does not describe the published set (gate item S5).
    """
    adapt_mode = ADAPTATIONS[adaptation]
    assert adapt_mode is not None
    model = _adaptation(_dual_patch_native_pca100(tmp_path), adapt_mode)

    counts = model.parameter_counts()

    assert counts["total_trainable"] == PUBLISHED_ADAPT_TRAINABLE[adaptation]


@pytest.mark.slow
def test_backbone_native_sem_only_trainable_count(tmp_path: Path) -> None:
    """The fourth published adapted cell: ``64x64 pad / SEM-only / fused``.

    PAPER_CANON §6 Table 3's remaining adapted row. It trains **7,079,748**
    parameters — inside §6's stated 1.13 M-8.45 M either way, so it does not
    change S5's conclusion, but it is one of the four counts the phase-7 log
    reasons about and so is pinned rather than left as prose.

    ``pad`` and ``upscale`` share this count: the input fit changes the tensor
    fed to a frozen branch, not any parameter shape.
    """
    dual = _dual_backbone_native(tmp_path, "pad")
    model = _adaptation(dual, "sem_only")

    assert model.parameter_counts()["total_trainable"] == 7_079_748


@pytest.mark.parametrize("adaptation", [a for a in ADAPTATIONS if ADAPTATIONS[a] is not None])
def test_adaptation_trains_a_sliver_of_the_route(tmp_path: Path, adaptation: str) -> None:
    """Every adaptation is small next to the frozen bodies it sits on.

    The invariant, stated so it holds for any front-end width: what trains is
    under 5 % of what stays frozen. A mode that unfroze a ViT-Base body would
    blow straight through this.
    """
    adapt_mode = ADAPTATIONS[adaptation]
    assert adapt_mode is not None
    model = _adaptation(_dual_patch_native_pca100(tmp_path), adapt_mode)

    counts = model.parameter_counts()

    assert counts["total_frozen"] > 150_000_000
    assert counts["total_trainable"] / counts["total_frozen"] < 0.05, counts


def test_unknown_adaptation_is_rejected(patch_native: HyperSIGMADual) -> None:
    with pytest.raises(ValueError, match="adapt_mode must be one of"):
        _adaptation(patch_native, "everything")


def test_mask_ratio_must_be_a_proper_fraction(patch_native: HyperSIGMADual) -> None:
    with pytest.raises(ValueError, match="mask_ratio must be in"):
        HyperSIGMAMaskedAdaptation(
            dual=patch_native, adapt_mode="joint_sem", hsi_channels=144, mask_ratio=1.0
        )

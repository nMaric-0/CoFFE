"""Config parity: the committed configs must state the paper's constants.

This file pins code to paper. Every constant asserted here is quoted from
PAPER_CANON, written out as a literal above the assertion, and compared against
what ``configs/`` and ``scripts/reproduce/`` actually say — so a config edit
that drifts from the paper fails here rather than at review time.

Two caveats the canon insists on, both honoured below:

* **``configs/`` is not uniformly a set of paper recipes.** The six
  ``configs/coffe/<scene>_simmim[_hsi].yaml`` files are the 5-seed significance
  runner's *base* configs and carry whatever mask rates they happen to carry
  (PAPER_CANON §9). Only the per-cell configs are held to the regime rates.
* **The mask rates are per cell, not global.** The Houston band+token cell ran
  at ``(0.75, 0.75)``, not the ``(0.85, 0.75)`` §1 states for that regime
  (§8 D19).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.reproduce import sig_significance_config as sig

REPO = Path(__file__).resolve().parents[2]
COFFE_CONFIGS = sorted((REPO / "configs" / "coffe").glob("*.yaml"))
MFT_CONFIGS = sorted((REPO / "configs" / "mft").glob("*.yaml"))
HYPERSIGMA_CONFIGS = sorted((REPO / "configs" / "hypersigma").glob("*.yaml"))
ALL_CONFIGS = COFFE_CONFIGS + MFT_CONFIGS + HYPERSIGMA_CONFIGS

# PAPER_CANON §2 — CoFFE encoder geometry.
PATCH_SIZE = 11
COFFE_EMBED_DIM = 128
COFFE_NUM_HEADS = 2
COFFE_NUM_LAYERS = 2

# The MFT control's faithful arch (Roy et al.; dim = FM*4 with FM = 16).
MFT_EMBED_DIM = 64
MFT_NUM_HEADS = 8
MFT_NUM_LAYERS = 2
MFT_MLP_DIM = 512

# PAPER_CANON §1 — masking regime -> (band rate, spatial-token rate).
REGIME_MASK_RATES = {
    "simmim_band": (0.85, 0.0),
    "simmim_token": (0.0, 0.75),
    "simmim_band_token": (0.85, 0.75),
}
#: PAPER_CANON §8 D19: this one cell used band 0.75, and its config must say so.
REGIME_MASK_RATE_EXCEPTIONS = {"houston_simmim_band_token": (0.75, 0.75)}
#: PAPER_CANON §1/§3 — the MAE baseline's token-drop rate.
MAE_MASK_RATIO = 0.75

#: PAPER_CANON §9 — the significance runner's base configs, explicitly NOT
#: per-cell recipes, so the regime mask-rate law does not apply to them.
SIGNIFICANCE_BASE_STEMS = {
    f"{scene}_simmim{suffix}" for scene in ("houston", "trento", "muufl") for suffix in ("", "_hsi")
}

#: PAPER_CANON §4 — the seeds behind every across-seed std in Table 2.
SEEDS = [42, 123, 456, 789, 1011]


def _load(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _comment_header(path: Path) -> str:
    """The leading comment block as one whitespace-collapsed line.

    Collapsing means an assertion on a sentence does not depend on where the
    author happened to wrap it.
    """
    lines = []
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped.startswith("#"):
            if stripped:
                break
            continue
        lines.append(stripped.lstrip("#").strip())
    return " ".join(" ".join(lines).split())


def _stem(path: Path) -> str:
    return path.stem


def _regime_of(stem: str) -> str | None:
    """The masking regime a per-cell config name encodes, or None.

    Longest match first: ``simmim_band_token`` must not be read as
    ``simmim_band``.
    """
    if stem in SIGNIFICANCE_BASE_STEMS:
        return None
    for regime in ("simmim_band_token", "simmim_band", "simmim_token"):
        if regime in stem:
            return regime
    return "mae" if "_mae" in stem else None


def test_the_config_set_is_the_one_this_file_thinks_it_is() -> None:
    """Guard against the parametrized tests below silently covering nothing."""
    assert len(COFFE_CONFIGS) == 30, [p.name for p in COFFE_CONFIGS]
    assert len(MFT_CONFIGS) == 6, [p.name for p in MFT_CONFIGS]
    assert HYPERSIGMA_CONFIGS, "no HyperSIGMA configs found"


# ----------------------------------------------------------------------
# Architecture constants (PAPER_CANON §2)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=_stem)
def test_every_config_uses_the_paper_patch_size(path: Path) -> None:
    """PAPER_CANON §2: P x P = 11 x 11, in every route.

    Pretrain/adapt configs nest it under ``data:``; the HyperSIGMA eval config
    is flat. Both spellings are read, and one of them has to be present.
    """
    cfg = _load(path)
    patch_size = cfg.get("data", {}).get("patch_size", cfg.get("patch_size"))
    assert patch_size == PATCH_SIZE


@pytest.mark.parametrize("path", COFFE_CONFIGS, ids=_stem)
def test_coffe_configs_state_the_paper_encoder(path: Path) -> None:
    """PAPER_CANON §2: D = 128, 2 heads, 2 layers — the compact encoder."""
    model = _load(path)["model"]

    assert model["name"] == "coffe"
    assert model["embed_dim"] == COFFE_EMBED_DIM
    assert model["num_heads"] == COFFE_NUM_HEADS
    assert model["num_layers"] == COFFE_NUM_LAYERS


@pytest.mark.parametrize("path", MFT_CONFIGS, ids=_stem)
def test_mft_configs_state_the_faithful_control_arch(path: Path) -> None:
    """The control is original MFT, not a re-parameterised CoFFE."""
    model = _load(path)["model"]

    assert model["name"] == "mft_original"
    assert model["embed_dim"] == MFT_EMBED_DIM
    assert model["num_heads"] == MFT_NUM_HEADS
    assert model["num_layers"] == MFT_NUM_LAYERS
    assert model["mlp_dim"] == MFT_MLP_DIM
    # The control has no projection head and keeps its external fusion token.
    assert model["use_projection"] is False
    assert model["attention_type"] == "mcross"


# ----------------------------------------------------------------------
# Canonical vocabulary (PAPER_CANON §1)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", COFFE_CONFIGS + MFT_CONFIGS, ids=_stem)
def test_configs_are_written_in_canonical_vocabulary(path: Path) -> None:
    """Writers emit canonical only (PAPER_CANON §7.3): no legacy names.

    ``paths:`` is excluded — those name directories on disk and are on the
    DO-NOT-RENAME list.
    """
    cfg = _load(path)

    assert cfg["model"]["name"] in ("coffe", "mft_original")
    assert cfg["pretrain"]["objective"] in ("simmim", "mae")
    for section in ("model", "data", "pretrain", "hardware"):
        rendered = yaml.safe_dump(cfg.get(section, {}))
        for legacy in ("mft_cpea", "enhanced"):
            assert legacy not in rendered, f"legacy vocabulary {legacy!r} in {section}:"


# ----------------------------------------------------------------------
# Mask rates, per regime and per cell (PAPER_CANON §1, §8 D19)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", COFFE_CONFIGS + MFT_CONFIGS, ids=_stem)
def test_per_cell_mask_rates_match_the_regime_the_filename_names(path: Path) -> None:
    """Each per-cell config carries its regime's ``(r_b, r_s)`` pair.

    Including the Houston band+token exception, whose config must state
    ``(0.75, 0.75)`` — the rate that produced OA 64.63 (PAPER_CANON §8 D19).
    The MFT control supports token masking only, so its band rate is 0.
    """
    stem = _stem(path)
    regime = _regime_of(stem)
    if regime is None:
        pytest.skip(f"{stem} is a significance base config, not a cell recipe")

    pretrain = _load(path)["pretrain"]

    if regime == "mae":
        assert pretrain["objective"] == "mae"
        assert pretrain["mask_ratio"] == MAE_MASK_RATIO
        return

    assert pretrain["objective"] == "simmim"
    expected = REGIME_MASK_RATE_EXCEPTIONS.get(stem, REGIME_MASK_RATES[regime])
    actual = (pretrain["band_mask_ratio"], pretrain["spatial_mask_ratio"])
    assert actual == expected, f"{stem}: {actual} != {expected}"


def test_the_houston_band_token_exception_is_the_only_one() -> None:
    """D19 is a single-cell exception; no other config may quietly join it.

    Five of the six band+token configs use the §1 rate; only Houston's fused
    cell deviates. If a second config ever deviates, that is a new discrepancy
    to report, not a rate to normalise.
    """
    deviating = []
    for path in COFFE_CONFIGS + MFT_CONFIGS:
        stem = _stem(path)
        if _regime_of(stem) != "simmim_band_token":
            continue
        pretrain = _load(path)["pretrain"]
        rates = (pretrain["band_mask_ratio"], pretrain["spatial_mask_ratio"])
        if rates != REGIME_MASK_RATES["simmim_band_token"]:
            deviating.append((stem, rates))

    assert deviating == [("houston_simmim_band_token", (0.75, 0.75))]


@pytest.mark.parametrize("stem", sorted(SIGNIFICANCE_BASE_STEMS))
def test_significance_base_configs_say_what_they_are(stem: str) -> None:
    """PAPER_CANON §9: these six must disclaim being cell recipes.

    Their mask rates are not asserted precisely because they are not a paper
    recipe — but the header has to say so, or a reader will treat them as one.
    """
    path = REPO / "configs" / "coffe" / f"{stem}.yaml"
    header = _comment_header(path)

    assert "5-seed significance experiment's base config" in header
    assert "it is NOT the recipe of any Table 2 cell" in header


# ----------------------------------------------------------------------
# The reproduce pipeline's eval protocol (PAPER_CANON §4)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["houston", "trento", "muufl"])
def test_reproduce_eval_params_are_the_paper_protocol(dataset: str) -> None:
    """PAPER_CANON §4: K = 5, 100 queries/class, 1000 episodes, Euclidean NCM.

    ``n_way`` is deliberately absent so the evaluator defaults to every class
    in the scene (15/6/11), and ``use_projection`` is explicitly off.
    """
    params = sig.eval_params(dataset)

    assert params["k_shot"] == 5
    assert params["k_query"] == 100
    assert params["num_episodes"] == 1000
    assert params["distance_metric"] == "euclidean"
    assert params["prototype_mode"] == "mean_features"
    assert params["use_projection"] is False
    assert params["seed"] == 42
    assert "n_way" not in params, "n_way must default to the scene's class count"


def test_significance_experiment_uses_the_paper_seeds() -> None:
    """PAPER_CANON §4: seeds [42, 123, 456, 789, 1011], five of them."""
    assert sig.SEEDS == SEEDS
    assert len(sig.SEEDS) == 5


def test_significance_experiment_trains_to_700_epochs() -> None:
    """PAPER_CANON §8 D17: the ± column is a *separate* 700-epoch experiment.

    Not 1500. The means come from the canonical long runs evaluated at epoch
    950/975; the across-seed std comes from these fresh 700-epoch runs. Pinned
    so nobody "corrects" it to the schedule length.
    """
    assert sig.EPOCHS == 700
    assert sig.EVAL_EPOCH == 700
    assert 700 % sig.SAVE_INTERVAL == 0, "epoch-700 checkpoint would never be written"


def test_significance_covers_the_three_scenes() -> None:
    assert sig.DATASETS == ["houston", "trento", "muufl"]


def test_canonical_run_dirs_keep_their_frozen_names() -> None:
    """PAPER_CANON §7.3: ``_ENHANCED_CANONICAL`` names dirs on disk.

    These strings are DO-NOT-RENAME — they are paths into frozen experiment
    trees, not vocabulary. The keys use the legacy variant ids for the same
    reason.
    """
    canonical = sig._ENHANCED_CANONICAL

    assert set(canonical) == {
        (scene, variant)
        for scene in ("houston", "trento", "muufl")
        for variant in ("spatial", "spectral", "both")
    }
    assert canonical[("houston", "spatial")] == "houston_enhanced_spatial_run1"
    assert canonical[("houston", "both")] == "houston_enhanced_spec_spat_combined"
    # PAPER_CANON §8 D20: Trento's ± was measured on run1 while the published
    # mean comes from run2. Pinned so the mismatch stays visible.
    assert canonical[("trento", "spectral")] == "trento_enhanced_spectral_run1"
    assert canonical[("trento", "both")] == "trento_enhanced_spectral_spatial_run1"


def test_significance_group_base_configs_all_exist() -> None:
    """Every (group, dataset, variant) cell must point at a config that exists."""
    missing = []
    for group, dataset, variant in sig.cells():
        base = sig.GROUPS[group].base_config(dataset, variant)
        if not (REPO / base).exists():
            missing.append((group, dataset, variant, base))

    assert missing == []


def test_hypersigma_eval_config_disclaims_its_own_episode_constants() -> None:
    """``houston_eval.yaml`` matches no Table 3 cell, and must say so.

    Its 600 episodes / ``k_query`` 30 are this file's own defaults, not the
    paper protocol (PAPER_CANON §4, §8 D18: Table 2 used 100/1000 and Table 3
    used 2000). Pinned because a reader who took this file for the protocol
    would mis-reproduce every Table 3 cell. Its ``distance_metric`` *is*
    canonical since the phase-4 gate.
    """
    path = REPO / "configs" / "hypersigma" / "houston_eval.yaml"
    cfg = _load(path)
    header = _comment_header(path)

    assert cfg["distance_metric"] == "euclidean"
    assert cfg["k_shot"] == 5  # K = 5 is law in every table
    # The two constants that are *not* the protocol, and their disclaimer.
    assert (cfg["k_query"], cfg["num_episodes"]) == (30, 600)
    assert "match no Table 3 cell" in header

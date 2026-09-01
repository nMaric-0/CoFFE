"""The legacy-vocabulary compat layer (PAPER_CANON §7.3).

Frozen experiment trees on Nikola's machines carry the pre-paper vocabulary:
``pretrain_config.yaml`` with ``model.name: "mft_cpea"`` and (implicitly or
explicitly) objective ``"enhanced"``, ``results.json`` with
``model_type: "MFTCPEACosine"``. These tests pin that such artifacts still
read, still build the same model, and warn exactly once.

No dataset and no checkpoint is needed: the model-construction tests build
encoders from synthetic shapes only.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import coffe_compat
from coffe_compat import (
    LEGACY_MODEL_NAMES,
    LEGACY_MODEL_TYPES,
    LEGACY_OBJECTIVES,
    LEGACY_VARIANTS,
    normalize_model_name,
    normalize_model_type,
    normalize_objective,
    normalize_pretrain_config,
    normalize_variant,
)


@pytest.fixture(autouse=True)
def _fresh_warning_state():
    """Each test sees an unwarned module (the once-per-value cache is global)."""
    coffe_compat.reset_deprecation_state()
    yield
    coffe_compat.reset_deprecation_state()


# ----------------------------------------------------------------------
# Alias maps
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "normalize, mapping",
    [
        (normalize_model_name, LEGACY_MODEL_NAMES),
        (normalize_objective, LEGACY_OBJECTIVES),
        (normalize_variant, LEGACY_VARIANTS),
        (normalize_model_type, LEGACY_MODEL_TYPES),
    ],
)
def test_every_legacy_value_maps_to_its_canonical(normalize, mapping):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for legacy, canonical in mapping.items():
            assert normalize(legacy) == canonical


@pytest.mark.parametrize(
    "normalize, mapping",
    [
        (normalize_model_name, LEGACY_MODEL_NAMES),
        (normalize_objective, LEGACY_OBJECTIVES),
        (normalize_variant, LEGACY_VARIANTS),
        (normalize_model_type, LEGACY_MODEL_TYPES),
    ],
)
def test_canonical_values_round_trip_silently(normalize, mapping):
    """A canonical value must pass through unchanged and warn about nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for canonical in mapping.values():
            assert normalize(canonical) == canonical


def test_unknown_and_non_string_values_pass_through():
    """Validation belongs to the caller; this layer never rejects or coerces."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert normalize_model_name("hypersigma_dual") == "hypersigma_dual"
        assert normalize_objective("mae") == "mae"
        assert normalize_model_type("HyperSIGMADual") == "HyperSIGMADual"
        assert normalize_model_name(None) is None
        assert normalize_objective(3) == 3


def test_deprecation_warning_fires_once_per_value_and_origin():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for _ in range(5):
            normalize_model_name("mft_cpea", origin="pretrain_config.yaml")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "mft_cpea" in message and "coffe" in message
    assert "pretrain_config.yaml" in message

    # A different origin is a different report: the point of the warning is to
    # name the artifact that still carries the legacy value.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        normalize_model_name("mft_cpea", origin="eval_config.json")
    assert len(caught) == 1


# ----------------------------------------------------------------------
# Frozen-style configs
# ----------------------------------------------------------------------


#: A cut-down copy of a real frozen `pretrain_config.yaml`
#: (`experiments/houston_enhanced_spatial_run1/`), legacy vocabulary included.
FROZEN_PRETRAIN_CONFIG = {
    "model": {
        "name": "mft_cpea",
        "embed_dim": 128,
        "num_heads": 2,
        "num_layers": 2,
        "lambda_factor": 0.5,
        "dropout": 0.1,
        "use_projection": True,
        "proj_hidden_dim": 512,
        "proj_num_layers": 1,
        "proj_l2_normalize": True,
    },
    "data": {"patch_size": 11},
    "pretrain": {
        "objective": "enhanced",
        "band_mask_ratio": 0.0,
        "spatial_mask_ratio": 0.75,
        "recon_center_sigma": 1.0,
        "epochs": 1500,
    },
    "paths": {"checkpoint_dir": "./checkpoints/pretrained/houston_enhanced_2layers"},
}


def test_frozen_config_normalizes_vocabulary_and_nothing_else():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out = normalize_pretrain_config(FROZEN_PRETRAIN_CONFIG, origin="pretrain_config.yaml")

    assert out["model"]["name"] == "coffe"
    assert out["pretrain"]["objective"] == "simmim"

    # Every other value — including the DO-NOT-RENAME checkpoint dir and the
    # mask rates that define the regime — is untouched.
    assert out["paths"] == FROZEN_PRETRAIN_CONFIG["paths"]
    assert out["data"] == FROZEN_PRETRAIN_CONFIG["data"]
    assert out["pretrain"]["band_mask_ratio"] == 0.0
    assert out["pretrain"]["spatial_mask_ratio"] == 0.75
    assert out["model"]["lambda_factor"] == 0.5

    # And the caller's dict is not mutated.
    assert FROZEN_PRETRAIN_CONFIG["model"]["name"] == "mft_cpea"
    assert FROZEN_PRETRAIN_CONFIG["pretrain"]["objective"] == "enhanced"


def test_config_without_the_optional_keys_is_left_alone():
    """Most frozen SimMIM configs omit `objective` entirely (it defaulted)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert normalize_pretrain_config({}) == {}
        assert normalize_pretrain_config({"pretrain": {"epochs": 1500}}) == {
            "pretrain": {"epochs": 1500}
        }


def test_frozen_config_builds_the_same_model_as_its_canonical_twin():
    """A legacy config and its canonical rewrite must construct one model."""
    from scripts.evaluate import load_model_with_checkpoint
    from models import CoFFE

    legacy = dict(FROZEN_PRETRAIN_CONFIG["model"])
    legacy["use_projection"] = False
    canonical = dict(legacy, name="coffe")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from_legacy = load_model_with_checkpoint("random", "houston", legacy, "cpu")
    from_canonical = load_model_with_checkpoint("random", "houston", canonical, "cpu")

    assert isinstance(from_legacy, CoFFE)
    assert type(from_legacy) is type(from_canonical)
    assert from_legacy.state_dict().keys() == from_canonical.state_dict().keys()
    assert from_legacy.cls_token_weight == from_canonical.cls_token_weight == 0.5
    assert sum(p.numel() for p in from_legacy.parameters()) == sum(
        p.numel() for p in from_canonical.parameters()
    )


def test_legacy_model_name_still_selects_the_mft_control():
    """`mft_original` was already canonical; normalization must not disturb it."""
    from scripts.evaluate import load_model_with_checkpoint
    from models import MFTOriginal

    cfg = {"name": "mft_original", "embed_dim": 64, "num_heads": 8, "num_layers": 2,
           "mlp_dim": 512, "patch_size": 11}
    model = load_model_with_checkpoint("random", "houston", cfg, "cpu")
    assert isinstance(model, MFTOriginal)


# ----------------------------------------------------------------------
# Class aliases
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "legacy_name, canonical_module, canonical_name",
    [
        ("MFTCPEACosine", "models", "CoFFE"),
        ("MFTOriginalCosine", "models", "MFTOriginal"),
        ("HyperSIGMACosine", "models.hypersigma", "HyperSIGMAFewShot"),
        ("EnhancedMaskedSpectralSpatialModel", "pretrain", "SimMIMPretrainModel"),
    ],
)
def test_legacy_class_alias_resolves_to_the_canonical_class(
    legacy_name, canonical_module, canonical_name
):
    """Nikola's out-of-repo notebooks import the old class names."""
    import importlib

    canonical = getattr(importlib.import_module(canonical_module), canonical_name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        alias = getattr(coffe_compat, legacy_name)
    assert alias is canonical
    assert len(caught) == 1
    assert legacy_name in str(caught[0].message)
    assert canonical_name in str(caught[0].message)


@pytest.mark.parametrize(
    "package, legacy_name, canonical_name",
    [
        ("models", "MFTCPEACosine", "CoFFE"),
        ("models", "MFTOriginalCosine", "MFTOriginal"),
        ("models.hypersigma", "HyperSIGMACosine", "HyperSIGMAFewShot"),
        ("pretrain", "EnhancedMaskedSpectralSpatialModel", "SimMIMPretrainModel"),
    ],
)
def test_legacy_name_still_imports_from_its_original_package(
    package, legacy_name, canonical_name
):
    """`from models import MFTCPEACosine` is what old notebooks actually write."""
    import importlib

    mod = importlib.import_module(package)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        alias = getattr(mod, legacy_name)
    assert alias is getattr(mod, canonical_name)
    assert len(caught) == 1
    assert legacy_name in dir(mod)


def test_unknown_attribute_on_a_package_is_still_an_attribute_error():
    import models
    import pretrain
    from models import hypersigma

    for mod in (models, pretrain, hypersigma):
        with pytest.raises(AttributeError):
            getattr(mod, "NoSuchThing")


def test_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError):
        coffe_compat.NoSuchThing


def test_alias_class_constructs_and_shares_state_dict_keys():
    """The alias is the class itself, so checkpoints load through either name."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        LegacyCoFFE = coffe_compat.MFTCPEACosine

    from models import CoFFE

    kwargs = dict(hsi_channels=144, aux_channels=1, embed_dim=128, num_heads=2,
                  num_layers=2, patch_size=11, use_projection=False)
    torch.manual_seed(0)
    via_alias = LegacyCoFFE(**kwargs)
    torch.manual_seed(0)
    via_canonical = CoFFE(**kwargs)

    assert via_alias.state_dict().keys() == via_canonical.state_dict().keys()
    for key, tensor in via_alias.state_dict().items():
        assert torch.equal(tensor, via_canonical.state_dict()[key])

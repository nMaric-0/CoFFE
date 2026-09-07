"""The live checkpoint key-mapping rules (PAPER_CANON §7.2, §8 D15).

``coffe.eval.episodic.fix_state_dict_keys`` is the **live** key-fixing logic —
the dead twin ``utils/checkpoints.py`` was deleted in phase 3, and the
``fix_state_dict_keys`` mentions in the pretrain modules are docstrings. It is
what lets pre-rename checkpoints load unchanged, which §7.2 makes a hard
invariant: ``nn.Module`` attribute names define state_dict keys and are frozen
unless a key-map shim and a loading test land together.

This file pins the *rules* on synthetic dicts. The equivalence harness's G4
group pins the *outcome* against committed pre-refactor checkpoint fixtures
(``tests/equivalence/fixtures/``); the two are complementary, and neither
replaces the other.
"""

from __future__ import annotations

import pytest
import torch

from coffe.eval.episodic import fix_state_dict_keys, load_checkpoint_with_key_mapping
from coffe.models import CoFFE

#: Pretraining-only submodules whose weights must be dropped at eval, so they
#: never appear as "unexpected keys" and never shadow an eval parameter.
#: ``similarity`` is the old ``DenseSimilarity`` head: the evaluator is a
#: parameter-free nearest-class-mean classifier (PAPER_CANON §4).
DISCARDED_PREFIXES = (
    "spatial_masking",
    "spectral_masking",
    "lidar_masking",
    "spatial_decoder",
    "spectral_decoder",
    "lidar_decoder",
    "denoise_decoder",
    "enc_to_dec",
    "mask_token",
    "null_lidar",
    "noise_augmentation",
    "decoder",
    "aux_masking",
    "aux_decoder",
    "contrastive_head",
    "similarity",
)


def _eval_model() -> CoFFE:
    return CoFFE(
        hsi_channels=144,
        aux_channels=1,
        embed_dim=128,
        num_heads=2,
        num_layers=2,
        patch_size=11,
        use_projection=False,
    ).eval()


@pytest.mark.parametrize("prefix", DISCARDED_PREFIXES)
def test_pretraining_only_weights_are_discarded(prefix: str) -> None:
    """Each pretraining-only prefix is dropped rather than carried through."""
    model_state = _eval_model().state_dict()
    checkpoint = {f"{prefix}.weight": torch.zeros(2, 2)}

    fixed = fix_state_dict_keys(checkpoint, model_state)

    assert fixed == {}


def test_the_discard_list_is_the_one_this_file_declares() -> None:
    """A prefix added to (or dropped from) the live list must be noticed here.

    Read off the function's own source so the list cannot drift silently: the
    set of intentionally-discarded keys is part of the checkpoint contract.
    """
    import inspect

    source = inspect.getsource(fix_state_dict_keys)
    for prefix in DISCARDED_PREFIXES:
        assert f'"{prefix}"' in source, f"{prefix!r} no longer in the live skip list"


def test_dataparallel_module_prefix_is_stripped() -> None:
    """A checkpoint saved from ``nn.DataParallel`` loads on a bare module."""
    model_state = _eval_model().state_dict()
    key = "class_agnostic_emb"
    checkpoint = {f"module.{key}": model_state[key].clone()}

    fixed = fix_state_dict_keys(checkpoint, model_state)

    assert list(fixed) == [key]


def test_encoder_prefixed_keys_are_unwrapped_onto_the_bare_encoder() -> None:
    """A pretrain wrapper's ``encoder.<x>`` keys map onto the eval encoder's ``<x>``.

    Pretraining wraps the encoder (``SimMIMPretrainModel.encoder``), so its
    checkpoints carry an extra level the eval model does not have.
    """
    model_state = _eval_model().state_dict()
    checkpoint = {
        "encoder.class_agnostic_emb": model_state["class_agnostic_emb"].clone(),
        "encoder.pos_embed": model_state["pos_embed"].clone(),
    }

    fixed = fix_state_dict_keys(checkpoint, model_state)

    assert sorted(fixed) == ["class_agnostic_emb", "pos_embed"]


def test_the_encoders_own_transformer_keys_survive_unwrapping() -> None:
    """``encoder.layers.*`` is the transformer itself and must not be stripped.

    The eval model *has* an ``encoder`` submodule (the transformer), so the
    unwrapping rule has to distinguish "wrapper prefix" from "real attribute".
    This is the case that would silently orphan every attention weight if the
    rule were a blind ``removeprefix``.
    """
    model = _eval_model()
    model_state = model.state_dict()
    encoder_keys = [k for k in model_state if k.startswith("encoder.layers.")]
    assert encoder_keys, "the encoder's transformer keys should exist"

    checkpoint = {k: model_state[k].clone() for k in encoder_keys}
    fixed = fix_state_dict_keys(checkpoint, model_state)

    assert sorted(fixed) == sorted(encoder_keys)


def test_a_full_pretrain_state_dict_loads_every_eval_parameter(tmp_path) -> None:
    """The end-to-end property §7.2 protects: nothing is left randomly initialised.

    A real ``SimMIMPretrainModel`` state_dict is written to disk, read back
    through the live loader, key-fixed, and must cover every eval parameter —
    with the pretraining-only tensors dropped and nothing unexpected left.
    """
    from coffe.pretrain.simmim import SimMIMPretrainModel

    torch.manual_seed(0)
    encoder = _eval_model()
    pretrain = SimMIMPretrainModel(
        encoder=encoder,
        hsi_channels=144,
        aux_channels=1,
        patch_size=11,
        embed_dim=128,
        decoder_hidden_dim=256,
        band_mask_ratio=0.0,
        spatial_mask_ratio=0.75,
    )
    path = tmp_path / "checkpoint_epoch_2.pth"
    torch.save({"model_state_dict": pretrain.state_dict()}, path)

    state_dict, config = load_checkpoint_with_key_mapping(str(path), "cpu")
    assert config is None  # this checkpoint carries no config block

    # The key template comes from a *fresh* eval model — that is what the
    # evaluator does — while the value comparison below is against the encoder
    # that was actually saved, so equality is a real round-trip check and not a
    # comparison of two random initialisations.
    model_state = _eval_model().state_dict()
    fixed = fix_state_dict_keys(state_dict, model_state)

    missing = sorted(set(model_state) - set(fixed))
    assert missing == [], f"would be left randomly initialised: {missing}"
    unexpected = sorted(set(fixed) - set(model_state))
    assert unexpected == [], f"unexplained checkpoint keys: {unexpected}"

    saved = encoder.state_dict()
    assert sorted(fixed) == sorted(saved)
    for key, value in fixed.items():
        assert torch.equal(value, saved[key]), f"{key} did not survive the round trip"

    # And the two random initialisations really do differ, so the equality above
    # cannot be passing by accident.
    assert not all(torch.equal(saved[k], model_state[k]) for k in saved)


@pytest.mark.parametrize("wrapper_key", ["model_state_dict", "state_dict", "encoder_state_dict"])
def test_all_three_checkpoint_layouts_are_accepted(tmp_path, wrapper_key: str) -> None:
    """Checkpoints from the trainer, from ``save_encoder``, and bare dicts."""
    model_state = _eval_model().state_dict()
    path = tmp_path / f"{wrapper_key}.pth"
    torch.save({wrapper_key: model_state, "config": {"hsi_channels": 144}}, path)

    loaded, config = load_checkpoint_with_key_mapping(str(path), "cpu")

    assert sorted(loaded) == sorted(model_state)
    assert config == {"hsi_channels": 144}


def test_a_bare_state_dict_is_accepted_too(tmp_path) -> None:
    model_state = _eval_model().state_dict()
    path = tmp_path / "bare.pth"
    torch.save(model_state, path)

    loaded, config = load_checkpoint_with_key_mapping(str(path), "cpu")

    assert sorted(loaded) == sorted(model_state)
    assert config is None

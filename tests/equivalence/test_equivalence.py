"""The behaviour-equivalence tests every refactor phase must keep green.

These compare the *current* code against ``golden/*.json``, captured from the
pre-refactor code by ``make_golden.py``. A failure here means a refactor step
changed a computed number, which PAPER_CANON §7.1 forbids. **Never loosen a
tolerance to make one pass** — find what changed instead, and if the change
looks desirable, stop and report it rather than applying it.

Tolerances
----------
* eval assignments, realized mask rates, mask-ratio invariants — **exact**
* floats — ``rtol=1e-6, atol=1e-8`` (see ``conftest.RTOL`` / ``ATOL``)

Golden groups
-------------
G1 pretrain-loss trajectory | G2 encoder forward + init | G3 episodic eval
G4 checkpoint compatibility (the ``fixtures/`` checkpoints, asserted here
against G2/G3) | G5 masking semantics
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from ._harness import (
    FIXTURE_DIR,
    GOLDEN_DIR,
    HYPERSIGMA_REGIMES,
    MFT_OBJECTIVES,
    OBJECTIVES,
    SCENES,
    assignment_hash,
    build_all_scenes,
    build_eval_model,
    build_hypersigma_model,
    build_pretrain_model,
    episode_assignments,
    eval_params,
    fixed_input,
    live_eval_feature,
    load_eval_model_from_checkpoint,
    load_scene_dataset,
    oa_fingerprint,
    pretrain_config,
    set_determinism,
    state_dict_fingerprint,
    tensor_fingerprint,
)
from .conftest import ATOL, RTOL
from .make_golden import FIXTURES, fixture_path

pytestmark = pytest.mark.equivalence


def _golden(name: str) -> dict[str, Any]:
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(
            f"missing golden {path}. Regenerate with "
            f"`python tests/equivalence/make_golden.py` from a pre-refactor tree."
        )
    with path.open() as fh:
        return json.load(fh)


def assert_close(actual: float, expected: float, what: str) -> None:
    assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL), (
        f"{what}: {actual!r} != golden {expected!r} "
        f"(delta {actual - expected:+.3e}, rtol={RTOL}, atol={ATOL})"
    )


def assert_fingerprint(actual: dict[str, Any], expected: dict[str, Any], what: str) -> None:
    assert actual["shape"] == expected["shape"], f"{what}: shape changed"
    for stat in ("sum", "abs_sum", "norm"):
        assert_close(actual[stat], expected[stat], f"{what}.{stat}")


def assert_state_dict(actual: dict[str, Any], expected: dict[str, Any], what: str) -> None:
    # PAPER_CANON §7.2: nn.Module attribute names define checkpoint keys and are
    # frozen. A key set change fails here before any number is compared.
    assert set(actual) == set(expected), (
        f"{what}: state_dict keys changed. "
        f"missing={sorted(set(expected) - set(actual))} "
        f"unexpected={sorted(set(actual) - set(expected))}"
    )
    for key in sorted(expected):
        assert_fingerprint(actual[key], expected[key], f"{what}[{key}]")


# ----------------------------------------------------------------------
# Scene sanity — the harness's own inputs
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scene_name", sorted(SCENES))
def test_scene_shape_faithful(scene_root: Path, scene_name: str) -> None:
    """The mini-scenes go through the repo's dataset code and match PAPER_CANON §5."""
    spec = SCENES[scene_name]
    dataset = load_scene_dataset(spec, scene_root, split="all")

    assert len(dataset) == spec.num_samples
    assert dataset.hsi.shape == (spec.num_samples, spec.hsi_channels, 11, 11)
    assert dataset.aux.shape == (spec.num_samples, spec.aux_channels, 11, 11)
    assert dataset.hsi_channels == spec.hsi_channels
    assert dataset.aux_channels == spec.aux_channels
    assert dataset.num_classes == spec.num_classes
    # Labels are 1..N, as in the real .mat files (class 0 = background absent).
    assert sorted(dataset.class_indices) == list(range(1, spec.num_classes + 1))
    # Min-max normalisation to [0, 1] per band (PAPER_CANON §5).
    assert float(dataset.hsi.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(dataset.hsi.max()) == pytest.approx(1.0, abs=1e-6)


def test_coffe_param_count_matches_canon() -> None:
    """PAPER_CANON §2: the Houston eval encoder has 579,328 parameters (±1%)."""
    set_determinism()
    model = build_eval_model(SCENES["houston_mini"], model_name="coffe", use_aux=True)
    count = sum(p.numel() for p in model.parameters())
    assert count == pytest.approx(579_328, rel=0.01), count


# ----------------------------------------------------------------------
# G1 — pretrain-loss trajectory
# ----------------------------------------------------------------------


def _g1_cases():
    for objective_id in OBJECTIVES:
        for model_name in ("coffe", "mft_original"):
            if model_name == "mft_original" and objective_id not in MFT_OBJECTIVES:
                continue
            yield pytest.param(model_name, objective_id, id=f"{model_name}-{objective_id}")


@pytest.mark.parametrize("model_name,objective_id", list(_g1_cases()))
def test_g1_pretrain_loss_trajectory(
    scene_root: Path, tmp_path: Path, model_name: str, objective_id: str
) -> None:
    from coffe.pretrain.loop import run_pretrain

    golden = _golden("g1_pretrain_loss")
    key = f"{model_name}:{objective_id}"
    expected = golden["entries"][key]

    spec = SCENES[golden["scene"]]
    cfg = pretrain_config(spec, objective_id, data_root=scene_root, model_name=model_name)
    set_determinism()
    history = run_pretrain(cfg, str(tmp_path / "checkpoints"), str(tmp_path / "log"))

    assert history["epochs_run"] == expected["epochs_run"]
    assert len(history["train_losses"]) == len(expected["train_losses"])
    for epoch, (actual, want) in enumerate(
        zip(history["train_losses"], expected["train_losses"]), start=1
    ):
        assert_close(float(actual), want, f"G1[{key}] epoch {epoch} mean loss")


def test_g1_legacy_vocabulary_config_trains_identically() -> None:
    """A frozen-style config (model.name "mft_cpea", objective "enhanced") must
    produce the *same loss trajectory* as its canonical twin.

    The rest of the harness speaks canonical vocabulary, so without this the
    only thing pinning the compat layer end-to-end would be its unit tests
    (PAPER_CANON §7.3: frozen experiment trees must stay runnable).
    """
    import warnings

    from coffe import compat as coffe_compat
    from coffe.pretrain.loop import run_pretrain

    # The deprecation cache is process-global; clear it so this test does not
    # depend on which tests ran before it.
    coffe_compat.reset_deprecation_state()

    golden = _golden("g1_pretrain_loss")
    objective_id = "simmim_token"
    expected = golden["entries"][f"coffe:{objective_id}"]
    spec = SCENES[golden["scene"]]

    with tempfile.TemporaryDirectory(prefix="coffe-legacy-") as tmp:
        tmp_path = Path(tmp)
        scene_root = tmp_path / "raw"
        build_all_scenes(scene_root)

        cfg = pretrain_config(spec, objective_id, data_root=scene_root, model_name="coffe")
        # Rewrite exactly the two keys a frozen pretrain_config.yaml carries in
        # the retired vocabulary. Nothing else changes.
        cfg["model"]["name"] = "mft_cpea"
        cfg["pretrain"]["objective"] = "enhanced"

        set_determinism()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            history = run_pretrain(cfg, str(tmp_path / "checkpoints"), str(tmp_path / "log"))

    for epoch, (actual, want) in enumerate(
        zip(history["train_losses"], expected["train_losses"]), start=1
    ):
        assert_close(float(actual), want, f"G1[legacy-vocab] epoch {epoch} mean loss")

    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("mft_cpea" in m for m in messages), "no deprecation for the legacy model.name"
    assert any("enhanced" in m for m in messages), "no deprecation for the legacy objective"
    coffe_compat.reset_deprecation_state()


# ----------------------------------------------------------------------
# G2 — encoder forward + init
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,model_name,use_aux",
    [
        ("coffe_hsi_lidar", "coffe", True),
        ("coffe_hsi", "coffe", False),
        ("mft_original", "mft_original", True),
    ],
)
def test_g2_encoder_forward(key: str, model_name: str, use_aux: bool) -> None:
    golden = _golden("g2_encoder_forward")
    expected = golden["entries"][key]
    spec = SCENES[golden["scene"]]

    set_determinism()
    model = build_eval_model(spec, model_name=model_name, use_aux=use_aux)
    hsi, aux = fixed_input(spec)

    assert sum(p.numel() for p in model.parameters()) == expected["param_count"]
    assert_state_dict(
        state_dict_fingerprint(model.state_dict()), expected["state_dict"], f"G2[{key}]"
    )
    feature = live_eval_feature(model, hsi, aux if use_aux else None)
    assert_fingerprint(tensor_fingerprint(feature), expected["feature"], f"G2[{key}].z")


@pytest.mark.slow
@pytest.mark.parametrize("regime", sorted(HYPERSIGMA_REGIMES))
def test_g2_hypersigma_forward(tmp_path: Path, regime: str) -> None:
    """The HyperSIGMA wrapper's plumbing, with randomly initialised ViT bodies.

    Needs no released checkpoint (PAPER_CANON §7.5).
    """
    golden = _golden("g2_encoder_forward")
    expected = golden["entries"][f"hypersigma_{regime}"]
    spec = SCENES[golden["scene"]]

    set_determinism()
    model = build_hypersigma_model(spec, regime, tmp_path)
    hsi, _ = fixed_input(spec)

    assert sum(p.numel() for p in model.parameters()) == expected["param_count"]
    assert_state_dict(
        state_dict_fingerprint(model.state_dict()),
        expected["state_dict"],
        f"G2[hypersigma_{regime}]",
    )
    feature = live_eval_feature(model, hsi, None)
    assert_fingerprint(
        tensor_fingerprint(feature), expected["feature"], f"G2[hypersigma_{regime}].z"
    )


def test_dead_forward_episode_still_matches_live_path(scene_root: Path) -> None:
    """PAPER_CANON §8 D3: ``CoFFE.forward_episode`` is dead code — the
    live eval path is the inlined loop in ``coffe/eval/episodic.py:410-425``.

    The two are meant to be equivalent. This test records that they still are,
    so a phase that edits one and not the other is caught (and so that pruning
    ``forward_episode`` later can be shown to be behaviour-preserving).
    """
    spec = SCENES["houston_mini"]
    set_determinism()
    model = build_eval_model(spec, model_name="coffe", use_aux=True)
    model.distance_metric = "euclidean"

    n_way, k_shot, k_query = 4, 2, 3
    s_hsi, s_aux = fixed_input(spec, batch=n_way * k_shot, seed=11)
    q_hsi, q_aux = fixed_input(spec, batch=n_way * k_query, seed=12)
    s_labels = torch.arange(n_way).repeat_interleave(k_shot)

    with torch.no_grad():
        dead = model.forward_episode(s_hsi, s_aux, s_labels, q_hsi, q_aux)
        s_feat = live_eval_feature(model, s_hsi, s_aux)
        q_feat = live_eval_feature(model, q_hsi, q_aux)
        live = model._forward_mean_features(s_feat, s_labels, q_feat)

    torch.testing.assert_close(dead, live, rtol=RTOL, atol=ATOL)


# ----------------------------------------------------------------------
# G3 / G4 — episodic eval against the committed pre-refactor fixtures
# ----------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_g4_fixture_present_and_small(key: str) -> None:
    path = fixture_path(key)
    assert path.exists(), (
        f"missing checkpoint fixture {path}. It is committed; regenerate with "
        f"`make_golden.py --only g3` only from a pre-refactor tree."
    )
    total = sum(p.stat().st_size for p in FIXTURE_DIR.glob(f"*{path.suffix}"))
    assert total < 5 * 1024 * 1024, f"fixtures/ grew to {total / 1e6:.2f} MB (budget 5 MB)"


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_g3_episodic_eval(scene_root: Path, key: str) -> None:
    """The full paper eval path, scaled down, against a pre-refactor checkpoint.

    This is simultaneously the G4 checkpoint-compatibility test: the fixture is
    loaded through ``load_checkpoint_with_key_mapping`` + ``fix_state_dict_keys``
    (the live key-mapping code, PAPER_CANON §8 D15), so breaking state_dict keys
    or loader plumbing fails here.
    """
    from coffe.eval.episodic import run_evaluation

    golden = _golden("g3_episodic_eval")
    expected = golden["entries"][key]
    spec = SCENES[golden["scene"]]
    model_name = FIXTURES[key]["model_name"]

    params = eval_params(spec, data_root=scene_root, model_name=model_name)
    set_determinism()
    results = run_evaluation(checkpoint=str(fixture_path(key)), **params)

    assignments = episode_assignments(results)
    assert len(assignments[0]["original_classes"]) == expected["n_way"]

    # Exact: one flipped query prediction changes the digest.
    assert assignment_hash(assignments) == expected["assignment_hash"], (
        f"G3[{key}]: the per-episode per-query argmin assignment matrix changed"
    )

    actual_oa = oa_fingerprint(results)
    assert actual_oa["num_episodes"] == expected["OA"]["num_episodes"]
    for stat in ("mean", "std", "ci_95"):
        assert_close(actual_oa[stat], expected["OA"][stat], f"G3[{key}].OA.{stat}")


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_g4_fixture_forward(key: str) -> None:
    """Load each fixture into the current model class and forward the fixed input."""
    golden = _golden("g3_episodic_eval")
    expected = golden["entries"][key]["loaded_feature"]
    spec = SCENES[golden["scene"]]

    set_determinism()
    model = load_eval_model_from_checkpoint(
        spec,
        fixture_path(key),
        model_name=FIXTURES[key]["model_name"],
    )
    feature = live_eval_feature(model, *fixed_input(spec))
    assert_fingerprint(tensor_fingerprint(feature), expected, f"G4[{key}].z")


# Keys a *pretraining* checkpoint legitimately carries that the eval encoder does
# not, with the reason each is discarded. Every entry in the golden's
# `unexpected` list must be explained by one of these.
_EXPECTED_DISCARDS = {
    # Pretrain-only buffers/heads live on the pretraining wrapper, not the
    # encoder, so they have no `encoder.` prefix and no eval counterpart.
    "_recon_center_weights": "centre-weighted reconstruction buffer (pretrain only)",
    # PAPER_CANON §2: the projection head is discarded at eval
    # (`use_projection: False` makes `self.projection = nn.Identity()`).
    "encoder.projection.": "projection head, off at eval",
}


def _explain_discard(key: str) -> str | None:
    for prefix, reason in _EXPECTED_DISCARDS.items():
        if key == prefix or key.startswith(prefix):
            return reason
    return None


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_g4_fixture_keys_load_without_gaps(key: str) -> None:
    """No eval parameter may be left randomly initialised by a fixture load, and
    the set of intentionally-discarded checkpoint keys must not drift.

    This is the assertion that bites if someone renames an ``nn.Module``
    attribute without the key-map shim PAPER_CANON §7.2 requires: a renamed
    encoder tensor shows up as *missing* (the rename left a parameter unloaded)
    or as an unexplained *unexpected* key.
    """
    from coffe.eval.episodic import (
        fix_state_dict_keys,
        load_checkpoint_with_key_mapping,
    )

    expected = _golden("g3_episodic_eval")["entries"][key]["key_report"]
    spec = SCENES[_golden("g3_episodic_eval")["scene"]]

    set_determinism()
    model = build_eval_model(
        spec,
        model_name=FIXTURES[key]["model_name"],
        use_aux=True,
    )
    model_state = model.state_dict()
    state_dict, _ = load_checkpoint_with_key_mapping(str(fixture_path(key)), "cpu")
    fixed = fix_state_dict_keys(state_dict, model_state)

    missing = sorted(set(model_state) - set(fixed))
    assert not missing, (
        f"G4[{key}]: eval parameters absent from the checkpoint, so they would be "
        f"left randomly initialised: {missing}"
    )

    unexpected = sorted(set(fixed) - set(model_state))
    unexplained = [k for k in unexpected if _explain_discard(k) is None]
    assert not unexplained, (
        f"G4[{key}]: checkpoint keys the eval model does not have, for no "
        f"documented reason: {unexplained}"
    )
    assert unexpected == expected["unexpected"], (
        f"G4[{key}]: the set of discarded checkpoint keys changed"
    )
    assert missing == expected["missing"]
    assert len(state_dict) == expected["checkpoint_keys"]
    assert len(fixed) == expected["after_fix_keys"]
    assert len(model_state) == expected["model_keys"]

    for name, tensor in fixed.items():
        if name in model_state:
            assert tensor.shape == model_state[name].shape, (
                f"G4[{key}]: shape changed for {name}: "
                f"{tuple(tensor.shape)} vs {tuple(model_state[name].shape)}"
            )


# ----------------------------------------------------------------------
# G5 — masking semantics
# ----------------------------------------------------------------------


@pytest.mark.parametrize("objective_id", sorted(OBJECTIVES))
def test_g5_masking_semantics(objective_id: str) -> None:
    golden = _golden("g5_masking")
    expected = golden["entries"][objective_id]
    spec = SCENES[golden["scene"]]

    set_determinism()
    model = build_pretrain_model(spec, objective_id)
    model.eval()
    hsi, aux = fixed_input(
        spec, batch=golden["input"]["batch"], seed=golden["input"]["generator_seed"]
    )
    mask_seed = golden["input"]["mask_seed"]

    # Realized rates are compared EXACTLY at 4 decimals: masking is a
    # Bernoulli/argsort draw off the global RNG, and any change in how it is
    # drawn moves these.
    torch.manual_seed(mask_seed)
    if "band_mask_ratio_realized" in expected:
        _, band_mask = model.band_masking(torch.cat([hsi, aux], dim=1))
        assert round(float(band_mask.mean()), 4) == expected["band_mask_ratio_realized"]
    if "token_mask_ratio_realized" in expected:
        _, token_mask = model.spatial_masking(model.encoder.tokenize(hsi, aux))
        assert round(float(token_mask.mean()), 4) == expected["token_mask_ratio_realized"]

    torch.manual_seed(mask_seed)
    with torch.no_grad():
        loss, info = model(hsi, aux)

    assert round(float(info["mask"].mean()), 4) == expected["union_mask_ratio"], (
        f"G5[{objective_id}]: union mask rate changed"
    )
    assert_close(float(loss), expected["masked_loss"], f"G5[{objective_id}].masked_loss")
    assert_fingerprint(
        tensor_fingerprint(info["pred"]), expected["pred"], f"G5[{objective_id}].pred"
    )

    if "recon_center_weight_mean" in expected:
        # PAPER_CANON §3: centre weights have mean one. Exact by construction
        # (`rw / rw.mean()`), so a tight absolute tolerance is right here.
        actual = float(model._recon_center_weights.mean())
        assert actual == pytest.approx(1.0, abs=1e-6)
        assert_close(
            actual,
            expected["recon_center_weight_mean"],
            f"G5[{objective_id}].recon_center_weight_mean",
        )


def test_g5_union_is_maximum_of_both_masks() -> None:
    """Eq. 1 is scored on the UNION of the band and token masks.

    A structural check independent of the goldens: with both masks on, the union
    rate must be at least each individual rate and at most their sum.
    """
    spec = SCENES["houston_mini"]
    set_determinism()
    model = build_pretrain_model(spec, "simmim_band_token")
    model.eval()
    hsi, aux = fixed_input(spec, batch=8, seed=4242)

    torch.manual_seed(99)
    with torch.no_grad():
        _, info = model(hsi, aux)
    union = float(info["mask"].mean())

    band = OBJECTIVES["simmim_band_token"]["band_mask_ratio"]
    token = OBJECTIVES["simmim_band_token"]["spatial_mask_ratio"]
    assert max(band, token) - 0.02 <= union <= min(1.0, band + token) + 0.02, union
    assert set(torch.unique(info["mask"]).tolist()) <= {0.0, 1.0}


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def test_golden_metadata_recorded() -> None:
    meta = _golden("meta")
    for field in ("python", "torch", "numpy", "git_sha", "dirty"):
        assert field in meta, f"golden/meta.json is missing {field!r}"

    if meta["torch"] != torch.__version__:
        pytest.fail(
            f"goldens were generated with torch {meta['torch']}, this environment "
            f"has {torch.__version__}. A torch upgrade invalidates the "
            f"fingerprints; re-baseline deliberately from the `pre-refactor` tag "
            f"(see tests/equivalence/make_golden.py) rather than loosening "
            f"tolerances."
        )

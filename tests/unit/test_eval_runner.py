"""The eval runner's checkpoint lookup and architecture-from-config seeding.

``coffe.runners.eval_runner`` is how every paper evaluation was launched: it
reads the run's frozen ``pretrain_config.yaml`` to seed the architecture, finds
the checkpoint, and forwards to ``coffe.eval.episodic``. That seeding is what
makes PAPER_CANON §7.3 work in practice — a frozen config saying
``model.name: "mft_cpea"`` must still select the right model years later.

No dataset and no real checkpoint: the experiment trees here are synthetic
directories containing a YAML and empty ``.pth`` files, because both functions
under test only ever look at names and config keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from coffe.compat import reset_deprecation_state
from coffe.runners.eval_runner import (
    _arch_defaults_from_pretrain,
    find_checkpoint,
    load_pretrain_config,
)

#: The frozen vocabulary a pre-paper ``pretrain_config.yaml`` carries
#: (PAPER_CANON §7.3).
LEGACY_PRETRAIN_CONFIG = {
    "model": {
        "name": "mft_cpea",
        "embed_dim": 128,
        "num_heads": 2,
        "num_layers": 2,
        "lambda_factor": 0.5,
        "dropout": 0.1,
        "use_projection": False,
        "proj_hidden_dim": 512,
        "proj_num_layers": 1,
        "proj_l2_normalize": True,
        "use_aux": True,
    },
    "data": {"patch_size": 11},
    "pretrain": {"objective": "enhanced", "band_mask_ratio": 0.0, "spatial_mask_ratio": 0.75},
}


@pytest.fixture(autouse=True)
def _fresh_warning_state():
    """``coffe.compat`` warns once per (value, origin); reset between tests."""
    reset_deprecation_state()
    yield
    reset_deprecation_state()


def _experiment(root: Path, name: str, *, config: dict, epochs: tuple[int, ...] = ()) -> Path:
    """A synthetic ``experiments/<name>/`` with a config and empty checkpoints."""
    exp = root / name
    (exp / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp / "pretrain_config.yaml").write_text(yaml.safe_dump(config))
    for epoch in epochs:
        (exp / "checkpoints" / f"checkpoint_epoch_{epoch}.pth").write_bytes(b"")
    return exp


# ----------------------------------------------------------------------
# Architecture seeded from the frozen config
# ----------------------------------------------------------------------


def test_legacy_model_name_is_normalised_when_seeding_the_architecture() -> None:
    """PAPER_CANON §7.3: ``mft_cpea`` still resolves, with one DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="mft_cpea"):
        arch = _arch_defaults_from_pretrain(LEGACY_PRETRAIN_CONFIG)

    assert arch["name"] == "coffe"


def test_the_paper_architecture_is_read_off_the_frozen_config() -> None:
    """The keys that must survive the trip from pretrain config to evaluator.

    ``lambda_factor`` is among them, and it must arrive as 0.5 — the eval
    feature depends on it (PAPER_CANON §8 D3).
    """
    with pytest.warns(DeprecationWarning):
        arch = _arch_defaults_from_pretrain(LEGACY_PRETRAIN_CONFIG)

    assert arch["embed_dim"] == 128
    assert arch["num_heads"] == 2
    assert arch["num_layers"] == 2
    assert arch["lambda_factor"] == 0.5
    assert arch["use_projection"] is False
    assert arch["use_aux"] is True
    assert arch["patch_size"] == 11  # read from data:, not model:


def test_the_mft_control_arch_survives_the_trip() -> None:
    config = {
        "model": {
            "name": "mft_original",
            "embed_dim": 64,
            "num_heads": 8,
            "num_layers": 2,
            "mlp_dim": 512,
            "attention_type": "mcross",
            "use_projection": False,
        },
        "data": {"patch_size": 11},
    }

    arch = _arch_defaults_from_pretrain(config)

    assert arch["name"] == "mft_original"
    assert arch["mlp_dim"] == 512
    assert arch["attention_type"] == "mcross"


def test_non_architecture_keys_are_not_forwarded() -> None:
    """Mask rates and schedules belong to pretraining, not to the evaluator.

    Forwarding them would put unknown keys on the evaluator's namespace; the
    whitelist is what stops that.
    """
    with pytest.warns(DeprecationWarning):
        arch = _arch_defaults_from_pretrain(LEGACY_PRETRAIN_CONFIG)

    for leaked in ("band_mask_ratio", "spatial_mask_ratio", "objective", "epochs"):
        assert leaked not in arch


def test_an_empty_config_seeds_nothing_rather_than_guessing() -> None:
    assert _arch_defaults_from_pretrain({}) == {}
    assert _arch_defaults_from_pretrain({"model": None, "data": None}) == {}


def test_load_pretrain_config_reads_the_frozen_file(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _experiment(root, "houston_enhanced_spatial_run1", config=LEGACY_PRETRAIN_CONFIG)

    config = load_pretrain_config("houston_enhanced_spatial_run1", experiments_root=root)

    # Read verbatim: normalisation happens in _arch_defaults_from_pretrain, so
    # the loader must not quietly rewrite the frozen artifact's vocabulary.
    assert config["model"]["name"] == "mft_cpea"
    assert config["pretrain"]["objective"] == "enhanced"


def test_a_missing_pretrain_config_is_an_error_not_a_default(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    (root / "bare_run").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"No pretrain_config\.yaml"):
        load_pretrain_config("bare_run", experiments_root=root)


# ----------------------------------------------------------------------
# Checkpoint lookup
# ----------------------------------------------------------------------


def test_an_explicit_epoch_selects_that_checkpoint(tmp_path: Path) -> None:
    """The paper path: every published cell names its epoch (950 / 975 / 800).

    PAPER_CANON §8 D17 — the evaluated checkpoint is never the final one, so an
    explicit epoch is the normal case, not an override.
    """
    root = tmp_path / "experiments"
    _experiment(root, "run", config=LEGACY_PRETRAIN_CONFIG, epochs=(800, 950, 1500))

    found = find_checkpoint("run", epoch=950, experiments_root=root)

    assert Path(found).name == "checkpoint_epoch_950.pth"


def test_a_missing_epoch_is_an_error_not_a_silent_fallback(tmp_path: Path) -> None:
    """Asking for an epoch that was never saved must fail, not evaluate another."""
    root = tmp_path / "experiments"
    _experiment(root, "run", config=LEGACY_PRETRAIN_CONFIG, epochs=(800, 950))

    with pytest.raises(FileNotFoundError, match="Checkpoint for epoch 975 not found"):
        find_checkpoint("run", epoch=975, experiments_root=root)


def test_checkpoint_final_wins_when_no_epoch_is_requested(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    exp = _experiment(root, "run", config=LEGACY_PRETRAIN_CONFIG, epochs=(800, 950))
    (exp / "checkpoints" / "checkpoint_final.pth").write_bytes(b"")

    found = find_checkpoint("run", epoch=None, experiments_root=root)

    assert Path(found).name == "checkpoint_final.pth"


def test_an_experiment_without_checkpoints_is_an_error(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _experiment(root, "run", config=LEGACY_PRETRAIN_CONFIG)

    with pytest.raises(FileNotFoundError, match="No checkpoint files found"):
        find_checkpoint("run", epoch=None, experiments_root=root)


def test_an_unknown_experiment_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        find_checkpoint("nope", experiments_root=tmp_path / "experiments")


@pytest.mark.xfail(
    reason=(
        "KNOWN DEFECT (found in phase 7, reported at the gate): with epoch=None "
        "and no checkpoint_final.pth, find_checkpoint takes sorted(glob(...))[-1], "
        "which orders filenames lexicographically — 'checkpoint_epoch_1500.pth' < "
        "'checkpoint_epoch_950.pth' — so it returns epoch 950 instead of 1500. "
        "No published number is affected: every paper cell passes an explicit "
        "epoch (PAPER_CANON §8 D17), and the significance runner passes 700. "
        "Not fixed here because that changes behaviour; this test states the "
        "intended contract and will XPASS once the sort is made numeric."
    ),
    strict=False,
)
def test_latest_checkpoint_should_be_the_highest_epoch(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _experiment(root, "run", config=LEGACY_PRETRAIN_CONFIG, epochs=(800, 950, 1500))

    found = find_checkpoint("run", epoch=None, experiments_root=root)

    assert Path(found).name == "checkpoint_epoch_1500.pth"


def test_the_current_latest_checkpoint_behaviour_is_lexicographic(tmp_path: Path) -> None:
    """The companion to the xfail above: what actually happens today.

    Pinned so the defect is visible in the suite rather than folklore, and so
    the pair has to be updated together when it is fixed.
    """
    root = tmp_path / "experiments"
    _experiment(root, "run", config=LEGACY_PRETRAIN_CONFIG, epochs=(800, 950, 1500))

    found = find_checkpoint("run", epoch=None, experiments_root=root)

    assert Path(found).name == "checkpoint_epoch_950.pth"


def test_a_single_epoch_is_unaffected_by_the_ordering_defect(tmp_path: Path) -> None:
    """The common case — one checkpoint per run dir — is correct either way."""
    root = tmp_path / "experiments"
    _experiment(root, "run", config=LEGACY_PRETRAIN_CONFIG, epochs=(700,))

    found = find_checkpoint("run", epoch=None, experiments_root=root)

    assert Path(found).name == "checkpoint_epoch_700.pth"


def test_eval_params_must_name_a_dataset(tmp_path: Path) -> None:
    """A runner call without ``dataset`` fails before anything is written."""
    from coffe.runners.eval_runner import run_evaluation

    with pytest.raises(ValueError, match="must include 'dataset'"):
        run_evaluation("run", "eval", eval_params={}, experiments_root=tmp_path / "experiments")

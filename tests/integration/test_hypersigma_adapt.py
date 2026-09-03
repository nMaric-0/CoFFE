"""The label-free HyperSIGMA adaptation loop, end to end.

PAPER_CANON §1 describes the HyperSIGMA route as "label-free continued masked
reconstruction from released checkpoints (75 % token masking, per-patch
z-scored targets)", and §3 times it at ~36 h for 2000 epochs. Every *adapted*
Table 3 cell came out of ``coffe.pretrain.hypersigma_adapt.run_adapt``, which
had no coverage before phase 7.

One epoch on a synthetic mini-scene with randomly initialised ViT bodies (no
released checkpoint, CLAUDE.md hard rule 5). What is verified: the loop runs,
it writes the checkpoints the eval step expects, no label is ever consulted,
and — the invariant that matters — the released bodies receive no gradient.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from tests.conftest import CANON_DATASET_SPECS, spec_for
from tests.equivalence._harness import write_scene

pytestmark = pytest.mark.slow

SCENE = "trento"
MASK_RATIO = 0.75  # PAPER_CANON §1
BATCH_SIZE = 16

#: Prefixes of the released ViT-Base bodies (see
#: ``tests/unit/test_hypersigma_contracts.py``).
FROZEN_BODY_PREFIXES = ("dual.spat.model.blocks.", "dual.spec.model.blocks.")


@pytest.fixture(scope="module")
def scene_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("hs_adapt") / "raw"
    write_scene(spec_for(SCENE), root)
    return root


@pytest.fixture(scope="module")
def pca_path(tmp_path_factory) -> str:
    bands = CANON_DATASET_SPECS[SCENE]["hsi_channels"]
    rng = np.random.default_rng(0)
    fitted = PCA(n_components=3, svd_solver="full").fit(
        rng.standard_normal((2048, bands)).astype(np.float32)
    )
    path = tmp_path_factory.mktemp("hs_adapt_pca") / "pca.pkl"
    with path.open("wb") as fh:
        pickle.dump(fitted, fh)
    return str(path)


def _config(scene_root: Path, pca_path: str, adapt_mode: str, epochs: int = 1) -> dict:
    """An adapt config mirroring ``configs/hypersigma/*.yaml``, schedule aside.

    Only ``epochs``/``batch_size`` are scaled down; ``mask_ratio`` and the
    decoder widths are the published ones.
    """
    return {
        "model": {"adapt_mode": adapt_mode, "spat_patch_k": 3},
        "data": {
            "dataset": SCENE,
            "data_root": str(scene_root),
            "patch_size": 11,
            "num_workers": 0,
        },
        "paths": {
            "pca_spat_path": pca_path,
            "pca_stats_path": None,
            "spat_ckpt": None,
            "spec_ckpt": None,
        },
        "pretrain": {
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "val_split": 0.2,
            "lr": 1.0e-4,
            "min_lr": 1.0e-6,
            "weight_decay": 0.05,
            "warmup_epochs": 0,
            "grad_clip": 1.0,
            "mask_ratio": MASK_RATIO,
            "spat_decoder_hidden": 256,
            "spec_decoder_hidden": 256,
            "fused_decoder_hidden": 512,
            "decoder_dropout": 0.0,
            "save_interval": 1,
            "val_interval": 1,
            "log_interval": 1,
            "use_amp": False,
        },
        "hardware": {"device": "cpu", "seed": 42, "deterministic": False},
    }


def test_one_adaptation_epoch_runs_and_writes_its_checkpoints(
    scene_root: Path, pca_path: str, tmp_path: Path
) -> None:
    """The loop completes and leaves the artifacts the eval step consumes."""
    from coffe.pretrain.hypersigma_adapt import run_adapt

    checkpoint_dir = tmp_path / "ckpt"
    summary = run_adapt(
        _config(scene_root, pca_path, "sem_only"),
        str(checkpoint_dir),
        str(tmp_path / "logs"),
    )

    assert summary["adapt_mode"] == "sem_only"
    assert summary["mask_ratio"] == MASK_RATIO
    assert summary["resolved_device"] == "cpu"
    assert summary["epochs_run"] == 1
    assert np.isfinite(summary["final_train_loss"])
    assert np.isfinite(summary["best_val_loss"])

    written = {path.name for path in checkpoint_dir.glob("*.pth")}
    assert "checkpoint_final.pth" in written
    assert "checkpoint_epoch_1.pth" in written
    # `checkpoint.pth` is the best-val snapshot the eval CLI defaults to.
    assert "checkpoint.pth" in written


@pytest.mark.parametrize("adapt_mode", ["spatial_only", "spectral_only", "joint_sem"])
def test_every_adaptation_mode_completes_an_epoch(
    scene_root: Path, pca_path: str, tmp_path: Path, adapt_mode: str
) -> None:
    """PAPER_CANON §1's ``spatial`` / ``spectral`` / ``joint_sem`` all train.

    Each mode builds different decoders and returns a different subset of the
    three losses, so running one is not evidence for the others.
    """
    from coffe.pretrain.hypersigma_adapt import run_adapt

    summary = run_adapt(
        _config(scene_root, pca_path, adapt_mode),
        str(tmp_path / f"ckpt_{adapt_mode}"),
        str(tmp_path / f"logs_{adapt_mode}"),
    )

    assert summary["adapt_mode"] == adapt_mode
    assert np.isfinite(summary["final_train_loss"])
    assert summary["param_counts"]["total_trainable"] > 0


def test_the_released_bodies_receive_no_gradient_during_a_training_step(
    scene_root: Path, pca_path: str
) -> None:
    """The label-free claim's teeth: a real backward leaves the bodies untouched.

    ``requires_grad`` is asserted statically in
    ``tests/unit/test_hypersigma_contracts.py``; this runs an actual forward +
    backward through the adaptation and checks that no released block weight
    ends up with a gradient — which is what would happen if a hook or a
    reparameterisation reconnected them.
    """
    from coffe.pretrain.hypersigma_adapt import _build_dataset, _build_model

    config = _config(scene_root, pca_path, "joint_sem")
    dataset, dataset_name = _build_dataset(config)
    model = _build_model(config, dataset_name)
    model.train()

    batch = torch.stack([dataset[i]["hsi"] for i in range(4)])
    out = model(batch)
    # ``out["loss"]`` is the graph-attached total the training loop backprops;
    # the per-branch ``loss_*`` entries are detached reporting copies.
    assert out["loss"].requires_grad
    out["loss"].backward()

    leaked = [
        name
        for name, param in model.named_parameters()
        if name.startswith(FROZEN_BODY_PREFIXES) and param.grad is not None
    ]
    assert leaked == [], f"released body weights received gradients: {leaked[:5]}"

    # And the pieces that are supposed to train did get gradients.
    trained = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert trained, "nothing received a gradient — the step did no work"


def test_adaptation_never_reads_a_label(scene_root: Path, pca_path: str) -> None:
    """ "Label-free" (PAPER_CANON §1): the forward takes only the HSI patch.

    Checked on the signature and on a labelless call: ``forward(hsi)`` is the
    whole interface, so no label can enter the objective.
    """
    import inspect

    from coffe.pretrain.hypersigma_adapt import _build_dataset, _build_model
    from coffe.pretrain.hypersigma_mae import HyperSIGMAMaskedAdaptation

    parameters = inspect.signature(HyperSIGMAMaskedAdaptation.forward).parameters
    assert list(parameters) == ["self", "hsi"]

    config = _config(scene_root, pca_path, "sem_only")
    dataset, dataset_name = _build_dataset(config)
    model = _build_model(config, dataset_name).eval()

    with torch.no_grad():
        out = model(torch.stack([dataset[i]["hsi"] for i in range(2)]))

    assert any(key.startswith("loss") for key in out)


def test_a_band_count_override_that_contradicts_the_dataset_is_rejected(
    scene_root: Path, pca_path: str
) -> None:
    """A wrong ``hsi_channels`` must fail loudly, not build a mis-sized model."""
    from coffe.pretrain.hypersigma_adapt import _resolve_hsi_channels

    with pytest.raises(ValueError, match="does not match dataset"):
        _resolve_hsi_channels({"hsi_channels": 144}, SCENE)  # Trento has 63

    assert _resolve_hsi_channels({}, SCENE) == CANON_DATASET_SPECS[SCENE]["hsi_channels"]

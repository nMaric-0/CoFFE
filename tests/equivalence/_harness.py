"""Shared machinery for the behaviour-equivalence harness.

Everything here is *scaffolding*: it builds synthetic scenes, assembles configs
and turns tensors into comparable fingerprints. It deliberately contains **no
copy of the repository's preprocessing, masking, training or evaluation logic** —
those are always reached by calling the real entry points:

* pretraining  -> ``coffe.pretrain.loop.run_pretrain``
  (the same implementation ``coffe/runners/pretrain_runner.run_pretrain`` wraps)
* evaluation   -> ``coffe.eval.episodic.run_evaluation``
  (the same implementation ``coffe/runners/eval_runner.run_evaluation`` wraps)
* datasets     -> ``coffe.eval.episodic.DATASETS`` (the repo's own map onto
  ``coffe.data.datasets.*PatchedDataset``), fed synthetic ``.mat`` files in the exact
  on-disk layout the real scenes use, so patch handling and min-max
  normalisation run the repo's code.

Synthetic scenes are written as ``.mat`` files rather than constructed as
tensors precisely so that ``PatchedMultimodalDataset._load_split`` /
``_extract_array`` / ``_ensure_nchw`` / ``_ensure_1d`` / ``_normalize`` /
``_build_class_indices`` all execute unmodified.

The one place a *sequence* of repo calls is reproduced here is
:func:`live_eval_feature`, which mirrors the feature lines of
``coffe/eval/episodic.py:410-425``. That is the LIVE eval feature path
(PAPER_CANON §8 D3: ``CoFFE.forward_episode`` is dead code and must not
be what the harness pins). G3 exercises the real evaluator end-to-end, so the
mirror is only used for the G2 encoder-forward fingerprints; a dedicated test
asserts the mirror and the dead ``forward_episode`` still agree.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# The checkpoint fixtures are committed despite the blanket `*.pth` rule in
# .gitignore: phase 5 added an explicit `!/tests/equivalence/fixtures/*.pth`
# negation, so they carry their real extension again (they were named
# `*.pth.fixture` in phases 2-4, when .gitignore was out of scope).
FIXTURE_SUFFIX = ".pth"

PATCH_SIZE = 11


# ----------------------------------------------------------------------
# Synthetic scenes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SceneSpec:
    """A mini-scene that is shape-faithful to a real scene (PAPER_CANON §5).

    ``folder`` is an on-disk-name literal mirroring the real dataset layout and
    is therefore DO-NOT-RENAME (PAPER_CANON §7.3).
    """

    name: str
    key: str  # dataset key accepted by the repo's DATASETS maps
    folder: str
    hsi_channels: int
    aux_channels: int
    num_classes: int
    per_class_train: int
    per_class_test: int
    seed: int

    @property
    def num_samples(self) -> int:
        return self.num_classes * (self.per_class_train + self.per_class_test)


SCENES: dict[str, SceneSpec] = {
    # ~30 px/class, 11x11 patches. Sized to be fast but shape-faithful.
    "houston_mini": SceneSpec(
        name="houston_mini",
        key="houston",
        folder="Houston11x11",
        hsi_channels=144,
        aux_channels=1,
        num_classes=15,
        per_class_train=10,
        per_class_test=20,
        seed=20260231,
    ),
    "trento_mini": SceneSpec(
        name="trento_mini",
        key="trento",
        folder="Trento11x11",
        hsi_channels=63,
        aux_channels=1,
        num_classes=6,
        per_class_train=10,
        per_class_test=20,
        seed=20260232,
    ),
    "muufl_mini": SceneSpec(
        name="muufl_mini",
        key="muufl",
        folder="MUUFL11x11",
        hsi_channels=64,
        aux_channels=2,
        num_classes=11,
        per_class_train=10,
        per_class_test=20,
        seed=20260233,
    ),
}

# Class separation vs. within-class noise. Every class shares one base spectrum
# and differs from it by a small per-band offset, so the classes genuinely
# overlap: nearest-class-mean accuracy on the mini-scenes lands well inside
# (chance, 100%). A golden pinned at either extreme would barely notice a
# degraded encoder. `_CLASS_SPREAD` is the knob: a sweep over 0.005-0.09 put
# CoFFE at 33/82/95% and MFTOriginal at 12/19/26% for 0.02/0.04/0.06 (chance is
# 1/15 = 6.7%), so 0.04 keeps both encoders clear of chance and of saturation.
_CLASS_SPREAD = 0.04
_HSI_NOISE = 0.35
_AUX_NOISE = 0.25
_BUMP_AMPLITUDE = 0.25


def _center_bump(patch_size: int = PATCH_SIZE, sigma: float = 2.5) -> np.ndarray:
    """Gaussian bump centred on the patch centre, peak 1.0, shape [P, P].

    Only the centre pixel of a real patch carries the label, so the synthetic
    class signal is concentrated there. This keeps the centre-weighted
    reconstruction loss and centre-weighted pooling options meaningful.
    """
    c = patch_size // 2
    coords = np.arange(patch_size, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma**2))


def write_scene(spec: SceneSpec, data_root: Path) -> Path:
    """Write a synthetic scene as ``.mat`` files under ``data_root/<folder>/``.

    The layout, dtypes and axis order match the real pre-patched scenes exactly
    (verified against ``data/raw/*``): HSI ``[N, 11, 11, C]`` float64, LiDAR
    ``[N, 11, 11, C_aux]`` float64, labels ``[1, N]`` int64 valued 1..num_classes
    (class 0 = background is absent, as in the real files).
    """
    import scipy.io as sio

    base = Path(data_root) / spec.folder
    base.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    n_cls = spec.num_classes
    base_hsi = rng.uniform(0.30, 0.70, size=spec.hsi_channels)
    base_aux = rng.uniform(0.30, 0.70, size=spec.aux_channels)
    class_hsi = base_hsi[None, :] + rng.normal(0.0, _CLASS_SPREAD, size=(n_cls, spec.hsi_channels))
    class_aux = base_aux[None, :] + rng.normal(0.0, _CLASS_SPREAD, size=(n_cls, spec.aux_channels))
    bump = _center_bump()[None, :, :, None]  # [1, P, P, 1]

    for suffix, n_per in (("Tr", spec.per_class_train), ("Te", spec.per_class_test)):
        hsi_parts, aux_parts, labels = [], [], []
        for c in range(n_cls):
            shape_h = (n_per, PATCH_SIZE, PATCH_SIZE, spec.hsi_channels)
            shape_a = (n_per, PATCH_SIZE, PATCH_SIZE, spec.aux_channels)
            hsi = (
                class_hsi[c][None, None, None, :]
                + _BUMP_AMPLITUDE * bump * class_hsi[c][None, None, None, :]
                + rng.normal(0.0, _HSI_NOISE, size=shape_h)
            )
            aux = (
                class_aux[c][None, None, None, :]
                + _BUMP_AMPLITUDE * bump * class_aux[c][None, None, None, :]
                + rng.normal(0.0, _AUX_NOISE, size=shape_a)
            )
            hsi_parts.append(hsi)
            aux_parts.append(aux)
            labels.extend([c + 1] * n_per)

        sio.savemat(base / f"HSI_{suffix}.mat", {"Data": np.concatenate(hsi_parts, axis=0)})
        sio.savemat(base / f"LIDAR_{suffix}.mat", {"Data": np.concatenate(aux_parts, axis=0)})
        sio.savemat(
            base / f"{suffix}Label.mat",
            {"Data": np.asarray(labels, dtype=np.int64)[None, :]},
        )

    return Path(data_root)


def build_all_scenes(data_root: Path) -> Path:
    for spec in SCENES.values():
        write_scene(spec, data_root)
    return Path(data_root)


def load_scene_dataset(spec: SceneSpec, data_root: Path, split: str = "all"):
    """Load a mini-scene through the repository's own dataset class."""
    from coffe.eval.episodic import DATASETS  # the repo's own map

    return DATASETS[spec.key](
        data_root=str(data_root),
        patch_size=PATCH_SIZE,
        split=split,
        normalize=True,
    )


# ----------------------------------------------------------------------
# Configs — mirroring the frozen paper run configs (PAPER_CANON §8 D13:
# the reproducible recipe lives in experiments/*/pretrain_config.yaml, not in
# configs/). Only the schedule length is scaled down; every semantic knob
# (arch, mask rates, loss, optimiser) is the paper's.
# ----------------------------------------------------------------------

# CoFFE encoder arch, from experiments/houston_enhanced_spatial_mask_test_run1_seed52
# (the headline Houston run, PAPER_CANON §8 D14) — and PAPER_CANON §2.
COFFE_ARCH: dict[str, Any] = {
    "embed_dim": 128,
    "num_heads": 2,
    "num_layers": 2,
    "lambda_factor": 0.5,
    "dropout": 0.1,
    "proj_hidden_dim": 512,
    "proj_num_layers": 1,
    "proj_l2_normalize": True,
}

# MFTOriginal arch, from experiments/mft_original_houston_spatial_faithful.
MFT_ARCH: dict[str, Any] = {
    "embed_dim": 64,
    "num_heads": 8,
    "num_layers": 2,
    "mlp_dim": 512,
    "dropout": 0.1,
    "attention_type": "mcross",
}

# Objective table, in PAPER_CANON §1 canonical vocabulary throughout (phase 4
# renamed the config values to match the ids: objective "simmim" | "mae", with
# the band/spatial mask-rate pair naming the regime). Frozen configs spell the
# SimMIM objective "enhanced"; `coffe.compat` maps it, and
# `tests/test_compat.py` pins that path.
OBJECTIVES: dict[str, dict[str, Any]] = {
    "simmim_band": {
        "objective": "simmim",
        "band_mask_ratio": 0.85,
        "spatial_mask_ratio": 0.0,
        "recon_center_sigma": 1.0,
        "decoder_hidden_dim": 256,
        "use_projection": True,
    },
    "simmim_token": {
        "objective": "simmim",
        "band_mask_ratio": 0.0,
        "spatial_mask_ratio": 0.75,
        "recon_center_sigma": 1.0,
        "decoder_hidden_dim": 256,
        "use_projection": True,
    },
    "simmim_band_token": {
        "objective": "simmim",
        "band_mask_ratio": 0.85,
        "spatial_mask_ratio": 0.75,
        "recon_center_sigma": 1.0,
        "decoder_hidden_dim": 256,
        "use_projection": True,
    },
    # PAPER_CANON §1 states band+token = (0.85, 0.75), and five of the six
    # canonical band+token runs used that. The Houston HSI+LiDAR cell
    # (`houston_enhanced_spec_spat_combined`, Table 2 = 64.63) used
    # band_mask_ratio 0.75. Pinned separately so the harness covers the rate
    # that actually produced a paper number. Reported as D19.
    "simmim_band_token_houston_run": {
        "objective": "simmim",
        "band_mask_ratio": 0.75,
        "spatial_mask_ratio": 0.75,
        "recon_center_sigma": 1.0,
        "decoder_hidden_dim": 256,
        "use_projection": True,
    },
    "mae": {
        "objective": "mae",
        "mask_ratio": 0.75,
        "decoder_dim": 64,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "norm_pix_loss": True,
        "recon_center_sigma": None,
        # The MAE recipe has no projection head; run_pretrain forces this off
        # anyway (coffe/pretrain/loop.py:309-315).
        "use_projection": False,
    },
}

# MFTOriginal supports only these two objectives
# (coffe/pretrain/loop.py:319-326).
MFT_OBJECTIVES = ("simmim_token", "mae")

G1_EPOCHS = 3
G1_BATCH_SIZE = 16
# epochs=3 forces a short warmup: CosineAnnealingLR gets T_max = epochs - warmup
# and must stay positive. The paper runs use warmup_epochs 80-100 over 1500-3000
# epochs; scaling the schedule is the only concession the harness makes.
G1_WARMUP_EPOCHS = 1


def pretrain_config(
    spec: SceneSpec,
    objective_id: str,
    *,
    data_root: Path,
    model_name: str = "coffe",
    use_aux: bool = True,
    epochs: int = G1_EPOCHS,
    batch_size: int = G1_BATCH_SIZE,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a `run_pretrain` config dict for one (scene, objective) pair."""
    if objective_id not in OBJECTIVES:
        raise KeyError(f"unknown objective id {objective_id!r}")
    if model_name == "mft_original" and objective_id not in MFT_OBJECTIVES:
        raise ValueError(
            f"model.name='mft_original' supports {MFT_OBJECTIVES}, got {objective_id!r}"
        )

    obj = dict(OBJECTIVES[objective_id])
    use_projection = obj.pop("use_projection")

    arch = dict(MFT_ARCH) if model_name == "mft_original" else dict(COFFE_ARCH)
    model_cfg: dict[str, Any] = {"name": model_name, **arch, "use_aux": use_aux}
    if model_name == "mft_original":
        model_cfg["use_projection"] = False
    else:
        model_cfg["use_projection"] = use_projection

    return {
        "model": model_cfg,
        "data": {"patch_size": PATCH_SIZE, "num_workers": 0, "pin_memory": False},
        "pretrain": {
            "datasets": [spec.key],
            "epochs": epochs,
            "batch_size": batch_size,
            "val_split": 0.0,
            "lr": 1.5e-4,
            "min_lr": 1.0e-6,
            "weight_decay": 0.05,
            "warmup_epochs": G1_WARMUP_EPOCHS,
            "grad_clip": 1.0,
            "save_interval": 10_000,
            "val_interval": 50_000,
            "log_interval": 10_000,
            "use_amp": False,
            **obj,
        },
        "paths": {"data_root": str(data_root)},
        "hardware": {"device": "cpu", "seed": seed, "deterministic": True},
    }


# Episodic-eval protocol, scaled down from PAPER_CANON §4 / the frozen
# eval_config.json of the headline run. Scaled: k_query 100 -> 10, episodes
# 1000 -> 20. Law-bound and unchanged: N-way full-class, K=5, euclidean,
# use_projection False, pool_sigma None, prototype_mode mean_features.
G3_K_SHOT = 5
G3_K_QUERY = 10
G3_EPISODES = 20
G3_SEED = 42


def eval_params(
    spec: SceneSpec,
    *,
    data_root: Path,
    model_name: str = "coffe",
    use_aux: bool = True,
    num_episodes: int = G3_EPISODES,
    k_shot: int = G3_K_SHOT,
    k_query: int = G3_K_QUERY,
    seed: int = G3_SEED,
) -> dict[str, Any]:
    """Keyword arguments for ``coffe.eval.episodic.run_evaluation``."""
    arch = dict(MFT_ARCH) if model_name == "mft_original" else dict(COFFE_ARCH)
    arch.pop("proj_hidden_dim", None)
    arch.pop("proj_num_layers", None)
    arch.pop("proj_l2_normalize", None)

    params: dict[str, Any] = {
        "dataset": spec.key,
        "name": model_name,
        "data_root": str(data_root),
        "split": "all",
        "patch_size": PATCH_SIZE,
        "n_way": None,  # N-way full-class
        "k_shot": k_shot,
        "k_query": k_query,
        "num_episodes": num_episodes,
        "distance_metric": "euclidean",
        "temperature": 10.0,
        "prototype_mode": "mean_features",
        "use_projection": False,
        "use_aux": use_aux,
        "pool_sigma": None,
        "seed": seed,
        "device": "cpu",
        "no_plots": True,
        # Capture every episode's assignment, not just the plotting sample.
        # Pure collection flag: it touches no RNG and no arithmetic
        # (coffe/eval/episodic.py:464-476).
        "num_example_episodes": num_episodes,
        "max_tsne_samples": 0,
        "output": None,
        **arch,
    }
    if model_name != "mft_original":
        params.update(
            proj_hidden_dim=COFFE_ARCH["proj_hidden_dim"],
            proj_num_layers=COFFE_ARCH["proj_num_layers"],
            proj_l2_normalize=COFFE_ARCH["proj_l2_normalize"],
        )
    return params


# ----------------------------------------------------------------------
# Model construction for the G2 / G5 fingerprints
# ----------------------------------------------------------------------


def build_eval_model(
    spec: SceneSpec,
    *,
    model_name: str = "coffe",
    use_aux: bool = True,
    use_projection: bool = False,
):
    """Build the eval-shaped encoder through the repository's own loader.

    Goes through ``coffe.eval.episodic.load_model_with_checkpoint`` with
    checkpoint ``"random"`` so the harness pins the real construction path
    (including its per-architecture defaults), not a private copy of it.

    ``use_projection`` defaults to False — the paper's eval setting
    (PAPER_CANON §4). :func:`build_pretrain_model` overrides it to the
    *pretraining* value, because the projection head is inside the masked
    reconstruction forward (``coffe/pretrain/simmim.py``).
    """
    from coffe.eval.episodic import load_model_with_checkpoint

    params = eval_params(spec, data_root=Path("."), model_name=model_name, use_aux=use_aux)
    params["use_projection"] = use_projection
    model_config = {
        k: params[k]
        for k in (
            "name",
            "embed_dim",
            "num_heads",
            "num_layers",
            "patch_size",
            "dropout",
            "distance_metric",
            "temperature",
            "prototype_mode",
            "use_projection",
            "use_aux",
            "pool_sigma",
        )
        if k in params
    }
    for k in (
        "lambda_factor",
        "mlp_dim",
        "attention_type",
        "proj_hidden_dim",
        "proj_num_layers",
        "proj_l2_normalize",
    ):
        if k in params:
            model_config[k] = params[k]

    return load_model_with_checkpoint("random", spec.key, model_config, "cpu")


def load_eval_model_from_checkpoint(
    spec: SceneSpec,
    checkpoint: Path,
    *,
    model_name: str = "coffe",
    use_aux: bool = True,
):
    """Same as :func:`build_eval_model` but loading a real checkpoint file.

    This is the G4 path: ``load_checkpoint_with_key_mapping`` +
    ``fix_state_dict_keys`` (coffe/eval/episodic.py:96-157) are the live
    key-mapping implementation (PAPER_CANON §8 D15).
    """
    from coffe.eval.episodic import load_model_with_checkpoint

    params = eval_params(spec, data_root=Path("."), model_name=model_name, use_aux=use_aux)
    model_config = {
        k: v
        for k, v in params.items()
        if k
        in {
            "name",
            "embed_dim",
            "num_heads",
            "num_layers",
            "patch_size",
            "dropout",
            "distance_metric",
            "temperature",
            "prototype_mode",
            "use_projection",
            "use_aux",
            "pool_sigma",
            "lambda_factor",
            "mlp_dim",
            "attention_type",
            "proj_hidden_dim",
            "proj_num_layers",
            "proj_l2_normalize",
        }
    }
    return load_model_with_checkpoint(str(checkpoint), spec.key, model_config, "cpu")


def build_pretrain_model(spec: SceneSpec, objective_id: str, *, use_aux: bool = True):
    """Construct the pretraining model for a G5 masking probe.

    Constructor arguments mirror ``coffe/pretrain/loop.py:340-474``; the
    masking and loss *semantics* under test live entirely inside the repo's
    modules and are executed unmodified. The dispatch itself is pinned
    end-to-end by G1, which goes through ``run_pretrain``.
    """
    from coffe.pretrain.mae_pretrain import MAEPretrainModel
    from coffe.pretrain.simmim import SimMIMPretrainModel

    obj = dict(OBJECTIVES[objective_id])
    # The encoder must carry the PRETRAINING projection setting: the head sits
    # inside the reconstruction forward, so building it eval-shaped would pin a
    # model pretraining never ran.
    encoder = build_eval_model(
        spec,
        model_name="coffe",
        use_aux=use_aux,
        use_projection=obj.pop("use_projection"),
    )

    common = dict(
        encoder=encoder,
        hsi_channels=spec.hsi_channels,
        aux_channels=spec.aux_channels,
        use_aux=use_aux,
        patch_size=PATCH_SIZE,
        embed_dim=COFFE_ARCH["embed_dim"],
        recon_sigma=obj["recon_center_sigma"],
    )
    if obj["objective"] == "mae":
        return MAEPretrainModel(
            **common,
            mask_ratio=obj["mask_ratio"],
            decoder_dim=obj["decoder_dim"],
            decoder_depth=obj["decoder_depth"],
            decoder_heads=obj["decoder_heads"],
            norm_pix_loss=obj["norm_pix_loss"],
        )
    return SimMIMPretrainModel(
        **common,
        decoder_hidden_dim=obj["decoder_hidden_dim"],
        band_mask_ratio=obj["band_mask_ratio"],
        spatial_mask_ratio=obj["spatial_mask_ratio"],
    )


# HyperSIGMA regimes (PAPER_CANON §1): `patch_native` = 11x11 straight into the
# encoders; `backbone_native` = 64x64 with the encoders at their pretrained
# input size. Both with randomly initialised ViT bodies — the wrapper's
# plumbing is what is under test, so no released checkpoint is needed.
HYPERSIGMA_REGIMES: dict[str, dict[str, Any]] = {
    "patch_native_fused": {
        "pca_components": 3,
        "native_geometry": False,
        "spat_patch_k": 3,
        "mode": "fused",
    },
    "backbone_native_upscale_fused": {
        "pca_components": 100,
        "native_geometry": True,
        "input_fit": "upscale",
        "mode": "fused",
    },
}


def _write_dummy_pca(path: Path, in_bands: int, n_components: int) -> str:
    """Fit a small deterministic PCA so HyperSIGMADual can be constructed.

    Mirrors the helper already used by tests/test_hypersigma_shapes.py.
    """
    import pickle

    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    pixels = rng.standard_normal(size=(2048, in_bands)).astype(np.float32)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(pca, fh)
    return str(path)


def build_hypersigma_model(spec: SceneSpec, regime: str, work_dir: Path):
    """Build ``HyperSIGMAFewShot`` over a random-init ``HyperSIGMADual``."""
    from coffe.models.hypersigma import HyperSIGMADual, HyperSIGMAFewShot

    cfg = HYPERSIGMA_REGIMES[regime]
    pca_path = _write_dummy_pca(
        Path(work_dir) / f"pca_{spec.hsi_channels}_{cfg['pca_components']}.pkl",
        spec.hsi_channels,
        cfg["pca_components"],
    )
    kwargs: dict[str, Any] = dict(
        pca_spat_path=pca_path,
        spat_ckpt=None,
        spec_ckpt=None,
        hsi_channels=spec.hsi_channels,
        freeze_body=True,
    )
    if cfg["native_geometry"]:
        kwargs.update(
            native_geometry=True,
            native_pca_spat_path=pca_path,
            native_spat_in_chans=cfg["pca_components"],
            input_fit=cfg["input_fit"],
        )
    else:
        kwargs.update(spat_patch_k=cfg["spat_patch_k"])

    dual = HyperSIGMADual(**kwargs)
    return HyperSIGMAFewShot(dual=dual, mode=cfg["mode"], distance_metric="euclidean")


# ----------------------------------------------------------------------
# Fingerprints
# ----------------------------------------------------------------------


def tensor_fingerprint(tensor) -> dict[str, Any]:
    """``sum`` / ``abs().sum()`` / ``norm()`` of a tensor, in float64.

    Full precision is stored; ``test_equivalence.py`` owns the tolerance.
    """
    import torch

    flat = tensor.detach().to(torch.float64).reshape(-1)
    return {
        "shape": list(tensor.shape),
        "sum": float(flat.sum()),
        "abs_sum": float(flat.abs().sum()),
        "norm": float(flat.norm()),
    }


def state_dict_fingerprint(state_dict) -> dict[str, dict[str, Any]]:
    return {k: tensor_fingerprint(v) for k, v in state_dict.items()}


def fixed_input(spec: SceneSpec, *, batch: int = 4, seed: int = 20260231):
    """A fixed random input batch, in the [0, 1] range min-max normalisation
    produces (``PatchedMultimodalDataset._normalize``)."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    hsi = torch.rand(batch, spec.hsi_channels, PATCH_SIZE, PATCH_SIZE, generator=gen)
    aux = torch.rand(batch, spec.aux_channels, PATCH_SIZE, PATCH_SIZE, generator=gen)
    return hsi, aux


def live_eval_feature(model, hsi, aux):
    """The LIVE eval feature, mirroring coffe/eval/episodic.py:410-425.

    For CoFFE this is ``z = mean_j(patch_emb_j) + lambda * cls_emb`` with
    lambda = 0.5 (PAPER_CANON §8 D3), *not* "patch tokens pooled".
    """
    import torch

    from coffe.utils.spatial_weights import center_weighted_pool

    with torch.no_grad():
        patch, cls, _ = model.forward_features(hsi, aux)
        adapted = model.eval_patch_embeddings(patch, cls)
        if getattr(model, "pool_sigma", None) is not None:
            return center_weighted_pool(adapted, model._center_pool_weights)
        return adapted.mean(dim=1)


def assignment_hash(episodes: Iterable[dict[str, Any]]) -> str:
    """SHA-256 over every episode's (classes, query labels, argmin assignment).

    Compared exactly: a single flipped query prediction changes the digest.
    """
    digest = hashlib.sha256()
    for ep in episodes:
        for field in ("original_classes", "q_labels", "q_preds"):
            digest.update(np.asarray(ep[field], dtype=np.int64).tobytes())
    return digest.hexdigest()


def episode_assignments(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the per-episode assignment records out of an evaluator result."""
    return [
        {
            "original_classes": list(ep["original_classes"]),
            "q_labels": np.asarray(ep["q_labels"]).tolist(),
            "q_preds": np.asarray(ep["q_preds"]).tolist(),
        }
        for ep in results["example_episodes_data"]
    ]


def oa_fingerprint(results: dict[str, Any], places: int = 6) -> dict[str, float]:
    return {
        "mean": round(float(results["OA"]["mean"]), places),
        "std": round(float(results["OA"]["std"]), places),
        "ci_95": round(float(results["OA"]["ci_95"]), places),
        "num_episodes": int(results["num_episodes"]),
    }


# ----------------------------------------------------------------------
# Environment metadata
# ----------------------------------------------------------------------


def environment_metadata() -> dict[str, Any]:
    import subprocess

    import torch

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover - git always present in this repo
            return "<unavailable>"

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(_git("status", "--porcelain")),
        "platform": sys.platform,
    }


def set_determinism(seed: int = 0) -> None:
    """Apply the harness determinism contract: CPU only, all RNGs seeded,
    deterministic kernels, no AMP.

    Called by ``conftest.py`` (autouse) and by ``make_golden.py``.
    """
    import random

    import torch

    # Only needed if a future GPU-marked test ever combines CUDA with
    # `use_deterministic_algorithms`; cuBLAS reads it lazily, so setting it here
    # is effective. Everything in this package runs on CPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    # Deliberately NOT set here: PYTHONHASHSEED (inert once the interpreter is
    # running) and TQDM_DISABLE (tqdm 4.67 does not read it). No fingerprint
    # depends on string-hash ordering: golden JSON is written with
    # `sort_keys=True` and every key comparison is over a sorted set.

"""Preprocessing: patch geometry, per-band min-max, HyperSIGMA band fitting.

PAPER_CANON §5 states the preprocessing: 11x11 patches centred on each labelled
pixel, border-padded at edges; every HSI band and the LiDAR raster min-max
normalised to [0, 1] independently. For the HyperSIGMA spatial branch: PCA
144->100 on Houston, linear band resampling 63/64->100 on Trento/MUUFL, with
the spectral branch receiving raw bands.

**One thing does not hold as written, and is pinned here as it actually is.**
The paper path never patches an image in this repo: it consumes the pre-patched
MFT-format ``.mat`` files, where the 11x11 extraction (and its border padding)
happened upstream in the MFT data preparation. The repo's own raw-image path,
``coffe.data.datasets.base.MultimodalEODataset``, **drops** border pixels
instead of padding them — and is on no paper path (nothing under ``scripts/``
or ``coffe/runners/`` constructs it). Both behaviours are asserted below;
neither is changed. Reported at the phase-7 gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from coffe.data.datasets import DATASET_REGISTRY, MultimodalEODataset, get_spec
from tests.conftest import CANON_DATASET_SPECS, SCENE_KEYS, spec_for
from tests.equivalence._harness import load_scene_dataset

PATCH = 11


@pytest.fixture(scope="module")
def datasets(mini_scene_root):
    return {key: load_scene_dataset(spec_for(key), mini_scene_root, "all") for key in SCENE_KEYS}


# ----------------------------------------------------------------------
# The live (pre-patched) path
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_patches_are_11x11_with_the_scenes_band_counts(datasets, scene: str) -> None:
    """PAPER_CANON §5: 11x11 patches, C_h HSI bands + C_a aux channels."""
    canon = CANON_DATASET_SPECS[scene]
    dataset = datasets[scene]

    sample = dataset[0]

    assert sample["hsi"].shape == (canon["hsi_channels"], PATCH, PATCH)
    assert sample["aux"].shape == (canon["aux_channels"], PATCH, PATCH)
    assert dataset.hsi_channels == canon["hsi_channels"]
    assert dataset.aux_channels == canon["aux_channels"]
    assert dataset.num_classes == canon["num_classes"]


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_channels_last_files_are_transposed_to_nchw(datasets, scene: str) -> None:
    """The ``.mat`` files are ``[N, 11, 11, C]``; the dataset yields ``[N, C, 11, 11]``.

    The mini-scenes are written channels-last exactly like the real files, so
    ``_ensure_nchw`` is what makes this pass.
    """
    canon = CANON_DATASET_SPECS[scene]
    dataset = datasets[scene]

    assert dataset.hsi.shape[1:] == (canon["hsi_channels"], PATCH, PATCH)
    assert dataset.aux.shape[1:] == (canon["aux_channels"], PATCH, PATCH)


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_each_band_is_min_max_normalised_to_the_unit_interval(datasets, scene: str) -> None:
    """PAPER_CANON §5: every band normalised to [0, 1] **independently**.

    Per band, not per tensor: each channel's own min must land on 0 and its own
    max on 1. A single global min-max would leave most bands strictly inside
    the interval, which the exactness of these assertions rules out.
    """
    dataset = datasets[scene]

    for tensor in (dataset.hsi, dataset.aux):
        per_band_min = tensor.amin(dim=(0, 2, 3))
        per_band_max = tensor.amax(dim=(0, 2, 3))
        assert torch.allclose(per_band_min, torch.zeros_like(per_band_min), atol=1e-6)
        assert torch.allclose(per_band_max, torch.ones_like(per_band_max), atol=1e-6)
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_normalisation_is_exactly_the_per_band_min_max_map(datasets) -> None:
    """The transform is ``(x - band_min) / (band_max - band_min)``, band by band.

    Compared against the same dataset loaded with ``normalize=False``, so the
    claim is about the actual pair of tensors rather than about the interval
    the result happens to land in.
    """
    normalised = datasets["trento"]
    unnormalised = type(normalised)(
        data_root=normalised.data_root, patch_size=PATCH, split="all", normalize=False
    )

    for attr in ("hsi", "aux"):
        raw = getattr(unnormalised, attr)
        got = getattr(normalised, attr)
        lo = raw.amin(dim=(0, 2, 3)).view(1, -1, 1, 1)
        hi = raw.amax(dim=(0, 2, 3)).view(1, -1, 1, 1)
        assert torch.allclose(got, (raw - lo) / (hi - lo), atol=1e-6), attr


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_labels_are_one_indexed_and_background_free(datasets, scene: str) -> None:
    """The frozen ``.mat`` files carry labels 1..N; class 0 (background) is absent."""
    canon = CANON_DATASET_SPECS[scene]
    dataset = datasets[scene]

    labels = dataset.labels.numpy()
    assert labels.min() == 1
    assert labels.max() == canon["num_classes"]
    assert sorted(dataset.class_indices) == list(range(1, canon["num_classes"] + 1))


@pytest.mark.parametrize("scene", SCENE_KEYS)
def test_class_indices_partition_the_dataset(datasets, scene: str) -> None:
    dataset = datasets[scene]

    flat = [i for indices in dataset.class_indices.values() for i in indices]

    assert sorted(flat) == list(range(len(dataset)))


def test_train_and_test_splits_concatenate_into_all(mini_scene_root) -> None:
    """``split="all"`` — what every paper eval used — is Tr + Te."""
    scene = spec_for("houston")
    train = load_scene_dataset(scene, mini_scene_root, "train")
    test = load_scene_dataset(scene, mini_scene_root, "test")
    everything = load_scene_dataset(scene, mini_scene_root, "all")

    assert len(everything) == len(train) + len(test)


# ----------------------------------------------------------------------
# The raw-image path: border pixels are dropped, not padded
# ----------------------------------------------------------------------


class _RawScene(MultimodalEODataset):
    """A tiny raw-image scene, to exercise the non-paper patching path."""

    def _load_data(self):
        rng = np.random.default_rng(0)
        hsi = rng.random((20, 20, 4)).astype(np.float32)
        aux = rng.random((20, 20, 1)).astype(np.float32)
        labels = np.zeros((20, 20), dtype=np.int64)
        labels[:, :] = 1  # every pixel labelled, border included
        return hsi, aux, labels

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def hsi_channels(self) -> int:
        return 4

    @property
    def aux_channels(self) -> int:
        return 1


def test_raw_image_path_drops_border_pixels_instead_of_padding() -> None:
    """PAPER_CANON §5 says "border-padded at edges"; this path excludes them.

    A 20x20 fully-labelled scene with ``patch_size=11`` (pad = 5) keeps only
    the 10x10 interior. That is the frozen behaviour and is **not** changed
    here: this class is on no paper path (the paper consumes pre-patched
    ``.mat`` files, where padding happened upstream in the MFT data
    preparation). Recorded as a phase-7 gate item.
    """
    dataset = _RawScene(data_root=".", patch_size=PATCH)

    coords = dataset.class_indices[1]

    assert len(coords) == 10 * 10  # 20 - 2*5 in each axis, not 20*20
    ys = [y for y, _ in coords]
    xs = [x for _, x in coords]
    assert min(ys) == min(xs) == 5
    assert max(ys) == max(xs) == 14


def test_raw_image_patches_are_centred_on_their_pixel() -> None:
    """For a kept coordinate, the patch is the 11x11 window centred on it."""
    dataset = _RawScene(data_root=".", patch_size=PATCH)

    y, x = 9, 12
    hsi, aux = dataset.extract_patch(y, x)

    assert hsi.shape == (4, PATCH, PATCH)
    assert aux.shape == (1, PATCH, PATCH)
    centre = PATCH // 2
    assert torch.allclose(hsi[:, centre, centre], torch.from_numpy(dataset.hsi[y, x, :]))


# ----------------------------------------------------------------------
# HyperSIGMA band fitting (PAPER_CANON §5)
# ----------------------------------------------------------------------


def test_houston_spatial_branch_is_fitted_by_pca_144_to_100(datasets, tmp_path) -> None:
    """PAPER_CANON §5: PCA 144->100 on Houston, fitted on the scene's pixels.

    Fitted through the repo's own ``fit_dataset_pca`` over the mini Houston
    scene, so the fit -> pickle -> load -> apply path runs end to end. The
    components are meaningless on synthetic data; the geometry is the point.
    """
    from coffe.models.hypersigma.preprocessing import (
        PCAPreprocessor,
        cumulative_explained_variance,
        fit_dataset_pca,
        load_pca,
    )

    out = tmp_path / "pca_houston_100band.pkl"
    fitted = fit_dataset_pca(datasets["houston"], n_components=100, save_path=str(out))

    assert fitted.n_features_in_ == 144
    assert fitted.components_.shape == (100, 144)
    assert 0.0 < cumulative_explained_variance(fitted) <= 1.0

    reloaded = load_pca(str(out))
    assert reloaded.components_.shape == (100, 144)

    front_end = PCAPreprocessor(reloaded)
    x = torch.rand(2, 144, PATCH, PATCH)
    with torch.no_grad():
        projected = front_end(x)
    assert projected.shape == (2, 100, PATCH, PATCH)


@pytest.mark.parametrize("bands", [63, 64], ids=["trento", "muufl"])
def test_trento_and_muufl_are_linearly_resampled_up_to_100(bands: int) -> None:
    """PAPER_CANON §5: linear band resampling 63/64 -> 100, no PCA."""
    from coffe.models.hypersigma.preprocessing import SpectralResample

    resample = SpectralResample(out_channels=100, mode="linear")
    x = torch.rand(2, bands, PATCH, PATCH)

    with torch.no_grad():
        out = resample(x)

    assert out.shape == (2, 100, PATCH, PATCH)
    # Linear interpolation is bounded by its inputs: no band may be invented
    # outside the observed range.
    assert out.min() >= x.min() - 1e-6
    assert out.max() <= x.max() + 1e-6


def test_hundred_band_input_passes_through_resampling_unchanged() -> None:
    """A scene already at 100 bands must not be perturbed."""
    from coffe.models.hypersigma.preprocessing import SpectralResample

    resample = SpectralResample(out_channels=100, mode="linear")
    x = torch.rand(2, 100, PATCH, PATCH)

    with torch.no_grad():
        out = resample(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_dataset_registry_matches_the_paper_table() -> None:
    """PAPER_CANON §5: the registry is the single source of per-scene specs."""
    assert sorted(DATASET_REGISTRY) == sorted(CANON_DATASET_SPECS)
    for scene, canon in CANON_DATASET_SPECS.items():
        spec = get_spec(scene)
        assert spec.hsi_channels == canon["hsi_channels"]
        assert spec.aux_channels == canon["aux_channels"]
        assert spec.num_classes == canon["num_classes"]


def test_unknown_dataset_is_rejected_with_the_known_set() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_spec("pavia")

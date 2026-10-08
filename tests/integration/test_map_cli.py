"""The map pipeline end to end: recover coordinates, then paint a scene.

``scripts/recover_patch_coords.py`` and ``scripts/predict_map.py`` are the two
halves of turning frozen-protocol decisions into a picture next to the ground
truth, and they only mean anything together — the first produces the table the
second consumes, in the dataset index space the feature cache uses. This runs
both, on a synthetic scene whose ground-truth masks are built here so the
recovery has a known answer.

``--help`` for both is covered by ``tests/integration/test_cli.py``'s glob.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from coffe.data.scene_coords import SceneCoords
from tests.conftest import spec_for
from tests.equivalence._harness import write_scene

REPO = Path(__file__).resolve().parents[2]
SCENE = "trento"
WIDTH = 24


def _run(*args: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=str(REPO), capture_output=True, text=True, timeout=timeout
    )


def _mask_for(labels: np.ndarray, shape: tuple[int, int], first_row: int) -> np.ndarray:
    """Lay ``labels`` out row by row from ``first_row`` — a row-major scan."""
    mask = np.zeros(shape, dtype=np.int64)
    for i, label in enumerate(labels):
        mask[first_row + i // WIDTH, i % WIDTH] = int(label)
    return mask


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """A synthetic pack plus the two masks it would have been cut from."""
    import scipy.io as sio

    spec = spec_for(SCENE)
    root = tmp_path_factory.mktemp("map_cli") / "raw"
    write_scene(spec, root)

    folder = root / spec.folder
    labels_tr = sio.loadmat(folder / "TrLabel.mat")["Data"].squeeze()
    labels_te = sio.loadmat(folder / "TeLabel.mat")["Data"].squeeze()

    rows_tr = int(np.ceil(len(labels_tr) / WIDTH))
    rows_te = int(np.ceil(len(labels_te) / WIDTH))
    shape = (rows_tr + rows_te + 2, WIDTH)  # +2: unlabelled rows between/after
    masks = folder.parent / "masks"
    masks.mkdir(exist_ok=True)
    sio.savemat(masks / "gt_train.mat", {"TRLabel": _mask_for(labels_tr, shape, 0)})
    sio.savemat(masks / "gt_test.mat", {"TSLabel": _mask_for(labels_te, shape, rows_tr + 1)})

    return {
        "root": root,
        "masks": masks,
        "shape": shape,
        "num_samples": len(labels_tr) + len(labels_te),
        "num_test": len(labels_te),
        "num_classes": spec.num_classes,
    }


def _recover(scene, out: Path, *extra: str, report: Path | None = None):
    return _run(
        "scripts/recover_patch_coords.py",
        "--dataset",
        SCENE,
        "--data-root",
        str(scene["root"]),
        "--gt-train",
        str(scene["masks"] / "gt_train.mat"),
        "--gt-test",
        str(scene["masks"] / "gt_test.mat"),
        "--out",
        str(out),
        *(("--report", str(report)) if report else ()),
        *extra,
    )


@pytest.fixture(scope="module")
def coords_npz(scene, tmp_path_factory):
    out = tmp_path_factory.mktemp("coords") / "coords_trento.npz"
    report = out.parent / "report.json"
    # The mini pack is class-major, so its label sequence is class-sorted and
    # cannot separate a raster order from a class-grouped one (see
    # test_an_ambiguous_layout_is_refused_until_an_order_is_named). Real scene
    # masks are not class-sorted in raster order; here the order is named.
    result = _recover(
        scene, out, "--order-train", "row_major", "--order-test", "row_major", report=report
    )
    assert result.returncode == 0, result.stderr
    return out, json.loads(report.read_text())


def test_an_ambiguous_layout_is_refused_until_an_order_is_named(scene, tmp_path):
    """No guessing: when the labels cannot separate two tables, say so."""
    out = tmp_path / "coords.npz"

    refused = _recover(scene, out)

    assert refused.returncode == 1
    assert "disagree on which pixel" in refused.stderr
    assert not out.exists()
    assert (
        _recover(scene, out, "--order-train", "row_major", "--order-test", "row_major").returncode
        == 0
    )


def test_recovery_finds_the_order_the_masks_were_written_in(coords_npz, scene):
    _, report = coords_npz

    assert report["orders"] == {"train": "row_major", "test": "row_major"}
    assert report["num_samples"] == scene["num_samples"]
    assert report["scene_shape"] == list(scene["shape"])


def test_a_mask_from_the_wrong_scene_fails_loudly(scene, tmp_path):
    """The failure mode that matters: a plausible-looking mask that is not the
    one this pack was cut from must not silently produce a map."""
    import scipy.io as sio

    rng = np.random.default_rng(0)
    bogus = rng.integers(0, scene["num_classes"] + 1, size=scene["shape"]).astype(np.int64)
    path = tmp_path / "bogus.mat"
    sio.savemat(path, {"gt": bogus})

    result = _run(
        "scripts/recover_patch_coords.py",
        "--dataset",
        SCENE,
        "--data-root",
        str(scene["root"]),
        "--gt-train",
        str(path),
        "--out",
        str(tmp_path / "coords.npz"),
    )

    assert result.returncode == 1
    assert "no candidate scan order" in result.stderr
    assert not (tmp_path / "coords.npz").exists()


def test_a_single_pack_can_be_recovered_on_its_own(scene, tmp_path):
    """``--pack Te``: one mask, the test pack, whose index space is then the
    dataset's ``split="test"``."""
    out = tmp_path / "coords_te.npz"

    result = _run(
        "scripts/recover_patch_coords.py",
        "--dataset",
        SCENE,
        "--data-root",
        str(scene["root"]),
        "--gt-train",
        str(scene["masks"] / "gt_test.mat"),
        "--pack",
        "Te",
        "--order-train",
        "row_major",
        "--out",
        str(out),
    )
    assert result.returncode == 0, result.stderr

    coords = SceneCoords.load(out)
    assert coords.orders == {"test": "row_major"}
    assert set(coords.pack.tolist()) == {1}
    assert len(coords) == scene["num_test"]


@pytest.fixture(scope="module")
def painted(scene, coords_npz, tmp_path_factory):
    coords, _ = coords_npz
    out_dir = tmp_path_factory.mktemp("maps")
    result = _run(
        "scripts/predict_map.py",
        "--checkpoint",
        "random",
        "--dataset",
        SCENE,
        "--data-root",
        str(scene["root"]),
        "--coords",
        str(coords),
        "--out-dir",
        str(out_dir),
        "--device",
        "cpu",
        "--embed-dim",
        "32",
        "--num-layers",
        "1",
        "--batch-size",
        "64",
    )
    assert result.returncode == 0, result.stderr
    return out_dir


def test_the_pass_writes_decisions_a_summary_and_a_picture(painted, scene):
    stem = f"{SCENE}_seed42"
    summary = json.loads((painted / f"summary_{SCENE}.json").read_text())

    assert (painted / f"map_{stem}.png").stat().st_size > 0
    assert summary["num_support"] == scene["num_classes"] * 5  # PAPER_CANON §4
    assert summary["num_support"] + summary["num_queries"] == scene["num_samples"]
    assert 0.0 <= summary["metrics"]["OA"] <= 100.0

    with np.load(painted / f"decisions_{stem}.npz") as blob:
        assert len(blob["predictions"]) == summary["num_queries"]


def test_the_painted_map_covers_every_labelled_pixel_and_nothing_else(painted, scene):
    with np.load(painted / f"maps_{SCENE}_seed42.npz") as blob:
        truth, prediction, agreement = blob["truth"], blob["prediction"], blob["agreement"]

    assert truth.shape == scene["shape"]
    assert np.count_nonzero(truth) == scene["num_samples"]
    # A decision (or a known support class) exactly where there is ground truth.
    assert np.array_equal(prediction > 0, truth > 0)
    assert np.array_equal(agreement > 0, truth > 0)


def test_a_coordinate_table_for_another_split_is_refused(scene, coords_npz, tmp_path):
    """The table and the features must share an index space; ``--split test``
    against an ``all`` table is the easy way to get that wrong."""
    coords, _ = coords_npz
    result = _run(
        "scripts/predict_map.py",
        "--checkpoint",
        "random",
        "--dataset",
        SCENE,
        "--data-root",
        str(scene["root"]),
        "--split",
        "test",
        "--coords",
        str(coords),
        "--out-dir",
        str(tmp_path / "out"),
        "--device",
        "cpu",
    )

    assert result.returncode != 0
    assert "share an index space" in result.stderr

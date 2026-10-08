"""Recovering a pre-patched sample's pixel from the scene's ground-truth mask.

``coffe.data.scene_coords`` claims that a pack's label vector identifies the
scan order the patches were cut in, and that the matching order *is* the
missing index -> (row, col) table. The tests below hold it to that claim on
synthetic scenes: each candidate order is recovered from its own label
sequence, a mask that cannot distinguish two orders is reported as ambiguous
rather than guessed at, a wrong mask is rejected, and — the independent check —
patches cut from a cube in a known order have their centres exactly where the
recovered table says.

Everything here is synthetic; no dataset under ``data/raw/`` is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from coffe.data.scene_coords import (
    SCAN_ORDERS,
    OrderRecoveryError,
    SceneCoords,
    label_sequence,
    match_orders,
    recover,
    scan_coordinates,
    verify_with_cube,
)

SHAPE = (12, 17)
N_CLASSES = 4


def interleaved_mask(seed: int = 7, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """A mask whose classes are scattered, so no two candidate orders agree."""
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, N_CLASSES + 1, size=shape)  # 0 = unlabelled
    # Guarantee every class is present and the scene has unlabelled pixels.
    for c in range(1, N_CLASSES + 1):
        mask[0, c] = c
    mask[-1, :] = 0
    return mask.astype(np.int64)


def column_block_mask(shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """Classes in contiguous column blocks: column-major *is* class order."""
    mask = np.zeros(shape, dtype=np.int64)
    width = shape[1] // N_CLASSES
    for c in range(N_CLASSES):
        mask[:, c * width : (c + 1) * width] = c + 1
    return mask


def row_block_mask(shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """Classes in contiguous row blocks: row-major *is* class order."""
    mask = np.zeros(shape, dtype=np.int64)
    height = shape[0] // N_CLASSES
    for c in range(N_CLASSES):
        mask[c * height : (c + 1) * height, :] = c + 1
    return mask


# ----------------------------------------------------------------------
# Order recovery
# ----------------------------------------------------------------------


@pytest.mark.parametrize("order", ["column_major", "row_major"])
def test_a_raster_order_is_recovered_from_its_own_label_sequence(order: str) -> None:
    mask = interleaved_mask()
    labels = label_sequence(mask, order)

    assert match_orders(mask, labels) == [order]


def test_the_class_grouped_orders_share_one_label_sequence() -> None:
    """The fingerprint's limit, stated as a fact rather than discovered later.

    Grouping by class makes the sequence ``[1...1, 2...2, ...]`` whatever raster
    order is grouped, so the labels cannot say which. The two disagree on the
    pixels, so this is the case where the cube check is not optional.
    """
    mask = interleaved_mask()
    labels = label_sequence(mask, "class_then_column_major")

    assert match_orders(mask, labels) == [
        "class_then_column_major",
        "class_then_row_major",
    ]
    with pytest.raises(OrderRecoveryError, match="disagree on which pixel"):
        recover("synthetic", mask, labels)


@pytest.mark.parametrize("order", sorted(SCAN_ORDERS))
def test_recovered_coordinates_reproduce_the_mask(order: str) -> None:
    """The table addresses exactly the pixels the mask labels, with its values."""
    mask = interleaved_mask()
    labels = label_sequence(mask, order)

    coords = recover("synthetic", mask, labels, order_train=order)

    assert coords.orders == {"train": order}
    assert len(coords) == int(np.count_nonzero(mask))
    assert np.array_equal(coords.gt_map(), mask)


def test_column_major_is_matlabs_find_order() -> None:
    """Down each column, columns left to right — what ``find`` in MATLAB gives."""
    mask = np.array([[0, 2], [1, 3]], dtype=np.int64)

    assert np.array_equal(scan_coordinates(mask, "column_major"), [[1, 0], [0, 1], [1, 1]])
    assert np.array_equal(scan_coordinates(mask, "row_major"), [[0, 1], [1, 0], [1, 1]])


def test_orders_that_disagree_on_pixels_are_reported_not_guessed() -> None:
    """A class-contiguous mask matches three orders; two of them differ."""
    mask = column_block_mask()
    labels = label_sequence(mask, "column_major")

    matched = match_orders(mask, labels)
    assert set(matched) == {
        "column_major",
        "class_then_column_major",
        "class_then_row_major",
    }

    with pytest.raises(OrderRecoveryError, match="disagree on which pixel"):
        recover("synthetic", mask, labels)

    # ...but the caller may settle it (with verify_with_cube) and force one.
    coords = recover("synthetic", mask, labels, order_train="column_major")
    assert coords.orders == {"train": "column_major"}


def test_orders_that_agree_on_pixels_are_not_an_ambiguity() -> None:
    """Column-major over a column-blocked mask already *is* its class-grouped
    self, so naming either is naming one table — recovery just proceeds."""
    mask = column_block_mask()
    labels = label_sequence(mask, "column_major")
    orders = {name: SCAN_ORDERS[name] for name in ("column_major", "class_then_column_major")}

    coords = recover("synthetic", mask, labels, orders=orders)

    assert coords.orders["train"] in orders
    assert np.array_equal(coords.gt_map(), mask)


def test_a_mask_from_another_scene_is_rejected() -> None:
    mask = interleaved_mask()
    labels = label_sequence(mask, "column_major")
    rng = np.random.default_rng(0)
    shuffled = labels.copy()
    rng.shuffle(shuffled)

    assert match_orders(mask, shuffled) == []
    with pytest.raises(OrderRecoveryError, match="no candidate scan order"):
        recover("synthetic", mask, shuffled)


def test_a_count_mismatch_says_so() -> None:
    mask = interleaved_mask()
    labels = label_sequence(mask, "column_major")[:-3]

    with pytest.raises(OrderRecoveryError, match="wrong mask or the wrong split"):
        recover("synthetic", mask, labels)


def test_a_forced_order_that_does_not_fit_is_rejected() -> None:
    mask = interleaved_mask()
    labels = label_sequence(mask, "column_major")

    with pytest.raises(OrderRecoveryError, match="does not reproduce"):
        recover("synthetic", mask, labels, order_train="row_major")


# ----------------------------------------------------------------------
# The two packs
# ----------------------------------------------------------------------


def _split_masks(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """One mask split into disjoint train/test halves, as the scenes are."""
    mask = interleaved_mask(seed)
    rng = np.random.default_rng(seed + 1)
    train = mask.copy()
    test = mask.copy()
    keep_train = rng.random(mask.shape) < 0.3
    train[~keep_train] = 0
    test[keep_train] = 0
    return train, test


def test_the_table_concatenates_train_then_test() -> None:
    """The ``split="all"`` index space: train pack first, exactly as the
    dataset's own ``_load_data`` concatenates the two ``.mat`` files."""
    train, test = _split_masks()
    labels_tr = label_sequence(train, "column_major")
    labels_te = label_sequence(test, "column_major")

    coords = recover("synthetic", train, labels_tr, mask_test=test, labels_test=labels_te)

    n_tr = len(labels_tr)
    assert len(coords) == n_tr + len(labels_te)
    assert coords.pack[:n_tr].tolist() == [0] * n_tr
    assert coords.pack[n_tr:].tolist() == [1] * len(labels_te)
    assert coords.orders == {"train": "column_major", "test": "column_major"}
    # Both packs land on their own pixels and nowhere else.
    assert np.array_equal(coords.gt_map(), np.where(train > 0, train, test))


def test_masks_of_different_shapes_are_rejected() -> None:
    train, test = _split_masks()
    with pytest.raises(ValueError, match="disagree on the scene shape"):
        recover(
            "synthetic",
            train,
            label_sequence(train, "column_major"),
            mask_test=test[:-1],
            labels_test=label_sequence(test[:-1], "column_major"),
        )


def test_two_samples_on_one_pixel_are_rejected() -> None:
    """The guard that catches a wrong order before anything is painted."""
    with pytest.raises(ValueError, match="same pixel"):
        SceneCoords(
            scene="synthetic",
            shape=(4, 4),
            rows=np.array([1, 1]),
            cols=np.array([2, 2]),
            labels=np.array([1, 2]),
            pack=np.array([0, 0]),
            orders={"train": "column_major"},
        )


# ----------------------------------------------------------------------
# Painting and persistence
# ----------------------------------------------------------------------


def test_to_map_scatters_a_subset_and_leaves_the_rest_at_fill() -> None:
    mask = interleaved_mask()
    coords = recover("synthetic", mask, label_sequence(mask, "column_major"))

    picked = np.array([0, 3, 7])
    canvas = coords.to_map(np.array([9, 9, 9]), indices=picked, fill=-1)

    assert set(np.unique(canvas).tolist()) == {-1, 9}
    assert (canvas == 9).sum() == len(picked)
    assert canvas[coords.rows[3], coords.cols[3]] == 9


def test_save_load_round_trip(tmp_path) -> None:
    mask = interleaved_mask()
    coords = recover("synthetic", mask, label_sequence(mask, "row_major"))

    restored = SceneCoords.load(coords.save(tmp_path / "coords.npz"))

    assert restored.scene == coords.scene
    assert restored.shape == coords.shape
    assert restored.orders == coords.orders
    assert np.array_equal(restored.rows, coords.rows)
    assert np.array_equal(restored.cols, coords.cols)
    assert np.array_equal(restored.labels, coords.labels)
    assert np.array_equal(restored.pack, coords.pack)


# ----------------------------------------------------------------------
# The independent confirmation: patch centres against the cube
# ----------------------------------------------------------------------


def _cut_patches(cube: np.ndarray, coords: np.ndarray, patch: int = 11) -> np.ndarray:
    """Border-padded patches centred on ``coords``, as the MFT prep produced."""
    pad = patch // 2
    padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return np.stack([padded[r : r + patch, c : c + patch, :] for r, c in coords], axis=0)


def test_patch_centres_confirm_the_recovered_order() -> None:
    rng = np.random.default_rng(3)
    mask = interleaved_mask()
    cube = rng.normal(size=(*SHAPE, 5))
    truth = scan_coordinates(mask, "column_major")
    patches = _cut_patches(cube, truth)

    coords = recover("synthetic", mask, label_sequence(mask, "column_major"))
    report = verify_with_cube(coords, cube, patches, max_checks=len(patches))

    assert report["checked"] == len(patches)
    assert report["mismatched"] == 0
    assert report["max_abs_diff"] == 0.0


def test_patch_centres_reject_the_wrong_order() -> None:
    """The tie-breaker actually breaks ties.

    A row-blocked mask makes ``row_major`` and ``class_then_column_major`` share
    one label sequence while walking different pixels. Only the order the
    patches were really cut in survives the cube.
    """
    rng = np.random.default_rng(4)
    mask = row_block_mask()
    cube = rng.normal(size=(*SHAPE, 5))
    labels = label_sequence(mask, "row_major")
    assert set(match_orders(mask, labels)) >= {"row_major", "class_then_column_major"}

    patches = _cut_patches(cube, scan_coordinates(mask, "class_then_column_major"))
    right = recover("synthetic", mask, labels, order_train="class_then_column_major")
    wrong = recover("synthetic", mask, labels, order_train="row_major")

    assert verify_with_cube(right, cube, patches, max_checks=len(patches))["mismatched"] == 0
    assert verify_with_cube(wrong, cube, patches, max_checks=len(patches))["mismatched"] > 0


def test_the_check_holds_at_the_scene_border() -> None:
    """Padding changes a border patch's edges, never its centre."""
    rng = np.random.default_rng(5)
    mask = np.zeros(SHAPE, dtype=np.int64)
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = 1
    cube = rng.normal(size=(*SHAPE, 5))
    coords = recover(
        "synthetic", mask, label_sequence(mask, "column_major"), order_train="column_major"
    )
    patches = _cut_patches(cube, scan_coordinates(mask, "column_major"))

    assert verify_with_cube(coords, cube, patches, max_checks=4)["mismatched"] == 0


def test_the_test_pack_is_checked_at_its_own_offset() -> None:
    """``patch_offset`` lets the second pack be checked without a padded copy."""
    rng = np.random.default_rng(6)
    train, test = _split_masks()
    cube = rng.normal(size=(*SHAPE, 5))
    coords = recover(
        "synthetic",
        train,
        label_sequence(train, "column_major"),
        mask_test=test,
        labels_test=label_sequence(test, "column_major"),
    )
    n_tr = int((coords.pack == 0).sum())
    test_patches = _cut_patches(cube, scan_coordinates(test, "column_major"))

    report = verify_with_cube(
        coords, cube, test_patches, patch_offset=n_tr, max_checks=len(test_patches)
    )

    assert report["checked"] == len(test_patches)
    assert report["mismatched"] == 0

    with pytest.raises(IndexError):
        verify_with_cube(coords, cube, test_patches, indices=np.array([0]), patch_offset=n_tr)

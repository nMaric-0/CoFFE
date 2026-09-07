"""Recover each pre-patched sample's pixel coordinate in the original scene.

The packs this repository consumes (``data/raw/<Scene>11x11/``) are
coordinate-free: ``HSI_Tr.mat`` is ``[N, 11, 11, C]``, ``TrLabel.mat`` is a
flat ``[1, N]`` label vector, and nothing records *where* in the scene sample
``i`` was cut from. The 11x11 extraction happened upstream in the MFT data
preparation (README "Data"), which kept only the patch stack. So a prediction
made by the frozen protocol can be attributed to a sample index but not, out of
the box, to a pixel — which is what a classification map next to the ground
truth needs.

**The recovery.** MFT cut one patch per labelled pixel by scanning the scene's
ground-truth mask, so the pack's label vector is a fingerprint of that scan: if
enumerating the mask's non-zero pixels in some order reproduces the label
vector *element for element*, that order is the one the extraction used, and
the enumeration is the missing index -> (row, col) table. Over Houston's 2832 +
12197 entries an accidental match of a 15-valued sequence is not a thing that
happens, so a single matching order is proof rather than evidence.

:func:`match_orders` runs that test over the candidate orders in
:data:`SCAN_ORDERS` (MATLAB's column-major ``find`` order, C row-major, and
each grouped by class, which is how a per-class extraction loop comes out).
:func:`recover` turns the winner into a :class:`SceneCoords` covering the
``split="all"`` index space — train pack first, then test, exactly the
concatenation :meth:`~coffe.data.datasets.patched.PatchedMultimodalDataset._load_data`
performs.

**Where the fingerprint runs out.** A label sequence pins the *pixels*, not
always the order they were walked in: the two class-grouped candidates produce
the identical sequence by construction, and a class-contiguous mask makes a
raster order coincide with them too. When the matching orders enumerate the
same pixels in the same sequence that is no ambiguity at all and recovery
proceeds; when they genuinely disagree, the caller is told rather than handed a
guess. :func:`verify_with_cube` then settles it against the scene's own HSI
cube, where patch ``i``'s centre pixel must equal ``cube[row, col]``. The centre
pixel is never touched by the border padding, so that check is valid at the
scene edge too — and it is worth running once even when a single order matched.

Nothing here is on a paper path: it adds an attribution table beside the frozen
artifacts and changes no computed number.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SCAN_ORDERS",
    "OrderRecoveryError",
    "SceneCoords",
    "load_mat_array",
    "match_orders",
    "pack_dir",
    "recover",
    "scan_coordinates",
    "verify_with_cube",
]


class OrderRecoveryError(RuntimeError):
    """No candidate scan order reproduces the pack's label vector."""


# ----------------------------------------------------------------------
# Candidate scan orders
# ----------------------------------------------------------------------


def _column_major(mask: np.ndarray) -> np.ndarray:
    """MATLAB ``find`` order: down each column, columns left to right.

    ``np.nonzero`` enumerates in C order, so transposing the mask makes its
    columns the rows being walked.
    """
    cols, rows = np.nonzero(mask.T)
    return np.stack([rows, cols], axis=1)


def _row_major(mask: np.ndarray) -> np.ndarray:
    """C / numpy order: along each row, rows top to bottom."""
    rows, cols = np.nonzero(mask)
    return np.stack([rows, cols], axis=1)


def _grouped_by_class(
    base: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """``base``, then a stable sort by class — a per-class extraction loop."""

    def order(mask: np.ndarray) -> np.ndarray:
        coords = base(mask)
        labels = mask[coords[:, 0], coords[:, 1]]
        return coords[np.argsort(labels, kind="stable")]

    return order


#: The candidate scan orders, by name. Extend by passing ``orders=`` to
#: :func:`match_orders`; a scan order is any ``mask -> [n, 2]`` enumeration of
#: the mask's non-zero pixels.
SCAN_ORDERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "column_major": _column_major,
    "row_major": _row_major,
    "class_then_column_major": _grouped_by_class(_column_major),
    "class_then_row_major": _grouped_by_class(_row_major),
}


def scan_coordinates(mask: np.ndarray, order: str) -> np.ndarray:
    """The mask's labelled pixels as ``[n, 2]`` ``(row, col)`` in ``order``."""
    try:
        enumerate_in = SCAN_ORDERS[order]
    except KeyError:
        raise KeyError(f"unknown scan order '{order}'; known: {sorted(SCAN_ORDERS)}") from None
    return enumerate_in(np.asarray(mask))


def label_sequence(mask: np.ndarray, order: str) -> np.ndarray:
    """The class values the mask yields under ``order``, as ``int64``."""
    mask = np.asarray(mask)
    coords = scan_coordinates(mask, order)
    return mask[coords[:, 0], coords[:, 1]].astype(np.int64)


def match_orders(
    mask: np.ndarray,
    labels: np.ndarray,
    *,
    orders: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
) -> list[str]:
    """Every candidate order whose label sequence equals ``labels`` exactly.

    Args:
        mask: the scene ground truth, ``[H, W]``, 0 = unlabelled.
        labels: the pack's label vector, any shape squeezable to ``[n]``.
        orders: candidate orders; defaults to :data:`SCAN_ORDERS`.

    Returns:
        The matching order names. Empty means none matched — either the mask is
        not the one this pack was cut from, or the extraction used an order not
        in the candidate set. More than one means the mask cannot tell them
        apart; see :func:`verify_with_cube`.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H, W]; got {mask.shape}")
    wanted = np.asarray(labels).squeeze().astype(np.int64)
    if wanted.ndim != 1:
        raise ValueError(f"labels must squeeze to 1-D; got {np.asarray(labels).shape}")

    candidates = SCAN_ORDERS if orders is None else orders
    matched = []
    for name, enumerate_in in candidates.items():
        coords = enumerate_in(mask)
        if len(coords) != len(wanted):
            continue
        if np.array_equal(mask[coords[:, 0], coords[:, 1]].astype(np.int64), wanted):
            matched.append(name)
    return matched


def _resolve_order(
    mask: np.ndarray,
    labels: np.ndarray,
    *,
    what: str,
    order: str | None,
    orders: dict[str, Callable[[np.ndarray], np.ndarray]] | None,
) -> str:
    """One matching order for ``mask``/``labels``, or a diagnostic exception."""
    if order is not None:
        if not match_orders(mask, labels, orders={order: SCAN_ORDERS[order]}):
            raise OrderRecoveryError(
                f"{what}: the requested order '{order}' does not reproduce the pack's label vector."
            )
        return order

    matched = match_orders(mask, labels, orders=orders)

    n_mask = int(np.count_nonzero(mask))
    n_labels = int(np.asarray(labels).squeeze().size)
    if not matched:
        counts = (
            ""
            if n_mask == n_labels
            else (
                f" The mask holds {n_mask} labelled pixels and the pack {n_labels} "
                f"samples, so this is very likely the wrong mask or the wrong split."
            )
        )
        raise OrderRecoveryError(
            f"{what}: no candidate scan order reproduces the pack's label "
            f"vector (tried {sorted(orders or SCAN_ORDERS)}).{counts}"
        )

    # Several orders can share a label sequence while enumerating the *same*
    # pixels in the same sequence — the two class-grouped orders always do when
    # the raster order they wrap is already class-sorted. That is not an
    # ambiguity: the table is the same whichever name is used.
    candidates = SCAN_ORDERS if orders is None else orders
    tables: dict[bytes, list[str]] = {}
    for name in matched:
        key = np.ascontiguousarray(candidates[name](mask), dtype=np.int64).tobytes()
        tables.setdefault(key, []).append(name)
    if len(tables) == 1:
        return matched[0]

    groups = " vs ".join("/".join(names) for names in tables.values())
    raise OrderRecoveryError(
        f"{what}: {groups} all reproduce the pack's label vector but disagree "
        f"on which pixel is which, so the mask alone cannot say. Settle it with "
        f"verify_with_cube() and pass order=."
    )


# ----------------------------------------------------------------------
# The recovered table
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SceneCoords:
    """Where every sample of a pre-patched dataset sits in its scene.

    Row ``i`` describes dataset index ``i`` **in the ``split="all"`` index
    space** — the train pack's samples first, then the test pack's, which is
    the order ``PatchedMultimodalDataset`` concatenates them in and therefore
    the order the episode sampler and the feature cache index.
    """

    scene: str
    #: ``(H, W)`` of the original scene.
    shape: tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray
    #: The pack's label for each index (1-based; 0 = unlabelled never appears).
    labels: np.ndarray
    #: 0 for a sample from the train pack, 1 from the test pack.
    pack: np.ndarray
    #: The recovered scan order per pack, e.g. ``{"train": "column_major"}``.
    orders: dict[str, str]

    def __post_init__(self) -> None:
        n = len(self.rows)
        if not (len(self.cols) == len(self.labels) == len(self.pack) == n):
            raise ValueError("rows, cols, labels and pack must be the same length")
        h, w = self.shape
        if n and (self.rows.max() >= h or self.cols.max() >= w):
            raise ValueError(f"coordinates fall outside the {self.shape} scene")
        flat = self.rows.astype(np.int64) * w + self.cols.astype(np.int64)
        if len(np.unique(flat)) != n:
            raise ValueError("two samples resolve to the same pixel; the order is wrong")

    def __len__(self) -> int:
        return len(self.rows)

    def to_map(
        self,
        values: np.ndarray,
        *,
        indices: np.ndarray | None = None,
        fill: int = 0,
        dtype: Any = np.int16,
    ) -> np.ndarray:
        """Scatter per-sample ``values`` onto a ``shape`` canvas.

        Args:
            values: one value per sample, or one per entry of ``indices``.
            indices: dataset indices ``values`` belongs to; ``None`` means all.
            fill: what unlabelled (and unaddressed) pixels get.
            dtype: canvas dtype.
        """
        values = np.asarray(values)
        idx = np.arange(len(self)) if indices is None else np.asarray(indices)
        if len(values) != len(idx):
            raise ValueError(f"{len(values)} values for {len(idx)} indices")
        canvas = np.full(self.shape, fill, dtype=dtype)
        canvas[self.rows[idx], self.cols[idx]] = values
        return canvas

    def gt_map(self) -> np.ndarray:
        """The labels scattered back onto the scene.

        Equal to the mask(s) recovery was run against — cheap self-check that
        the table addresses the pixels it claims to.
        """
        return self.to_map(self.labels)

    def save(self, path: str | Path) -> Path:
        """Write the table as an ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene=np.array(self.scene),
            shape=np.asarray(self.shape, dtype=np.int64),
            rows=self.rows,
            cols=self.cols,
            labels=self.labels,
            pack=self.pack,
            order_names=np.array(list(self.orders.keys())),
            order_values=np.array(list(self.orders.values())),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> SceneCoords:
        """Read a table written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as blob:
            return cls(
                scene=str(blob["scene"]),
                shape=(int(blob["shape"][0]), int(blob["shape"][1])),
                rows=blob["rows"],
                cols=blob["cols"],
                labels=blob["labels"],
                pack=blob["pack"],
                orders=dict(zip(blob["order_names"].tolist(), blob["order_values"].tolist())),
            )


def recover(
    scene: str,
    mask_train: np.ndarray,
    labels_train: np.ndarray,
    *,
    mask_test: np.ndarray | None = None,
    labels_test: np.ndarray | None = None,
    order_train: str | None = None,
    order_test: str | None = None,
    orders: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
) -> SceneCoords:
    """Recover the ``split="all"`` coordinate table for one scene.

    Args:
        scene: dataset key (``"houston"``, ``"trento"``, ``"muufl"``).
        mask_train: the scene mask the *train* pack was cut from, ``[H, W]``.
        labels_train: ``TrLabel.mat``'s vector.
        mask_test: the mask the *test* pack was cut from. ``None`` recovers the
            train pack alone.
        labels_test: ``TeLabel.mat``'s vector.
        order_train / order_test: force a scan order instead of recovering it —
            for the case where two orders match and the cube settled which.
        orders: candidate orders; defaults to :data:`SCAN_ORDERS`.

    Raises:
        OrderRecoveryError: if no order matches, or several do and none was
            forced.
    """
    mask_train = np.asarray(mask_train)
    name = _resolve_order(
        mask_train, labels_train, what=f"{scene} train pack", order=order_train, orders=orders
    )
    coords = scan_coordinates(mask_train, name)
    rows, cols = [coords[:, 0]], [coords[:, 1]]
    labels = [np.asarray(labels_train).squeeze().astype(np.int64)]
    pack = [np.zeros(len(coords), dtype=np.int8)]
    recovered = {"train": name}

    if mask_test is not None:
        if labels_test is None:
            raise ValueError("mask_test given without labels_test")
        mask_test = np.asarray(mask_test)
        if mask_test.shape != mask_train.shape:
            raise ValueError(
                f"train and test masks disagree on the scene shape: "
                f"{mask_train.shape} vs {mask_test.shape}"
            )
        name_te = _resolve_order(
            mask_test, labels_test, what=f"{scene} test pack", order=order_test, orders=orders
        )
        coords_te = scan_coordinates(mask_test, name_te)
        rows.append(coords_te[:, 0])
        cols.append(coords_te[:, 1])
        labels.append(np.asarray(labels_test).squeeze().astype(np.int64))
        pack.append(np.ones(len(coords_te), dtype=np.int8))
        recovered["test"] = name_te

    return SceneCoords(
        scene=scene,
        shape=(int(mask_train.shape[0]), int(mask_train.shape[1])),
        rows=np.concatenate(rows).astype(np.int32),
        cols=np.concatenate(cols).astype(np.int32),
        labels=np.concatenate(labels),
        pack=np.concatenate(pack),
        orders=recovered,
    )


# ----------------------------------------------------------------------
# Confirmation against the scene's own cube
# ----------------------------------------------------------------------


def verify_with_cube(
    coords: SceneCoords,
    cube: np.ndarray,
    patches: np.ndarray,
    *,
    indices: np.ndarray | None = None,
    patch_offset: int = 0,
    max_checks: int = 512,
    seed: int = 0,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Check that patch centres land on the pixels ``coords`` claims.

    The 11x11 patch centred on ``(row, col)`` has ``cube[row, col]`` at its
    centre whatever the border padding did to its edges, so this is an
    independent confirmation of the recovered order — and the tie-breaker when
    :func:`match_orders` returns more than one candidate.

    Args:
        coords: the recovered table.
        cube: the scene's HSI (or LiDAR) raster, ``[H, W, C]``, **as stored on
            disk** — un-normalised, since ``patches`` is too.
        patches: the pack's raw array, ``[n, P, P, C]``.
        indices: which dataset indices to check; ``None`` draws ``max_checks``
            of them at random.
        patch_offset: dataset index of ``patches[0]``. The test pack starts
            where the train pack ends in the ``split="all"`` space, so checking
            it means ``patch_offset=len(train pack)`` rather than materialising
            a padded copy of a multi-gigabyte array.
        max_checks: cap on how many centres to compare (the real packs are
            gigabytes; a few hundred settles the question).
        seed: RNG seed for that draw.
        atol: absolute tolerance on the centre spectrum.

    Returns:
        ``{"checked", "mismatched", "max_abs_diff", "examples"}``.
    """
    cube = np.asarray(cube)
    patches = np.asarray(patches)
    if cube.ndim != 3:
        raise ValueError(f"cube must be [H, W, C]; got {cube.shape}")
    if patches.ndim != 4:
        raise ValueError(f"patches must be [n, P, P, C]; got {patches.shape}")
    if cube.shape[:2] != coords.shape:
        raise ValueError(f"cube is {cube.shape[:2]}, coords describe a {coords.shape} scene")

    if indices is None:
        rng = np.random.default_rng(seed)
        pool = np.arange(patch_offset, min(len(coords), patch_offset + len(patches)))
        indices = rng.choice(pool, size=min(max_checks, len(pool)), replace=False)
    indices = np.sort(np.asarray(indices))
    local = indices - patch_offset
    if local.min() < 0 or local.max() >= len(patches):
        raise IndexError(
            f"indices {indices.min()}..{indices.max()} fall outside the "
            f"{len(patches)} patches starting at dataset index {patch_offset}"
        )

    centre = patches.shape[1] // 2
    got = patches[local, centre, centre, :]
    want = cube[coords.rows[indices], coords.cols[indices], :]
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    per_sample = diff.max(axis=1)
    bad = np.nonzero(per_sample > atol)[0]

    return {
        "checked": int(len(indices)),
        "mismatched": int(len(bad)),
        "max_abs_diff": float(per_sample.max()) if len(per_sample) else 0.0,
        "examples": [
            {
                "index": int(indices[b]),
                "row": int(coords.rows[indices[b]]),
                "col": int(coords.cols[indices[b]]),
                "max_abs_diff": float(per_sample[b]),
            }
            for b in bad[:5]
        ],
    }


# ----------------------------------------------------------------------
# .mat loading
# ----------------------------------------------------------------------


def load_mat_array(path: str | Path, key: str | None = None) -> np.ndarray:
    """Load one array from a ``.mat`` file.

    Mirrors ``PatchedMultimodalDataset._extract_array``: with no ``key``, the
    first non-metadata array wins, which is how the packs (single variable,
    always named ``Data``) and the usual mask files are shaped.
    """
    import scipy.io as sio

    blob = sio.loadmat(str(path))
    if key is not None:
        if key not in blob:
            available = sorted(k for k in blob if not k.startswith("__"))
            raise KeyError(f"'{key}' not in {path}; variables: {available}")
        return np.asarray(blob[key])
    for name, value in blob.items():
        if not name.startswith("__") and isinstance(value, np.ndarray):
            return np.asarray(value)
    raise ValueError(f"no data array in {path}; keys: {list(blob)}")


def pack_dir(scene: str, data_root: str | Path = "data/raw") -> Path:
    """The directory holding ``scene``'s pre-patched pack.

    Read off the dataset class's own ``folder_name`` rather than re-spelling
    ``Houston11x11`` here: those on-disk literals are DO-NOT-RENAME and belong
    in one place (PAPER_CANON §7.3). ``folder_name`` is a constant property, so
    it needs no instance — and constructing one would load the pack.
    """
    from coffe.data.datasets import get_spec

    folder = get_spec(scene).patched_cls.folder_name.fget(None)  # type: ignore[attr-defined]
    return Path(data_root) / folder

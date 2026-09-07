#!/usr/bin/env python
"""Recover the pixel coordinate of every sample in a pre-patched scene pack.

The packs under ``data/raw/<Scene>11x11/`` carry no coordinates (see
``coffe.data.scene_coords``), so a prediction can be attributed to a sample
index but not to a pixel. This script recovers the missing table by finding the
scan order over the scene's ground-truth mask whose label sequence reproduces
the pack's label vector exactly, and writes it as an ``.npz`` that
``scripts/predict_map.py`` consumes.

You supply the masks: they are not in the packs and not in this repository.
Houston's are the DFC2013 ``TRLabel`` / ``TSLabel`` rasters (349x1905), Trento's
its 166x600 ground truth, MUUFL's the 325x220 one — from the owners named in
the README.

Usage:
    # Two masks, one per pack (the usual case: a fixed train/test split)
    python scripts/recover_patch_coords.py --dataset houston \
        --gt-train /path/TRLabel.mat --gt-test /path/TSLabel.mat

    # One mask covering one pack only
    python scripts/recover_patch_coords.py --dataset muufl \
        --gt-train /path/gt.mat --pack Tr

    # Confirm the recovered order against the scene's own cube
    python scripts/recover_patch_coords.py --dataset houston \
        --gt-train /path/TRLabel.mat --gt-test /path/TSLabel.mat \
        --hsi /path/Houston.mat --max-checks 512

Exit status is 1 when recovery fails or the cube check finds a mismatch, so
this is safe to gate a pipeline on.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from coffe.data.scene_coords import (
    SCAN_ORDERS,
    OrderRecoveryError,
    load_mat_array,
    match_orders,
    pack_dir,
    recover,
    verify_with_cube,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover per-sample (row, col) for a pre-patched scene pack"
    )
    parser.add_argument("--dataset", required=True, choices=["houston", "trento", "muufl"])
    parser.add_argument("--data-root", default="data/raw", help="where the packs live")
    parser.add_argument("--gt-train", required=True, help=".mat mask the *train* pack was cut from")
    parser.add_argument("--gt-test", default=None, help=".mat mask for the test pack")
    parser.add_argument("--gt-train-key", default=None, help="variable name inside --gt-train")
    parser.add_argument("--gt-test-key", default=None, help="variable name inside --gt-test")
    parser.add_argument(
        "--pack",
        default="Tr",
        choices=["Tr", "Te"],
        help="which pack --gt-train describes when no --gt-test is given",
    )
    parser.add_argument(
        "--order-train",
        default=None,
        choices=sorted(SCAN_ORDERS),
        help="force a scan order instead of recovering it",
    )
    parser.add_argument("--order-test", default=None, choices=sorted(SCAN_ORDERS))
    parser.add_argument(
        "--hsi",
        default=None,
        help="the scene's full HSI cube [H, W, C] — turns on the centre-pixel "
        "confirmation, which is also how a tie between two matching orders is "
        "settled",
    )
    parser.add_argument("--hsi-key", default=None, help="variable name inside --hsi")
    parser.add_argument("--max-checks", type=int, default=512, help="centres to compare")
    parser.add_argument(
        "--out",
        default=None,
        help="output .npz (default: results/coords_<dataset>.npz)",
    )
    parser.add_argument(
        "--report", default=None, help="optional JSON summary of what was recovered"
    )
    return parser


def _load_labels(dataset: str, data_root: str, suffix: str) -> np.ndarray:
    path = pack_dir(dataset, data_root) / f"{suffix}Label.mat"
    if not path.exists():
        raise FileNotFoundError(f"label file not found: {path}")
    return load_mat_array(path).squeeze().astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    swapped = args.gt_test is None and args.pack == "Te"
    mask_train = load_mat_array(args.gt_train, args.gt_train_key)
    labels_train = _load_labels(args.dataset, args.data_root, "Te" if swapped else "Tr")

    mask_test = labels_test = None
    if args.gt_test is not None:
        mask_test = load_mat_array(args.gt_test, args.gt_test_key)
        labels_test = _load_labels(args.dataset, args.data_root, "Te")

    print(f"[scan] {args.dataset}: mask {mask_train.shape}, {len(labels_train)} samples")
    for name, matched in (
        ("train" if not swapped else "test", match_orders(mask_train, labels_train)),
        *([("test", match_orders(mask_test, labels_test))] if mask_test is not None else []),
    ):
        print(f"[scan] {name} pack matches: {matched or 'none'}")

    try:
        coords = recover(
            args.dataset,
            mask_train,
            labels_train,
            mask_test=mask_test,
            labels_test=labels_test,
            order_train=args.order_train,
            order_test=args.order_test,
        )
    except OrderRecoveryError as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    if swapped:
        # The single mask described the *test* pack; say so on the table, whose
        # index space is then the dataset's `split="test"`.
        coords = replace(
            coords,
            pack=np.ones(len(coords), dtype=np.int8),
            orders={"test": coords.orders["train"]},
        )

    if not np.array_equal(coords.gt_map()[mask_train > 0], mask_train[mask_train > 0]):
        print(
            "[fail] the recovered table does not reproduce the mask it came from", file=sys.stderr
        )
        return 1

    report: dict = {
        "dataset": args.dataset,
        "scene_shape": list(coords.shape),
        "num_samples": len(coords),
        "orders": coords.orders,
    }

    if args.hsi is not None:
        cube = load_mat_array(args.hsi, args.hsi_key)
        checks = {}
        for suffix, pack_id in (("Tr", 0), ("Te", 1)):
            in_pack = np.nonzero(coords.pack == pack_id)[0]
            if not len(in_pack):
                continue
            patch_path = pack_dir(args.dataset, args.data_root) / f"HSI_{suffix}.mat"
            print(f"[cube] loading {patch_path} (this is the slow part)")
            patches = load_mat_array(patch_path)
            rng = np.random.default_rng(0)
            picked = rng.choice(in_pack, size=min(args.max_checks, len(in_pack)), replace=False)
            result = verify_with_cube(
                coords,
                cube,
                patches,
                indices=picked,
                # patches[0] is this pack's first sample, which sits at
                # in_pack[0] in the dataset index space.
                patch_offset=int(in_pack[0]),
            )
            checks[suffix] = result
            print(
                f"[cube] {suffix}: {result['checked']} centres checked, "
                f"{result['mismatched']} mismatched, max |diff| {result['max_abs_diff']:.3g}"
            )
            del patches
        report["cube_check"] = checks
        if any(c["mismatched"] for c in checks.values()):
            print("[fail] patch centres do not sit on the recovered pixels", file=sys.stderr)
            return 1

    out = Path(args.out or f"results/coords_{args.dataset}.npz")
    coords.save(out)
    print(f"[done] {len(coords)} coordinates -> {out}  orders={coords.orders}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(f"[done] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

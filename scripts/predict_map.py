#!/usr/bin/env python
"""Classify every labelled pixel once and paint the map next to the ground truth.

PAPER_CANON §4's protocol averages 1000-2000 episodes, each with its own 5-shot
support draw, so a pixel has a *distribution* of decisions rather than one. This
entry point runs the same encoder, the same features and the same
nearest-class-mean classifier in the one shape a map needs — support drawn once
at a seed, N = every class in the scene, each remaining labelled sample
classified exactly once (``coffe.eval.prediction_map``) — and scatters the
result onto the scene using a coordinate table recovered by
``scripts/recover_patch_coords.py``.

The OA printed here is that single pass, not a paper cell: expect it inside the
published confidence interval, not equal to the published mean.

Without ``--coords`` the pass still runs and the per-sample decisions are saved;
only the painting is skipped. That is the useful mode until the scene masks are
in hand, since the coordinate table is the only part that needs them.

Usage:
    # decisions only
    python scripts/predict_map.py --checkpoint <run>/checkpoints/checkpoint_epoch_950.pth \
        --dataset houston --out-dir results/maps/houston

    # decisions + the map, once coords_houston.npz exists
    python scripts/predict_map.py --checkpoint <ckpt> --dataset houston \
        --coords results/coords_houston.npz --out-dir results/maps/houston
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from coffe.data.scene_coords import SceneCoords
from coffe.eval.episodic import CLASS_NAMES, DATASETS, load_model_with_checkpoint
from coffe.eval.feature_cache import encode_dataset, load_features, save_features
from coffe.eval.prediction_map import predict_all
from coffe.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("predict_map")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One NCM decision per labelled pixel, as a classification map"
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint path, or 'random'")
    parser.add_argument("--dataset", required=True, choices=["houston", "trento", "muufl"])
    parser.add_argument("--data-root", default="./data/raw")
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "test", "all"],
        help="must be the split the coordinate table describes (default: all, "
        "what every paper eval used)",
    )
    parser.add_argument("--patch-size", type=int, default=11)

    # Architecture — the published CoFFE defaults (PAPER_CANON §2, §8 D3).
    parser.add_argument(
        "--model", default="coffe", choices=["coffe", "coffe_aux_token", "mft_original"]
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lambda-factor", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-dim", type=int, default=512, help="MFT control only")
    parser.add_argument("--attention-type", default="mcross", help="MFT control only")
    parser.add_argument(
        "--use-projection",
        action="store_true",
        default=False,
        help="the head is discarded at eval in every paper run (PAPER_CANON §4)",
    )
    parser.add_argument("--no-aux", dest="use_aux", action="store_false", default=True)
    parser.add_argument("--pool-sigma", type=float, default=None)

    # Protocol
    parser.add_argument("--k-shot", type=int, default=5, help="PAPER_CANON §4: 5")
    parser.add_argument("--distance-metric", default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42, help="fixes the support draw")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="paint one map per support draw; --seed is used when absent",
    )

    # Plumbing
    parser.add_argument("--device", default="auto", help="'cuda' | 'cuda:N' | 'cpu' | 'auto'")
    parser.add_argument("--batch-size", type=int, default=256, help="patches per forward pass")
    parser.add_argument(
        "--feature-cache",
        default=None,
        help="path stem for the encoded features; reused when it matches",
    )
    parser.add_argument("--coords", default=None, help="table from recover_patch_coords.py")
    parser.add_argument("--out-dir", default=None, help="default: results/maps/<dataset>")
    parser.add_argument("--no-plot", action="store_true", help="skip the PNG")
    return parser


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    out_dir = Path(args.out_dir or f"results/maps/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dataset = DATASETS[args.dataset](
        data_root=args.data_root,
        patch_size=args.patch_size,
        split=args.split,
        normalize=True,
    )
    logger.info("[data] %s %s split: %d samples", args.dataset, args.split, len(dataset))

    coords = None
    if args.coords:
        coords = SceneCoords.load(args.coords)
        if len(coords) != len(dataset):
            raise SystemExit(
                f"the coordinate table has {len(coords)} entries but the "
                f"'{args.split}' split has {len(dataset)} samples — they must "
                f"share an index space (recover the table for this split)"
            )
        logger.info("[coords] %s scene, orders=%s", coords.shape, coords.orders)

    model_config = {
        "name": args.model,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "lambda_factor": args.lambda_factor,
        "dropout": args.dropout,
        "mlp_dim": args.mlp_dim,
        "attention_type": args.attention_type,
        "patch_size": args.patch_size,
        "use_projection": args.use_projection,
        "use_aux": args.use_aux,
        "distance_metric": args.distance_metric,
        "temperature": args.temperature,
        "prototype_mode": "mean_features",
        "pool_sigma": args.pool_sigma,
    }
    model = load_model_with_checkpoint(args.checkpoint, args.dataset, model_config, device)

    # The encoder is frozen, so every patch has exactly one feature vector:
    # encode the scene once and index it (coffe.eval.feature_cache).
    expect = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "model": model_config,
    }
    features = None
    if args.feature_cache:
        cached = load_features(args.feature_cache, expect=expect)
        if cached is not None:
            features = cached[0]
    if features is None:
        features = encode_dataset(model, dataset, device=device, batch_size=args.batch_size)
        if args.feature_cache:
            save_features(args.feature_cache, features, expect)

    class_names = CLASS_NAMES.get(args.dataset)
    summaries = []
    for seed in args.seeds or [args.seed]:
        prediction = predict_all(
            features,
            dataset,
            scene=args.dataset,
            k_shot=args.k_shot,
            seed=seed,
            metric=args.distance_metric,
            temperature=args.temperature,
            device=device,
        )
        metrics = prediction.metrics()
        logger.info(
            "[seed %d] %d queries, OA %.2f  AA %.2f  Kappa %.2f",
            seed,
            len(prediction.query_indices),
            metrics["OA"],
            metrics["AA"],
            metrics["Kappa"],
        )

        stem = f"{args.dataset}_seed{seed}"
        np.savez_compressed(
            out_dir / f"decisions_{stem}.npz",
            support_indices=prediction.support_indices,
            support_classes=prediction.support_classes,
            query_indices=prediction.query_indices,
            query_classes=prediction.query_classes,
            predictions=prediction.predictions,
            classes=prediction.classes,
        )
        summary = {**prediction.to_dict(), "checkpoint": str(args.checkpoint)}
        summaries.append(summary)

        if coords is not None:
            maps = prediction.to_maps(coords)
            np.savez_compressed(out_dir / f"maps_{stem}.npz", **maps)
            if not args.no_plot:
                from coffe.utils.visualization import plot_classification_map

                plot_classification_map(
                    maps,
                    class_names=class_names,
                    dataset_label=args.dataset,
                    title=(
                        f"{args.dataset} — {args.k_shot}-shot "
                        f"{args.distance_metric} NCM, seed {seed} "
                        f"(OA {metrics['OA']:.2f})"
                    ),
                    save_path=str(out_dir / f"map_{stem}.png"),
                )
                logger.info("[maps] %s", out_dir / f"map_{stem}.png")
            else:
                logger.info("[maps] %s", out_dir / f"maps_{stem}.npz")

    (out_dir / f"summary_{args.dataset}.json").write_text(
        json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2) + "\n"
    )
    logger.info("[done] %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

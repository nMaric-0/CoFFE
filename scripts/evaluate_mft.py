#!/usr/bin/env python
"""
Few-shot evaluation for the **MFT architectural control** (Roy et al.).

This is the control pipeline. ``scripts/evaluate.py`` is CoFFE's, and
``scripts/evaluate_hypersigma.py`` is the foundation-model route's — one entry
point per route, so each carries its own architecture defaults and nothing has
to be re-specified on the command line to get the published configuration.

The control is the original MFT: separate HSI and auxiliary front ends fused
through an **external** token, pretrained under the same masked objectives as
CoFFE and evaluated under the same protocol (PAPER_CANON §1). Comparing it
against CoFFE's input-level fusion is the paper's architectural claim, so the
two routes are deliberately not folded into one CLI whose defaults could only
be right for one of them.

Protocol is identical to CoFFE's: frozen encoder, N-way (every class in the
scene), K = 5, class-balanced queries, Euclidean nearest-class-mean
(PAPER_CANON §4). The evaluator body is shared — ``coffe.eval.episodic`` — so
the two pipelines cannot drift apart on the protocol, only on architecture.

Usage:
    # Table 2 control cell, e.g. MFT SimMIM token / Houston
    python scripts/evaluate_mft.py \
        --checkpoint experiments/<run>/checkpoints/checkpoint_epoch_950.pth \
        --dataset houston

Note the evaluated epoch is not the final one: PAPER_CANON §8 D17 puts all six
MFT cells at epoch 950. Pass that checkpoint explicitly.
"""

import argparse
import sys
from pathlib import Path

# Keep the repo root importable so `python scripts/evaluate_mft.py` works in a
# tree that has not been `pip install -e .`-ed, exactly like the other entry
# points under scripts/. Harmless when the package is installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from coffe.eval.episodic import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot evaluation for the MFT architectural control "
            "(nearest-class-mean on frozen features)"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path, or 'random' for an untrained encoder",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["houston", "trento", "muufl"]
    )

    # Few-shot settings — identical to CoFFE's pipeline (PAPER_CANON §4).
    parser.add_argument(
        "--n-way",
        type=int,
        default=None,
        help="Classes per episode. Default (and the paper protocol): all "
        "classes in the scene, i.e. N = 15 Houston / 6 Trento / 11 MUUFL.",
    )
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--k-query", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=1000)

    # Dataset
    parser.add_argument("--data-root", type=str, default="./data/raw")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument("--patch-size", type=int, default=11)

    # Architecture — the faithful MFT control (dim = FM*4 with FM = 16, and the
    # authors' fixed 512-wide feed-forward). These differ from CoFFE's and are
    # the reason this is a separate entry point.
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer depth")
    parser.add_argument(
        "--mlp-dim", type=int, default=512, help="Feed-forward width (faithful MFT: 512)"
    )
    parser.add_argument(
        "--attention-type",
        type=str,
        default="mcross",
        choices=["mcross", "standard"],
        help="MFT cross-attention ('mcross', the authors' MCrossPA) or plain self-attention",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use-aux",
        action="store_true",
        default=True,
        help="Fuse aux (LiDAR) through the external token (default: True). "
        "Must match how the checkpoint was pretrained.",
    )
    parser.add_argument(
        "--no-aux",
        dest="use_aux",
        action="store_false",
        help="HSI-only. Note MFTOriginal rejects this: its architecture is built "
        "around the external aux token.",
    )

    # Nearest-class-mean settings (PAPER_CANON §4). Euclidean is the protocol;
    # cosine remains selectable as an option value only (§1).
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance to the class means (default: euclidean, the paper protocol).",
    )
    parser.add_argument(
        "--temperature", type=float, default=10.0, help="Logit scaling on the cosine path"
    )
    parser.add_argument(
        "--prototype-mode",
        type=str,
        default="mean_features",
        choices=["mean_features", "mean_distances"],
        help="Class mean computation: 'mean_features' (the paper protocol) or 'mean_distances'",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate once per seed and report aggregate stats, e.g. --seeds 42 123 456",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU (or use --device cpu)")
    parser.add_argument(
        "--device", type=str, default="auto", help="'cuda' | 'cuda:N' | 'auto' | 'cpu'"
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--num-example-episodes", type=int, default=3)
    parser.add_argument("--max-tsne-samples", type=int, default=0)

    args = parser.parse_args()

    # Route selection: this entry point *is* the control, so it is set here
    # rather than exposed as a flag.
    args.name = "mft_original"

    # Keys the shared evaluator body reads but which are meaningless on this
    # route, set so no caller has to know that.
    #   lambda_factor: CoFFE-only. MFTOriginal.eval_patch_embeddings is an
    #     explicit pass-through and its eval configs record None (§8 D3.4).
    #   use_projection / proj_*: the control has no projection head.
    args.lambda_factor = None
    args.use_projection = False
    args.proj_hidden_dim = None
    args.proj_num_layers = 2
    args.proj_l2_normalize = True
    args.pool_sigma = None  # MFT packs the fused CLS as one token; not applicable

    main(args)

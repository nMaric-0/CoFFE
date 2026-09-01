#!/usr/bin/env python
"""HyperSIGMA spatial-branch-only adaptation + evaluation, 100-band variant.

Runs the ``evaluate_hypersigma_pca100.ipynb`` recipe for ONE dataset with the
pipeline ``mode`` hardwired to ``spatial_only``: the spatial branch is fed
**100 channels** (vs the 3-band headline) while KEEPING the adapted 11x11 /
``spat_patch_k=3`` geometry. Only the SpatViT random-init pieces are adapted
(MAE), and evaluation uses pooled SpatViT features (``spat_pool``) -- the
SpecViT forward is skipped entirely.

How "100 bands not 3" is realised per dataset (decided from the registry band
count, identical to the pca100 config's documented behaviour):

* houston (144 bands >= 100): real PCA->100 (``pca_houston_100band.pkl`` +
  ``_stats.pkl``); fit it first with
  ``python scripts/fit_pca_hypersigma.py --dataset houston --n-components 100``.
* trento (63) / muufl (64) (< 100): PCA->100 is impossible, so the dual encoder
  spectrally resamples the raw bands up to 100 via ``spat_resample_to=100``
  when the ``pca_<ds>_100band.pkl`` pickle is absent (no PCA standardization).

``spat_resample_to=100`` is set uniformly; when the PCA pickle exists (houston)
it takes precedence and PCA is used, otherwise resample kicks in.

Each run is packaged under ``experiments/<name>/`` (config, metadata, logs,
results.json, plots) via the same runner functions the notebooks use, so
``scripts/aggregate_experiment_results.py`` picks the results up.

Usage (one dataset):
    python scripts/run_hypersigma_spatial_pca100.py --dataset houston --device cuda:1

Typically driven for all three datasets by
``scripts/run_hypersigma_spatial_pca100.sh``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import get_spec  # noqa: E402
from lib.adapt_runner import run_adapt_hypersigma  # noqa: E402
from lib.eval_runner import run_hypersigma_evaluation  # noqa: E402
from lib.experiments import ExperimentLogger  # noqa: E402

logger = logging.getLogger(__name__)

# The spatial branch is fed this many channels (vs the 3-band headline).
SPAT_COMPONENTS = 100


def run_dataset(
    dataset: str,
    *,
    device: str = "cuda",
    epochs: int = 2000,
    lr: float = 1e-5,
    batch_size: int = 128,
    k_shot: int = 5,
    k_query: int = 100,
    num_episodes: int = 2000,
    overwrite: bool = True,
) -> dict:
    """Adapt (spatial_only) + evaluate (spat_pool) one dataset at 100 bands.

    Adaptation is Level-2: the pretrained SpatViT transformer body stays frozen
    (it was pretrained on 64x64/patch-8/100ch) and only the parts that the new
    11x11/patch-3/100ch input geometry forces to change -- ``patch_embed.proj``
    (kernel 8x8 -> 3x3) and ``pos_embed`` (64 -> 16 tokens) -- plus the MAE
    decoder are trained. Training schedule (lr 1e-5, min_lr 5e-7, warmup 100,
    batch 128) mirrors notebooks/evaluate_hypersigma_pca100.ipynb.
    """
    spec = get_spec(dataset)
    bands = spec.hsi_channels

    # 100-band spatial input. houston (>=100 bands) uses a fitted PCA->100 +
    # standardization stats; trento/muufl (<100) have no such pickle, so the
    # dual encoder spectrally resamples raw bands -> 100 (spat_resample_to) and
    # there is no PCA standardization to apply.
    pca_spat_path = f"checkpoints/hypersigma/pca_{dataset}_{SPAT_COMPONENTS}band.pkl"
    pca_stats_path = (
        f"checkpoints/hypersigma/pca_{dataset}_{SPAT_COMPONENTS}band_stats.pkl"
        if bands >= SPAT_COMPONENTS
        else None
    )
    uses_pca = bands >= SPAT_COMPONENTS

    # Names follow the established convention (mirrors the pca100 notebook with
    # mode="spatial_only"); the C<N>way tag uses the dataset's full class count.
    adapt_experiment_name = f"hypersigma_adapt_{dataset}_pca{SPAT_COMPONENTS}_spatial_only_run1"
    experiment_name = f"hypersigma_{dataset}_pca{SPAT_COMPONENTS}_spatial_only_run1"
    eval_name = (
        f"hypersigma_{dataset}_pca{SPAT_COMPONENTS}_C{spec.num_classes}way_"
        f"{k_shot}shot_adapted_spatial_only_run1"
    )
    adapted_checkpoint = (
        f"experiments/{adapt_experiment_name}/checkpoints/checkpoint_final.pth"
    )

    logger.info(
        "[%s] bands=%d -> spatial input=%d (%s); epochs=%d lr=%g batch=%d device=%s",
        dataset, bands, SPAT_COMPONENTS,
        "PCA->100 + stats" if uses_pca else "spectral resample -> 100",
        epochs, lr, batch_size, device,
    )

    # --- Step 5: Level-2 MAE adaptation (spatial branch only) ---------------
    # Schedule mirrors notebooks/evaluate_hypersigma_pca100.ipynb: the pretrained
    # SpatViT body stays frozen (freeze_body defaults to True), so only
    # patch_embed.proj + pos_embed + the MAE decoder train. lr=1e-5 (vs the base
    # config's 1.5e-4) avoids disturbing the frozen-body feature space.
    overrides = {
        "model": {
            "adapt_mode": "spatial_only",
            "spat_patch_k": 3,
            "spat_resample_to": SPAT_COMPONENTS,
        },
        "paths": {
            "pca_spat_path": pca_spat_path,
            "pca_stats_path": pca_stats_path,
        },
        "pretrain": {
            "epochs": epochs,
            "lr": lr,
            "min_lr": 5e-7,
            "warmup_epochs": 100,
            "batch_size": batch_size,
        },
        "hardware": {"device": device},
    }
    adapt_description = (
        f"HyperSIGMA spatial-only Level-2 MAE adaptation on {dataset}, 100-band "
        f"spatial input ({'PCA->100' if uses_pca else 'spectral resample->100'}), "
        "adapted 11x11/patch-3 geometry. Only the SpatViT random-init pieces are "
        "trained; SpecViT + SEM are unused."
    )
    adapt_exp = run_adapt_hypersigma(
        name=adapt_experiment_name,
        description=adapt_description,
        config=f"configs/hypersigma/{dataset}_patchnative_joint_sem.yaml",
        overrides=overrides,
        overwrite=overwrite,
    )
    logger.info("[%s] adaptation finished -> %s", dataset, adapt_exp.root)

    # --- Step 6a: bootstrap the umbrella experiment dir (start_eval needs it) -
    exp_logger = ExperimentLogger(repo_root=PROJECT_ROOT)
    try:
        exp_logger.get_experiment(experiment_name)
    except FileNotFoundError:
        exp_logger.start_pretrain(
            name=experiment_name,
            description=(
                f"HyperSIGMA {dataset} spatial-only, 100-band spatial input "
                "(pca100 variant) few-shot evaluation umbrella."
            ),
            config={"meta": "hypersigma spatial_only pca100; see adapt experiment for training"},
        )

    # --- Step 6b: few-shot evaluation (spatial branch only) -----------------
    eval_params = {
        "dataset": dataset,
        "k_shot": k_shot,
        "k_query": k_query,
        "num_episodes": num_episodes,
        "split": "all",
        "seed": 42,
        "mode": "spat_pool",
        "device": device,
        "spat_patch_k": 3,
        "spat_resample_to": SPAT_COMPONENTS,
        "pca_spat_path": pca_spat_path,
        "pca_stats_path": pca_stats_path,
        "spat_ckpt": "checkpoints/hypersigma/spat-vit-base.pth",
        "spec_ckpt": "checkpoints/hypersigma/spec-vit-base.pth",
        "distance_metric": "euclidean",
        "temperature": 10.0,
        "prototype_mode": "mean_features",
        "num_example_episodes": 3,
        "max_tsne_samples": 100,
        "no_plots": False,
    }  # n_way omitted -> C-way (all classes), matching the pca100 notebook
    ev = run_hypersigma_evaluation(
        experiment_name=experiment_name,
        eval_name=eval_name,
        adapted_checkpoint=adapted_checkpoint,
        eval_params=eval_params,
        overwrite=overwrite,
    )
    logger.info("[%s] evaluation finished -> %s", dataset, ev.root)
    return {"adapt_dir": str(adapt_exp.root), "eval_dir": str(ev.root)}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["houston", "trento", "muufl"])
    parser.add_argument("--device", default="cuda",
                        help='Device for adapt + eval, e.g. "cuda", "cuda:1", "cpu".')
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Adaptation LR for the trainable input/decoder pieces "
                             "(matches evaluate_hypersigma_pca100.ipynb; base config is 1.5e-4).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--k-query", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Fail instead of reusing an existing experiment dir.")
    args = parser.parse_args()

    out = run_dataset(
        args.dataset,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.num_episodes,
        overwrite=not args.no_overwrite,
    )
    print(f"\n[{args.dataset}] adapt: {out['adapt_dir']}")
    print(f"[{args.dataset}] eval : {out['eval_dir']}")


if __name__ == "__main__":
    main()

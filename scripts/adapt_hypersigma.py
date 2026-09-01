#!/usr/bin/env python
"""Level-2 MAE adaptation of HyperSIGMA on a target dataset.

Trains the small randomly-initialized components of the dual encoder
(SpatViT ``patch_embed`` + ``pos_embed``, SpecViT ``spat_map`` +
``pos_embed`` + ``l1``, plus the SEM and the ``UnifiedBandMasking``
fill values) using masked HSI reconstruction. SpatViT/SpecViT
transformer bodies remain frozen.

Optimizer recipe follows the CoFFE pretraining configs: lr=1.5e-4,
weight_decay=0.05, warmup=100, batch_size=64, AMP off. The decoders are
HyperSIGMA-specific (`spat_decoder_hidden`/`spec_decoder_hidden` 256,
`fused_decoder_hidden` 512) and there is no centre-weighting here. Masking here is HyperSIGMA's own 75% token masking
(``pretrain.mask_ratio``), not CoFFE's band/token pair, and the epoch count is
per-config (2000-3000).

The target dataset is selected entirely by the ``data.dataset`` field
of the config (``houston`` | ``trento`` | ``muufl``). Band count and the
PCA / checkpoint paths default to the conventions in
``data/datasets/registry.py`` when not given explicitly, so switching
datasets is a one-field change.

Usage:
    python scripts/adapt_hypersigma.py \
        --config configs/hypersigma/houston_patchnative_joint_sem.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import get_spec  # noqa: E402
from models.hypersigma.hypersigma_dual import HyperSIGMADual  # noqa: E402
from pretrain.hypersigma_mae import HyperSIGMAMaskedAdaptation  # noqa: E402
from utils.io import load_config  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)


def _build_dataset(config: dict):
    data_cfg = config.get("data", {})
    name = data_cfg.get("dataset", "houston")
    Dataset = get_spec(name).patched_cls
    ds = Dataset(
        data_root=data_cfg.get("data_root", "./data/raw"),
        patch_size=data_cfg.get("patch_size", 11),
        split="all",
        normalize=True,
    )
    logger.info("Loaded %s (split=all): %d samples", name, len(ds))
    return ds, name


def _resolve_hsi_channels(model_cfg: dict, dataset_name: str) -> int:
    """Band count from the registry; validate any config override matches.

    A mismatch between the configured ``hsi_channels`` and the dataset's
    true band count would silently build a model with the wrong input
    width, so we fail loudly instead.
    """
    expected = get_spec(dataset_name).hsi_channels
    configured = model_cfg.get("hsi_channels")
    if configured is not None and int(configured) != expected:
        raise ValueError(
            f"model.hsi_channels={configured} does not match dataset "
            f"'{dataset_name}' ({expected} bands). Remove the override or "
            f"fix it; band count is derived from data/datasets/registry.py."
        )
    return expected


def _build_model(config: dict, dataset_name: str) -> HyperSIGMAMaskedAdaptation:
    model_cfg = config.get("model", {})
    paths = config.get("paths", {})
    pretrain_cfg = config.get("pretrain", {})
    spec = get_spec(dataset_name)
    hsi_channels = _resolve_hsi_channels(model_cfg, dataset_name)

    dual = HyperSIGMADual(
        pca_spat_path=paths.get("pca_spat_path", spec.pca_spat_path()),
        spat_ckpt=paths.get("spat_ckpt", "checkpoints/hypersigma/spat-vit-base.pth"),
        spec_ckpt=paths.get("spec_ckpt", "checkpoints/hypersigma/spec-vit-base.pth"),
        hsi_channels=hsi_channels,
        spat_patch_k=model_cfg.get("spat_patch_k", 3),
        freeze_body=True,
        embed_dim=model_cfg.get("embed_dim", 768),
        num_tokens=model_cfg.get("num_tokens", 100),
        dr_dim=model_cfg.get("dr_dim", 128),
        num_stages=model_cfg.get("num_stages", 4),
        pca_stats_path=paths.get("pca_stats_path", spec.pca_stats_path()),
        # PCA-100 variant: target a fixed spatial channel count; datasets below it
        # (Trento 63, MUUFL 64) spectrally resample up to it (default None = off).
        spat_resample_to=model_cfg.get("spat_resample_to", None),
        # Native-geometry SEM-tuning ablation (default off -> existing behavior).
        native_geometry=model_cfg.get("native_geometry", False),
        input_fit=model_cfg.get("input_fit", "upscale"),
        pad_anchor=model_cfg.get("pad_anchor", "center"),
        interp_mode=model_cfg.get("interp_mode", "bicubic"),
        native_pca_spat_path=paths.get("native_pca_spat_path", None),
    )
    dual.log_sanity()

    return HyperSIGMAMaskedAdaptation(
        dual=dual,
        adapt_mode=model_cfg.get("adapt_mode", "joint_sem"),
        hsi_channels=hsi_channels,
        patch_size=config.get("data", {}).get("patch_size", 11),
        mask_ratio=pretrain_cfg.get("mask_ratio", 0.75),
        spat_decoder_hidden=pretrain_cfg.get("spat_decoder_hidden", 256),
        spec_decoder_hidden=pretrain_cfg.get("spec_decoder_hidden", 256),
        fused_decoder_hidden=pretrain_cfg.get("fused_decoder_hidden", 512),
        decoder_dropout=pretrain_cfg.get("decoder_dropout", 0.0),
    )


def _build_loaders(dataset, config: dict, seed: int):
    from sklearn.model_selection import StratifiedShuffleSplit

    pretrain_cfg = config.get("pretrain", {})
    data_cfg = config.get("data", {})
    val_split = pretrain_cfg.get("val_split", 0.1)
    labels = dataset.labels.numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=pretrain_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=pretrain_cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, val_loader


def _trainable_parameter_groups(model: HyperSIGMAMaskedAdaptation):
    params = [p for p in model.parameters() if p.requires_grad]
    return params


def run_adapt(config: dict, checkpoint_dir: str, log_dir: str) -> dict:
    """Programmatic entry-point used by the notebook runner."""
    hardware = config.get("hardware", {})
    seed = hardware.get("seed", 42)
    set_seed(seed, deterministic=hardware.get("deterministic", False))
    device = hardware.get("device", "cuda")
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available; falling back to CPU")
    elif device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = str(device)
    cuda_device_name = ""
    if resolved_device.startswith("cuda"):
        idx = 0
        if ":" in resolved_device:
            try:
                idx = int(resolved_device.split(":", 1)[1])
            except ValueError:
                idx = 0
        try:
            cuda_device_name = torch.cuda.get_device_name(idx)
        except Exception:
            cuda_device_name = ""
    logger.info("Using device: %s (%s)", resolved_device, cuda_device_name or "n/a")

    dataset, dataset_name = _build_dataset(config)

    model = _build_model(config, dataset_name).to(device)

    counts = model.parameter_counts()
    logger.info(
        "[HyperSIGMA-MAE] adapt_mode=%s mask_ratio=%.2f", model.adapt_mode, model.mask_ratio,
    )
    logger.info(
        "[HyperSIGMA-MAE] Adaptation module params: trainable=%d, frozen=%d",
        counts["total_trainable"], counts["total_frozen"],
    )
    for k, v in counts.items():
        if k.startswith("total_"):
            continue
        logger.info("[HyperSIGMA-MAE]   %s: %d", k, v)

    train_loader, val_loader = _build_loaders(dataset, config, seed)

    pretrain_cfg = config.get("pretrain", {})
    epochs = pretrain_cfg.get("epochs", 3000)
    lr = pretrain_cfg.get("lr", 1.5e-5)
    min_lr = pretrain_cfg.get("min_lr", 1e-6)
    warmup_epochs = pretrain_cfg.get("warmup_epochs", 100)
    weight_decay = pretrain_cfg.get("weight_decay", 0.05)
    grad_clip = pretrain_cfg.get("grad_clip", 1.0)
    save_interval = pretrain_cfg.get("save_interval", 100)
    # Always-snapshot milestones — saved even when below save_interval, so
    # you get a quick look at the early-training trajectory regardless of
    # the interval used after them.
    checkpoint_milestones = sorted(set(
        int(e) for e in pretrain_cfg.get("checkpoint_milestones", [5, 10, 20, 50])
    ))
    val_interval = pretrain_cfg.get("val_interval", 50)

    optim = AdamW(
        _trainable_parameter_groups(model),
        lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )
    warmup = LinearLR(optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optim, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr)
    scheduler = SequentialLR(optim, [warmup, cosine], milestones=[warmup_epochs])

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "adapt_mode": model.adapt_mode,
        "mask_ratio": model.mask_ratio,
        "param_counts": counts,
        "resolved_device": resolved_device,
        "cuda_device_name": cuda_device_name,
    }

    start_time = time.monotonic()
    interrupted = False
    last_epoch_completed = 0
    try:
        for epoch in range(epochs):
            model.train()
            running = 0.0
            nb = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                hsi = batch["hsi"].to(device, non_blocking=True)
                out = model(hsi)
                loss = out["loss"]
                optim.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(_trainable_parameter_groups(model), grad_clip)
                optim.step()
                running += float(loss.detach())
                nb += 1
                pbar.set_postfix(loss=f"{running/max(1,nb):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            scheduler.step()
            train_loss = running / max(1, nb)
            history["train_loss"].append(train_loss)
            last_epoch_completed = epoch + 1

            if (epoch + 1) % val_interval == 0 or epoch == 0:
                model.eval()
                val_loss = 0.0
                vb = 0
                with torch.no_grad():
                    for batch in val_loader:
                        hsi = batch["hsi"].to(device, non_blocking=True)
                        val_loss += float(model(hsi)["loss"])
                        vb += 1
                val_loss = val_loss / max(1, vb)
                history["val_loss"].append({"epoch": epoch + 1, "loss": val_loss})
                logger.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch + 1, train_loss, val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "best_val_loss": best_val,
                        },
                        checkpoint_dir / "checkpoint.pth",
                    )

            epoch_num = epoch + 1
            is_milestone = epoch_num in checkpoint_milestones
            is_interval = save_interval and (epoch_num % save_interval == 0)
            if is_milestone or is_interval:
                tag = "milestone" if is_milestone else "interval"
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch_num}.pth"
                torch.save(
                    {
                        "epoch": epoch_num,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                    },
                    ckpt_path,
                )
                logger.info("Saved %s checkpoint: %s", tag, ckpt_path.name)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "[HyperSIGMA-MAE] KeyboardInterrupt at epoch %d -- saving "
            "checkpoint_interrupted.pth and returning partial history.",
            last_epoch_completed,
        )
        torch.save(
            {
                "epoch": last_epoch_completed,
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_val_loss": best_val,
                "interrupted": True,
            },
            checkpoint_dir / "checkpoint_interrupted.pth",
        )

    if interrupted:
        # Skip the final-state save; the interrupted checkpoint is the
        # most-current snapshot. checkpoint.pth (val-best) is still on disk
        # if any val pass beat the initial loss before the interrupt.
        elapsed = time.monotonic() - start_time
        history["wallclock_seconds"] = round(elapsed, 2)
        history["best_val_loss"] = best_val
        history["epochs_run"] = last_epoch_completed
        history["epochs_requested"] = epochs
        history["interrupted"] = True
        history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None
        return history

    # Always save the final state so eval can rely on a deterministic
    # path even if no val pass beat the initial loss.
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_val_loss": best_val,
        },
        checkpoint_dir / "checkpoint_final.pth",
    )
    # Make checkpoint.pth point at the final state if no val-best was ever saved.
    if not (checkpoint_dir / "checkpoint.pth").exists():
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_val_loss": best_val,
            },
            checkpoint_dir / "checkpoint.pth",
        )

    elapsed = time.monotonic() - start_time
    history["wallclock_seconds"] = round(elapsed, 2)
    history["best_val_loss"] = best_val
    history["epochs_run"] = epochs
    history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None
    return history


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)
    paths_cfg = config.get("paths", {})
    # Default checkpoint/log dirs follow the registry convention for the
    # selected dataset, so they don't silently land in the Houston path.
    dataset_name = config.get("data", {}).get("dataset", "houston")
    spat_patch_k = config.get("model", {}).get("spat_patch_k", 3)
    default_ckpt_dir = get_spec(dataset_name).adapt_ckpt_dir(spat_patch_k)
    checkpoint_dir = args.checkpoint_dir or paths_cfg.get("checkpoint_dir", default_ckpt_dir)
    log_dir = args.log_dir or paths_cfg.get(
        "log_dir", default_ckpt_dir.replace("checkpoints/", "logs/", 1)
    )
    return run_adapt(config, checkpoint_dir, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)

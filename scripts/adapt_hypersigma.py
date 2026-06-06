#!/usr/bin/env python
"""Level-2 MAE adaptation of HyperSIGMA on a target dataset.

Trains the small randomly-initialized components of the dual encoder
(SpatViT ``patch_embed`` + ``pos_embed``, SpecViT ``spat_map`` +
``pos_embed`` + ``l1``, plus the SEM and the ``UnifiedBandMasking``
fill values) using masked HSI reconstruction. SpatViT/SpecViT
transformer bodies remain frozen.

Recipe matches ``configs/pretrain/houston_pretrain_enhanced.yaml``:
band_mask_ratio=0.9, sigma=1.0, decoder_hidden_dim=256, lr=1.5e-4,
weight_decay=0.05, warmup=100, epochs=3000, batch_size=64, AMP off.

Usage:
    python scripts/adapt_hypersigma_houston.py \
        --config configs/pretrain/hypersigma_houston_adapt.yaml
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

from data.datasets import (  # noqa: E402
    HoustonPatchedDataset,
    TrentoPatchedDataset,
    MUUFLPatchedDataset,
)
from models.hypersigma.hypersigma_dual import HyperSIGMADual  # noqa: E402
from pretrain.hypersigma_mae import HyperSIGMAMaskedAdaptation  # noqa: E402
from utils.io import load_config  # noqa: E402
from utils.seed import set_seed  # noqa: E402

DATASETS = {
    "houston": HoustonPatchedDataset,
    "trento": TrentoPatchedDataset,
    "muufl": MUUFLPatchedDataset,
}

logger = logging.getLogger(__name__)


def _build_dataset(config: dict):
    data_cfg = config.get("data", {})
    name = data_cfg.get("dataset", "houston")
    Dataset = DATASETS[name]
    ds = Dataset(
        data_root=data_cfg.get("data_root", "./data/raw"),
        patch_size=data_cfg.get("patch_size", 11),
        split="all",
        normalize=True,
    )
    logger.info("Loaded %s (split=all): %d samples", name, len(ds))
    return ds, name


def _build_model(config: dict, dataset_name: str) -> HyperSIGMAMaskedAdaptation:
    model_cfg = config.get("model", {})
    paths = config.get("paths", {})
    pretrain_cfg = config.get("pretrain", {})

    dual = HyperSIGMADual(
        pca_spat_path=paths.get(
            "pca_spat_path",
            f"checkpoints/hypersigma/pca_{dataset_name}_3band.pkl",
        ),
        spat_ckpt=paths.get("spat_ckpt", "checkpoints/hypersigma/spat-vit-base.pth"),
        spec_ckpt=paths.get("spec_ckpt", "checkpoints/hypersigma/spec-vit-base.pth"),
        hsi_channels=model_cfg.get("hsi_channels", 144),
        spat_patch_k=model_cfg.get("spat_patch_k", 3),
        freeze_body=True,
        embed_dim=model_cfg.get("embed_dim", 768),
        num_tokens=model_cfg.get("num_tokens", 100),
        dr_dim=model_cfg.get("dr_dim", 128),
        num_stages=model_cfg.get("num_stages", 4),
    )
    dual.log_sanity()

    return HyperSIGMAMaskedAdaptation(
        dual=dual,
        hsi_channels=model_cfg.get("hsi_channels", 144),
        patch_size=config.get("data", {}).get("patch_size", 11),
        decoder_hidden_dim=pretrain_cfg.get("decoder_hidden_dim", 256),
        band_mask_ratio=pretrain_cfg.get("band_mask_ratio", 0.9),
        recon_sigma=pretrain_cfg.get("recon_center_sigma", 1.0),
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
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available; falling back to CPU")
    logger.info("Using device: %s", device)

    dataset, dataset_name = _build_dataset(config)

    model = _build_model(config, dataset_name).to(device)

    counts = {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    logger.info(
        "[HyperSIGMA-MAE] Adaptation module params: trainable=%d, frozen=%d",
        counts["trainable"], counts["frozen"],
    )

    train_loader, val_loader = _build_loaders(dataset, config, seed)

    pretrain_cfg = config.get("pretrain", {})
    epochs = pretrain_cfg.get("epochs", 3000)
    lr = pretrain_cfg.get("lr", 1.5e-4)
    min_lr = pretrain_cfg.get("min_lr", 1e-6)
    warmup_epochs = pretrain_cfg.get("warmup_epochs", 100)
    weight_decay = pretrain_cfg.get("weight_decay", 0.05)
    grad_clip = pretrain_cfg.get("grad_clip", 1.0)
    save_interval = pretrain_cfg.get("save_interval", 500)
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
    history = {"train_loss": [], "val_loss": []}

    start_time = time.monotonic()
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

        if save_interval and (epoch + 1) % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth",
            )

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
    checkpoint_dir = args.checkpoint_dir or paths_cfg.get(
        "checkpoint_dir", "checkpoints/hypersigma_adapted/houston_k3",
    )
    log_dir = args.log_dir or paths_cfg.get("log_dir", "logs/hypersigma_adapted")
    return run_adapt(config, checkpoint_dir, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)

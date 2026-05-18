#!/usr/bin/env python
"""
Unified masked pretraining script for MFT-CPEA.

HSI + LiDAR are concatenated as a single per-pixel band vector. Two masking
regimes are supported and can be combined (configured via the pretrain YAML):
  - band_mask_ratio: per-(pixel, band) Bernoulli mask on the raw input.
  - spatial_mask_ratio: MAE-style whole-pixel-token masking.

Usage:
    python scripts/pretrain_enhanced.py --config configs/pretrain/houston_pretrain_enhanced.yaml

    # Resume training
    python scripts/pretrain_enhanced.py --config configs/pretrain/houston_pretrain_enhanced.yaml --resume checkpoint_epoch_400.pth

    # Combined datasets
    python scripts/pretrain_enhanced.py --config configs/pretrain/houston_trento_pretrain_enhanced.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from utils.io import load_config
from utils.logging import setup_logging
from data.datasets import (
    HoustonPatchedDataset,
    TrentoPatchedDataset,
    MUUFLPatchedDataset,
)
from models import MFTCPEACosine
from pretrain.masked_modeling_enhanced import EnhancedMaskedSpectralSpatialModel
from trainers.pretrain_trainer import PretrainTrainer


# Map dataset names to classes (using patched format)
DATASETS = {
    "houston": HoustonPatchedDataset,
    "trento": TrentoPatchedDataset,
    "muufl": MUUFLPatchedDataset,
}


class CombinedPatchedDataset(torch.utils.data.ConcatDataset):
    """
    Combines multiple patched datasets.
    Handles different channel counts by padding.
    """

    def __init__(self, datasets):
        super().__init__(datasets)

        # Determine max channels for padding
        self.max_hsi_channels = max(ds.hsi_channels for ds in datasets)
        self.max_aux_channels = max(ds.aux_channels for ds in datasets)

        # Store original datasets for reference
        self.source_datasets = datasets

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # Pad HSI if needed
        hsi = sample["hsi"]
        if hsi.shape[0] < self.max_hsi_channels:
            pad_size = self.max_hsi_channels - hsi.shape[0]
            padding = torch.zeros(pad_size, hsi.shape[1], hsi.shape[2])
            hsi = torch.cat([hsi, padding], dim=0)
            sample["hsi"] = hsi

        # Pad aux if needed
        aux = sample["aux"]
        if aux.shape[0] < self.max_aux_channels:
            pad_size = self.max_aux_channels - aux.shape[0]
            padding = torch.zeros(pad_size, aux.shape[1], aux.shape[2])
            aux = torch.cat([aux, padding], dim=0)
            sample["aux"] = aux

        return sample


def _log_gpu_diagnostics(logger: logging.Logger) -> None:
    """Print a startup block describing the GPU environment.

    Always called regardless of which device is requested so misconfigured
    environments leave a paper trail in pretrain.log.
    """
    logger.info(f"torch={torch.__version__} cuda_build={torch.version.cuda} "
                f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        logger.info(f"Visible GPUs ({count}):")
        for i in range(count):
            logger.info(f"  [{i}] {torch.cuda.get_device_name(i)}")
    else:
        logger.info("No CUDA devices visible to PyTorch.")


def _resolve_device(requested: str, logger: logging.Logger) -> str:
    """Resolve `hardware.device` per the strict contract.

    - "cuda"        → require CUDA; raise if unavailable.
    - "cuda:N"      → require that specific index; raise if out of range.
    - "auto"        → cuda:0 if available, else cpu.
    - "cpu"         → cpu (forced).
    """
    if requested == "auto":
        chosen = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"device=auto → selected {chosen}")
    elif requested == "cpu":
        chosen = "cpu"
        logger.info("device=cpu (explicit)")
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"hardware.device={requested!r} but CUDA is not available "
                f"(torch={torch.__version__}, cuda_build={torch.version.cuda}, "
                f"torch.cuda.is_available()={torch.cuda.is_available()}). "
                "Fix the environment (install a CUDA-enabled torch build and "
                "ensure a GPU is visible) or set hardware.device to 'auto' or 'cpu'."
            )
        if requested == "cuda":
            chosen = "cuda:0"
        else:
            idx = int(requested.split(":", 1)[1])
            count = torch.cuda.device_count()
            if idx < 0 or idx >= count:
                raise RuntimeError(
                    f"hardware.device={requested!r} but only {count} CUDA "
                    f"device(s) visible (valid indices: 0..{count - 1})."
                )
            chosen = requested
        logger.info(f"device={requested} → selected {chosen} "
                    f"({torch.cuda.get_device_name(int(chosen.split(':', 1)[1]))})")
    else:
        raise RuntimeError(
            f"Unrecognised hardware.device={requested!r}. "
            "Expected 'cuda', 'cuda:N', 'auto', or 'cpu'."
        )

    if chosen.startswith("cuda"):
        torch.cuda.set_device(chosen)
    return chosen


def run_pretrain(
    config: dict,
    checkpoint_dir: str,
    log_dir: str,
    resume: str = None,
) -> dict:
    """Programmatic entry point used by notebooks and CLI alike.

    The `config` dict mirrors the YAML config layout. `checkpoint_dir` and
    `log_dir` are taken as-is (the caller is responsible for placing them
    inside an experiment directory if desired).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {config}")

    # Set seed
    from utils.seed import set_seed
    seed = config.get("hardware", {}).get("seed", 42)
    deterministic = config.get("hardware", {}).get("deterministic", False)
    set_seed(seed, deterministic=deterministic)

    # Device — strict contract; fail loudly when CUDA is requested but missing.
    _log_gpu_diagnostics(logger)
    requested_device = config.get("hardware", {}).get("device", "cuda")
    device = _resolve_device(requested_device, logger)

    # Load datasets for pretraining
    pretrain_config = config.get("pretrain", {})
    data_config = config.get("data", {})

    datasets_to_use = pretrain_config.get("datasets", ["houston"])
    logger.info(f"Loading datasets: {datasets_to_use}")

    # Load datasets (using "all" split to combine train+test for pretraining)
    all_datasets = []
    for ds_name in datasets_to_use:
        if ds_name not in DATASETS:
            logger.warning(f"Unknown dataset: {ds_name}, skipping")
            continue

        DatasetClass = DATASETS[ds_name]
        ds = DatasetClass(
            data_root=config["paths"]["data_root"],
            patch_size=data_config.get("patch_size", 11),
            split="all",  # Use all data for pretraining
            normalize=True
        )
        all_datasets.append(ds)
        logger.info(f"Loaded {ds_name}: HSI {ds.hsi.shape}, Aux {ds.aux.shape}, {len(ds)} samples")

    if not all_datasets:
        raise ValueError("No valid datasets loaded!")

    # Combine datasets if multiple
    if len(all_datasets) == 1:
        full_dataset = all_datasets[0]
        hsi_channels = full_dataset.hsi_channels
        aux_channels = full_dataset.aux_channels
    else:
        # Combine multiple datasets
        full_dataset = CombinedPatchedDataset(all_datasets)
        hsi_channels = max(ds.hsi_channels for ds in all_datasets)
        aux_channels = max(ds.aux_channels for ds in all_datasets)

    logger.info(f"Total samples: {len(full_dataset)}")
    logger.info(f"HSI channels: {hsi_channels}, Aux channels: {aux_channels}")

    # Split into train/val (stratified by label to preserve class distribution)
    val_split = pretrain_config.get("val_split", 0.1)

    # Extract labels for stratification
    if hasattr(full_dataset, 'labels'):
        all_labels = full_dataset.labels.numpy()
    else:
        # CombinedPatchedDataset or similar: extract labels from sub-datasets
        all_labels = []
        for ds in full_dataset.datasets:
            all_labels.append(ds.labels.numpy())
        all_labels = np.concatenate(all_labels)

    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(all_labels)), all_labels))
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    # Create dataloaders. pin_memory only matters on CUDA; turning it on for
    # CPU runs wastes memory and prints a torch warning.
    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", device.startswith("cuda"))
    drop_last = data_config.get("drop_last", True)
    persistent_workers = data_config.get("persistent_workers", False) and num_workers > 0
    prefetch_factor = data_config.get("prefetch_factor")
    loader_extra = {}
    if num_workers > 0:
        loader_extra["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_extra["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=pretrain_config.get("batch_size", 64),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **loader_extra,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=pretrain_config.get("batch_size", 64),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_extra,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create encoder (MFT-CPEA-Cosine model)
    model_config = config.get("model", {})
    embed_dim = model_config.get("embed_dim", 128)

    encoder = MFTCPEACosine(
        hsi_channels=hsi_channels,
        aux_channels=aux_channels,
        embed_dim=embed_dim,
        num_heads=model_config.get("num_heads", 8),
        num_layers=model_config.get("num_layers", 4),
        patch_size=data_config.get("patch_size", 11),
        lambda_factor=model_config.get("lambda_factor", 2.0),
        dropout=model_config.get("dropout", 0.1),
        # Projection head settings (trained for few-shot transfer)
        use_projection=model_config.get("use_projection", True),
        proj_hidden_dim=model_config.get("proj_hidden_dim", embed_dim * 4),
        proj_num_layers=model_config.get("proj_num_layers", 2),
        proj_l2_normalize=model_config.get("proj_l2_normalize", False),
    )

    logger.info(f"Projection head: hidden_dim={model_config.get('proj_hidden_dim', embed_dim * 4)}, "
                f"num_layers={model_config.get('proj_num_layers', 2)}, "
                f"l2_normalize={model_config.get('proj_l2_normalize', False)}")

    # Create unified-mask pretraining model
    band_mask_ratio = pretrain_config.get("band_mask_ratio", 0.9)
    spatial_mask_ratio = pretrain_config.get("spatial_mask_ratio", 0.0)

    pretrain_model = EnhancedMaskedSpectralSpatialModel(
        encoder=encoder,
        hsi_channels=hsi_channels,
        aux_channels=aux_channels,
        patch_size=data_config.get("patch_size", 11),
        embed_dim=embed_dim,
        decoder_hidden_dim=pretrain_config.get("decoder_hidden_dim", 256),
        band_mask_ratio=band_mask_ratio,
        spatial_mask_ratio=spatial_mask_ratio,
        recon_sigma=pretrain_config.get("recon_center_sigma", None),
    )

    total_params = sum(p.numel() for p in pretrain_model.parameters())
    encoder_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Encoder parameters: {encoder_params:,}")
    logger.info(f"Band mask ratio: {band_mask_ratio}")
    logger.info(f"Spatial mask ratio: {spatial_mask_ratio}")

    # Create trainer. Cast numeric fields defensively — PyYAML's YAML-1.1
    # resolver parses `1e-6` (no dot before the exponent) as a *string*,
    # not a float, which detonates downstream in the LR scheduler.
    trainer_config = {
        "epochs": int(pretrain_config.get("epochs", 800)),
        "lr": float(pretrain_config.get("lr", 1.5e-4)),
        "min_lr": float(pretrain_config.get("min_lr", 1e-6)),
        "weight_decay": float(pretrain_config.get("weight_decay", 0.05)),
        "warmup_epochs": int(pretrain_config.get("warmup_epochs", 40)),
        "warmup_start_factor": float(pretrain_config.get("warmup_start_factor", 0.01)),
        "adam_betas": tuple(float(b) for b in pretrain_config.get("adam_betas", (0.9, 0.95))),
        "grad_clip": float(pretrain_config.get("grad_clip", 1.0)),
        "save_interval": int(pretrain_config.get("save_interval", 100)),
        "val_interval": int(pretrain_config.get("val_interval", 50)),
        "log_interval": int(pretrain_config.get("log_interval", 10)),
        "use_amp": bool(pretrain_config.get("use_amp", False)),
    }

    trainer = PretrainTrainer(
        model=pretrain_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    # Train
    history = trainer.train(resume_from=resume)

    # Stamp environment info on the history so the runner can record it in
    # pretrain_metadata.json (useful when sanity-checking that a run actually
    # used the GPU it asked for).
    history["resolved_device"] = device
    if device.startswith("cuda"):
        history["cuda_device_name"] = torch.cuda.get_device_name(int(device.split(":", 1)[1]))

    logger.info("Enhanced pretraining complete!")
    logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")
    logger.info(f"Final training loss: {history['final_train_loss']:.4f}")

    return history


def main(args):
    setup_logging(args.log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    checkpoint_dir = config["paths"].get("checkpoint_dir", "./checkpoints/pretrained_enhanced")
    log_dir = config["paths"].get("log_dir", "./logs/pretrain_enhanced")

    return run_pretrain(config, checkpoint_dir, log_dir, resume=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Pretrain MFT-CPEA encoder")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path"
    )

    args = parser.parse_args()
    main(args)

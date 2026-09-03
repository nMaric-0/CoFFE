"""
Per-scene masked pretraining for CoFFE and the MFT architectural control.

HSI + LiDAR are concatenated as a single per-pixel band vector. Two composable
SimMIM-style masks are supported (configured via the pretrain YAML):
  - band_mask_ratio: per-(pixel, band) Bernoulli mask on the raw input.
  - spatial_mask_ratio: whole-pixel-token (spatial) masking, in place.
The MAE objective (token drop + transformer decoder) is selected with
``pretrain.objective: "mae"``.

Usage:
    python scripts/pretrain.py --config configs/coffe/houston_simmim.yaml

    # Resume training
    python scripts/pretrain.py --config configs/coffe/houston_simmim.yaml \\
        --resume checkpoint_epoch_400.pth
"""

import argparse
import logging
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from coffe.compat import normalize_model_name, normalize_objective
from coffe.data.datasets import (
    HoustonPatchedDataset,
    MUUFLPatchedDataset,
    TrentoPatchedDataset,
)
from coffe.models import CoFFE, MFTOriginal
from coffe.pretrain.mae_pretrain import MAEPretrainModel
from coffe.pretrain.mft_mae import MFTMAEPretrainModel
from coffe.pretrain.mft_spatial_mae import MFTSpatialMaskPretrainModel
from coffe.pretrain.simmim import SimMIMPretrainModel
from coffe.pretrain.trainer import PretrainTrainer
from coffe.utils.io import load_config
from coffe.utils.logging import setup_logging

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

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        super().__init__(datasets)

        # Determine max channels for padding
        self.max_hsi_channels = max(ds.hsi_channels for ds in datasets)
        self.max_aux_channels = max(ds.aux_channels for ds in datasets)

        # Store original datasets for reference
        self.source_datasets = datasets

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
    logger.info(
        f"torch={torch.__version__} cuda_build={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()}"
    )
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
        logger.info(
            f"device={requested} → selected {chosen} "
            f"({torch.cuda.get_device_name(int(chosen.split(':', 1)[1]))})"
        )
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
    resume: str | None = None,
) -> dict:
    """Programmatic entry point used by notebooks and CLI alike.

    The `config` dict mirrors the YAML config layout. `checkpoint_dir` and
    `log_dir` are taken as-is (the caller is responsible for placing them
    inside an experiment directory if desired).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {config}")

    # Set seed
    from coffe.utils.seed import set_seed

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
            normalize=True,
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

    use_aux = config.get("model", {}).get("use_aux", True)
    logger.info(f"Total samples: {len(full_dataset)}")
    logger.info(
        f"HSI channels: {hsi_channels}, Aux channels: {aux_channels}, "
        f"use_aux={use_aux} ({'HSI+aux' if use_aux else 'HSI-only'})"
    )

    # Split into train/val (stratified by label to preserve class distribution).
    # val_split <= 0 disables validation: the full dataset becomes the train set
    # and no val loader is built. The trainer handles val_loader=None.
    val_split = pretrain_config.get("val_split", 0.1)

    if val_split <= 0:
        train_dataset = full_dataset
        val_dataset = None
    else:
        # Extract labels for stratification
        if hasattr(full_dataset, "labels"):
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

    if val_dataset is None:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=pretrain_config.get("batch_size", 64),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **loader_extra,
        )

    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset is None:
        logger.info("Validation samples: 0 (val_split <= 0, validation disabled)")
    else:
        logger.info(f"Validation samples: {len(val_dataset)}")

    # Create encoder. Default is CoFFE; model.name "mft_original" selects the
    # faithful original-MFT control instead. Frozen configs say "mft_cpea";
    # normalize_model_name maps that to "coffe" (PAPER_CANON §7.3).
    model_config = config.get("model", {})
    embed_dim = model_config.get("embed_dim", 128)
    model_name = normalize_model_name(model_config.get("name", "coffe"), origin="model.name")

    # Pretraining objective: "simmim" (default; in-place band/spatial masking +
    # MLP decoder) or "mae" (token-drop recipe: remove 75% of tokens, encode
    # visible only, transformer decoder, no projection head). Frozen configs
    # say "enhanced" for SimMIM.
    objective = normalize_objective(
        pretrain_config.get("objective", "simmim"), origin="pretrain.objective"
    )
    use_projection = model_config.get("use_projection", True)
    if objective == "mae" and use_projection:
        # The MAE recipe has no projection head; the transformer decoder replaces
        # it. Force it off regardless of config so the encoder checkpoint matches
        # what eval rebuilds (eval defaults use_projection=False).
        logger.info("objective=mae: forcing use_projection=False (no projection head)")
        use_projection = False

    if model_name == "mft_original":
        # Faithful original-MFT control. It has no projection head (eval rebuilds
        # the bare encoder). Two objectives are supported:
        #   "mae"    -> token-drop MAE (coffe/pretrain/mft_mae.py)
        #   "simmim" -> spatial-token masking + MLP decoder (coffe/pretrain/mft_spatial_mae.py)
        if objective not in ("mae", "simmim"):
            raise ValueError(
                f"model.name='mft_original' supports objective in {{'mae','simmim'}}, "
                f"got objective={objective!r}"
            )
        encoder = MFTOriginal(
            hsi_channels=hsi_channels,
            aux_channels=aux_channels,
            use_aux=use_aux,
            embed_dim=embed_dim,
            num_heads=model_config.get("num_heads", 8),
            num_layers=model_config.get("num_layers", 2),
            mlp_dim=model_config.get("mlp_dim", 512),
            patch_size=data_config.get("patch_size", 11),
            dropout=model_config.get("dropout", 0.1),
            attention_type=model_config.get("attention_type", "mcross"),
        )
    else:
        encoder = CoFFE(
            hsi_channels=hsi_channels,
            aux_channels=aux_channels,
            use_aux=use_aux,
            embed_dim=embed_dim,
            num_heads=model_config.get("num_heads", 8),
            num_layers=model_config.get("num_layers", 4),
            patch_size=data_config.get("patch_size", 11),
            # Config key stays `lambda_factor` (frozen vocabulary, §7.3);
            # unused during pretraining, kept so eval rebuilds the same model.
            cls_token_weight=model_config.get("lambda_factor", 2.0),
            dropout=model_config.get("dropout", 0.1),
            # Projection head settings (trained for few-shot transfer)
            use_projection=use_projection,
            proj_hidden_dim=model_config.get("proj_hidden_dim", embed_dim * 4),
            proj_num_layers=model_config.get("proj_num_layers", 2),
            proj_l2_normalize=model_config.get("proj_l2_normalize", False),
        )

    if model_name == "mft_original" and objective == "mae":
        pretrain_model = MFTMAEPretrainModel(
            encoder=encoder,
            hsi_channels=hsi_channels,
            aux_channels=aux_channels,
            use_aux=use_aux,
            patch_size=data_config.get("patch_size", 11),
            embed_dim=embed_dim,
            mask_ratio=pretrain_config.get("mask_ratio", 0.75),
            decoder_dim=pretrain_config.get("decoder_dim", 64),
            decoder_depth=pretrain_config.get("decoder_depth", 4),
            decoder_heads=pretrain_config.get("decoder_heads", 4),
            decoder_mlp_ratio=pretrain_config.get("decoder_mlp_ratio", 4.0),
            norm_pix_loss=pretrain_config.get("norm_pix_loss", True),
            recon_sigma=pretrain_config.get("recon_center_sigma", None),
        )

        total_params = sum(p.numel() for p in pretrain_model.parameters())
        encoder_params = sum(p.numel() for p in encoder.parameters())
        logger.info("Model: original-MFT baseline (channel tokenization, mCrossPA)")
        logger.info("Objective: standard MAE (transformer decoder, cross-attn to encoded CLS)")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Encoder parameters: {encoder_params:,}")
        logger.info(
            f"Mask ratio: {pretrain_model.mask_ratio} "
            f"({pretrain_model.len_keep}/{pretrain_model.num_tokens} tokens visible)"
        )
    elif model_name == "mft_original" and objective == "simmim":
        spatial_mask_ratio = pretrain_config.get("spatial_mask_ratio", 0.75)
        pretrain_model = MFTSpatialMaskPretrainModel(
            encoder=encoder,
            hsi_channels=hsi_channels,
            aux_channels=aux_channels,
            use_aux=use_aux,
            patch_size=data_config.get("patch_size", 11),
            embed_dim=embed_dim,
            decoder_hidden_dim=pretrain_config.get("decoder_hidden_dim", 256),
            spatial_mask_ratio=spatial_mask_ratio,
            recon_sigma=pretrain_config.get("recon_center_sigma", None),
        )

        total_params = sum(p.numel() for p in pretrain_model.parameters())
        encoder_params = sum(p.numel() for p in encoder.parameters())
        logger.info("Model: original-MFT baseline (channel tokenization, mCrossPA)")
        logger.info(
            "Objective: SimMIM token (in-place spatial-token masking + MLP "
            "decoder, CLS-injection, center-weighted MSE)"
        )
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Encoder parameters: {encoder_params:,}")
        logger.info(f"Spatial mask ratio: {spatial_mask_ratio}")
        if pretrain_config.get("band_mask_ratio", 0.0):
            logger.warning(
                "band_mask_ratio is ignored for model.name='mft_original' "
                "(only spatial masking is supported)."
            )
    elif objective == "mae":
        mask_ratio = pretrain_config.get("mask_ratio", 0.75)
        decoder_dim = pretrain_config.get("decoder_dim", 64)
        decoder_depth = pretrain_config.get("decoder_depth", 4)
        decoder_heads = pretrain_config.get("decoder_heads", 4)
        norm_pix_loss = pretrain_config.get("norm_pix_loss", True)

        pretrain_model = MAEPretrainModel(
            encoder=encoder,
            hsi_channels=hsi_channels,
            aux_channels=aux_channels,
            use_aux=use_aux,
            patch_size=data_config.get("patch_size", 11),
            embed_dim=embed_dim,
            mask_ratio=mask_ratio,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            decoder_mlp_ratio=pretrain_config.get("decoder_mlp_ratio", 4.0),
            norm_pix_loss=norm_pix_loss,
            recon_sigma=pretrain_config.get("recon_center_sigma", None),
        )

        total_params = sum(p.numel() for p in pretrain_model.parameters())
        encoder_params = sum(p.numel() for p in encoder.parameters())
        logger.info("Objective: MAE (transformer decoder, no projection head)")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Encoder parameters: {encoder_params:,}")
        logger.info(
            f"Mask ratio: {mask_ratio} "
            f"({pretrain_model.len_keep}/{pretrain_model.num_tokens} tokens visible)"
        )
        logger.info(
            f"Decoder: dim={decoder_dim}, depth={decoder_depth}, "
            f"heads={decoder_heads}, norm_pix_loss={norm_pix_loss}"
        )
    else:
        logger.info(
            f"Projection head: hidden_dim={model_config.get('proj_hidden_dim', embed_dim * 4)}, "
            f"num_layers={model_config.get('proj_num_layers', 2)}, "
            f"l2_normalize={model_config.get('proj_l2_normalize', False)}"
        )

        # Create unified-mask pretraining model
        band_mask_ratio = pretrain_config.get("band_mask_ratio", 0.9)
        spatial_mask_ratio = pretrain_config.get("spatial_mask_ratio", 0.0)

        pretrain_model = SimMIMPretrainModel(
            encoder=encoder,
            hsi_channels=hsi_channels,
            aux_channels=aux_channels,
            use_aux=use_aux,
            patch_size=data_config.get("patch_size", 11),
            embed_dim=embed_dim,
            decoder_hidden_dim=pretrain_config.get("decoder_hidden_dim", 256),
            band_mask_ratio=band_mask_ratio,
            spatial_mask_ratio=spatial_mask_ratio,
            recon_sigma=pretrain_config.get("recon_center_sigma", None),
            recon_loss=pretrain_config.get("recon_loss", "mse"),
        )

        total_params = sum(p.numel() for p in pretrain_model.parameters())
        encoder_params = sum(p.numel() for p in encoder.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Encoder parameters: {encoder_params:,}")
        logger.info(f"Band mask ratio: {band_mask_ratio}")
        logger.info(f"Spatial mask ratio: {spatial_mask_ratio}")

    # Create trainer
    trainer_config = {
        "epochs": pretrain_config.get("epochs", 800),
        "lr": pretrain_config.get("lr", 1.5e-4),
        "min_lr": pretrain_config.get("min_lr", 1e-6),
        "weight_decay": pretrain_config.get("weight_decay", 0.05),
        "warmup_epochs": pretrain_config.get("warmup_epochs", 40),
        "warmup_start_factor": pretrain_config.get("warmup_start_factor", 0.01),
        "adam_betas": tuple(pretrain_config.get("adam_betas", (0.9, 0.95))),
        "grad_clip": pretrain_config.get("grad_clip", 1.0),
        "save_interval": pretrain_config.get("save_interval", 100),
        "val_interval": pretrain_config.get("val_interval", 50),
        "log_interval": pretrain_config.get("log_interval", 10),
        "use_amp": pretrain_config.get("use_amp", False),
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

    logger.info("Pretraining complete!")
    logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")
    logger.info(f"Final training loss: {history['final_train_loss']:.4f}")

    return history


def main(args: argparse.Namespace) -> dict:
    """CLI entry point: load the YAML config and run per-scene pretraining."""
    setup_logging(args.log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    checkpoint_dir = config["paths"].get("checkpoint_dir", "./checkpoints/pretrained")
    log_dir = config["paths"].get("log_dir", "./logs/pretrain")

    return run_pretrain(config, checkpoint_dir, log_dir, resume=args.resume)

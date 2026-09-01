"""
Trainer for self-supervised pretraining.

Implements training loop with:
- AdamW optimizer with weight decay
- Cosine annealing learning rate with warmup
- Gradient clipping
- Checkpointing and logging
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from tqdm import tqdm
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Loss component keys returned by pretraining models
_LOSS_COMPONENTS = ("band_recon",)
# Abbreviated names for progress bar display
_LOSS_ABBREV = {"band_recon": "band"}


class PretrainTrainer:
    """
    Trainer for self-supervised pretraining of the CoFFE encoder.
    
    Features:
    - AdamW optimizer with configurable weight decay
    - Learning rate warmup + cosine decay
    - Gradient clipping
    - Periodic validation and checkpointing
    - TensorBoard / WandB logging support
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints/pretrained",
        log_dir: str = "./logs/pretrain"
    ):
        """
        Args:
            model: MaskedSpectralSpatialModel instance
            train_loader: Training data loader
            val_loader: Optional validation data loader
            config: Training configuration dictionary
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Default config
        self.config = {
            "epochs": 800,
            "lr": 1.5e-4,
            "min_lr": 1e-6,
            "weight_decay": 0.05,
            "warmup_epochs": 40,
            "warmup_start_factor": 0.01,
            "adam_betas": (0.9, 0.95),
            "grad_clip": 1.0,
            "save_interval": 100,
            "val_interval": 50,
            "log_interval": 10,
            "use_amp": False,  # Automatic mixed precision
        }
        if config:
            self.config.update(config)
        self.config["adam_betas"] = tuple(self.config["adam_betas"])

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            betas=self.config["adam_betas"],
        )

        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=self.config["warmup_start_factor"],
            end_factor=1.0,
            total_iters=self.config["warmup_epochs"]
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"] - self.config["warmup_epochs"],
            eta_min=self.config["min_lr"]
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config["warmup_epochs"]]
        )
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.config["use_amp"] else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        
        # TensorBoard writer (optional)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with 'total' average loss and per-component averages
        """
        self.model.train()
        total_loss = 0.0
        component_sums = {k: 0.0 for k in _LOSS_COMPONENTS}
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config['epochs']}",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            hsi = batch["hsi"].to(self.device)
            aux = batch["aux"].to(self.device)

            # Forward pass with optional mixed precision
            if self.config["use_amp"]:
                with torch.amp.autocast('cuda'):
                    loss, info = self.model(hsi, aux)
            else:
                loss, info = self.model(hsi, aux)

            # Backward pass
            self.optimizer.zero_grad()

            if self.config["use_amp"]:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config["grad_clip"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["grad_clip"]
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["grad_clip"]
                    )

                self.optimizer.step()

            # Update stats
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            self.global_step += 1

            # Accumulate component losses
            batch_components = {}
            for key in _LOSS_COMPONENTS:
                if key in info and torch.is_tensor(info[key]):
                    val = info[key].item()
                    component_sums[key] += val
                    batch_components[key] = val

            # Update progress bar with component losses
            postfix = {"loss": f"{batch_loss:.4f}"}
            for key, val in batch_components.items():
                postfix[_LOSS_ABBREV[key]] = f"{val:.4f}"
            postfix["lr"] = f"{self.scheduler.get_last_lr()[0]:.2e}"
            pbar.set_postfix(postfix)

            # Logging
            if self.global_step % self.config["log_interval"] == 0:
                if self.writer:
                    self.writer.add_scalar("train/loss", batch_loss, self.global_step)
                    for key, val in batch_components.items():
                        self.writer.add_scalar(
                            f"train/loss_{key}", val, self.global_step
                        )
                    self.writer.add_scalar(
                        "train/lr",
                        self.scheduler.get_last_lr()[0],
                        self.global_step
                    )

        result = {"total": total_loss / num_batches}
        for key in _LOSS_COMPONENTS:
            if component_sums[key] > 0:
                result[key] = component_sums[key] / num_batches
        return result
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary with 'total' average loss and per-component averages
        """
        if self.val_loader is None:
            return {"total": float('inf')}

        self.model.eval()
        total_loss = 0.0
        component_sums = {k: 0.0 for k in _LOSS_COMPONENTS}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            hsi = batch["hsi"].to(self.device)
            aux = batch["aux"].to(self.device)

            if self.config["use_amp"]:
                with torch.amp.autocast('cuda'):
                    loss, info = self.model(hsi, aux)
            else:
                loss, info = self.model(hsi, aux)

            total_loss += loss.item()
            for key in _LOSS_COMPONENTS:
                if key in info and torch.is_tensor(info[key]):
                    component_sums[key] += info[key].item()
            num_batches += 1

        result = {"total": total_loss / num_batches}
        for key in _LOSS_COMPONENTS:
            if component_sums[key] > 0:
                result[key] = component_sums[key] / num_batches
        return result
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
    
    def save_encoder(self, filename: str = "encoder.pth"):
        """
        Save only the encoder weights (for downstream tasks).

        Args:
            filename: Encoder weights filename
        """
        path = self.checkpoint_dir / filename
        encoder = self.model.get_encoder()

        # Save encoder state with config for proper loading
        encoder_checkpoint = {
            "state_dict": encoder.state_dict(),
            "config": {
                "hsi_channels": self.model.hsi_channels,
                "aux_channels": self.model.aux_channels,
                "embed_dim": self.model.embed_dim,
                "patch_size": self.model.patch_size,
            }
        }
        torch.save(encoder_checkpoint, path)
        logger.info(f"Saved encoder weights to {path} (hsi={self.model.hsi_channels}, aux={self.model.aux_channels})")
    
    def train(
        self,
        resume_from: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            resume_from: Optional checkpoint to resume from
            callback: Optional callback function called after each epoch
                      with signature callback(trainer, epoch, train_loss, val_loss)
        
        Returns:
            Dictionary with training history
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Starting pretraining for {self.config['epochs']} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.config["epochs"]):
            self.current_epoch = epoch
            
            # Train
            train_result = self.train_epoch()
            train_loss = train_result["total"]
            self.train_losses.append(train_loss)

            # Update learning rate
            self.scheduler.step()

            # Validate
            val_loss = float('inf')
            val_components = {}
            if self.val_loader and (epoch + 1) % self.config["val_interval"] == 0:
                val_result = self.validate()
                val_loss = val_result["total"]
                val_components = {k: v for k, v in val_result.items() if k != "total"}
                self.val_losses.append(val_loss)

                if self.writer:
                    self.writer.add_scalar("val/loss", val_loss, epoch)
                    for key, val in val_components.items():
                        self.writer.add_scalar(f"val/loss_{key}", val, epoch)

                # Check if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pth", is_best=True)

            # Logging
            component_str = ""
            train_components = {k: v for k, v in train_result.items() if k != "total"}
            if train_components:
                parts = [f"{_LOSS_ABBREV[k]}={v:.4f}" for k, v in train_components.items()]
                component_str = f" ({', '.join(parts)})"

            logger.info(
                f"Epoch {epoch + 1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}{component_str} - "
                f"Val Loss: {val_loss:.4f} - "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )

            if self.writer:
                self.writer.add_scalar("train/epoch_loss", train_loss, epoch)
                for key, val in train_components.items():
                    self.writer.add_scalar(f"train/epoch_loss_{key}", val, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config["save_interval"] == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Callback
            if callback:
                callback(self, epoch, train_loss, val_loss)
        
        # Save final checkpoint and encoder
        self.save_checkpoint("final.pth")
        self.save_encoder("encoder_final.pth")
        
        if self.writer:
            self.writer.close()
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "epochs_run": self.current_epoch + 1 if self.train_losses else 0,
        }


def create_pretrain_dataloaders(
    datasets: list,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> tuple:
    """
    Create training and validation dataloaders for pretraining.
    
    Args:
        datasets: List of base datasets (Houston, Trento, etc.)
        batch_size: Batch size
        val_split: Fraction for validation
        num_workers: Number of data loading workers
        seed: Random seed for splitting
        
    Returns:
        train_loader, val_loader
    """
    from .masked_modeling import PretrainDataset, CombinedPretrainDataset
    
    # Wrap each dataset
    pretrain_datasets = [PretrainDataset(ds) for ds in datasets]
    
    # Combine if multiple datasets
    if len(pretrain_datasets) == 1:
        full_dataset = pretrain_datasets[0]
    else:
        # Determine max channels for padding
        max_hsi = max(ds.base_dataset.hsi_channels for ds in pretrain_datasets)
        max_aux = max(ds.base_dataset.aux_channels for ds in pretrain_datasets)
        
        full_dataset = CombinedPretrainDataset(
            pretrain_datasets,
            target_hsi_channels=max_hsi,
            target_aux_channels=max_aux
        )
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

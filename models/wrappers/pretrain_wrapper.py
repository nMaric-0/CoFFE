"""
Pretraining Wrapper for Backbone Models.

Provides two self-supervised pretraining approaches:
1. Masked Autoencoding (MAE-style): Mask tokens BEFORE encoder, reconstruct pre-encoder values
2. Contrastive Learning (SimCLR-style): Pull augmented views together

The backbone remains unchanged - only pretraining heads are added.
After pretraining, discard the heads and use backbone with CPEAWrapper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import random

from ..backbones.base import BackboneBase


class MAEDecoder(nn.Module):
    """
    Decoder for Masked Autoencoding.

    Takes encoded visible tokens and reconstructs the masked token embeddings.
    The target is the PRE-ENCODER token values (not post-encoder).
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.decoder_dim = decoder_dim

        # Project encoder output to decoder dim
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_dim)

        # Learnable mask tokens (one per masked position)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)

        # Decoder positional embedding (for all token positions)
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, decoder_dim) * 0.02
        )

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Output projection back to embed_dim (to match pre-encoder tokens)
        self.output_proj = nn.Linear(decoder_dim, embed_dim)

        # LayerNorm
        self.norm = nn.LayerNorm(decoder_dim)

    def forward(
        self,
        encoded_visible: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode and reconstruct masked tokens.

        Args:
            encoded_visible: [B, num_visible+1, embed_dim] encoded visible tokens + CLS
            mask: [B, num_tokens] boolean mask (True = masked position)

        Returns:
            reconstructed: [B, num_tokens, embed_dim] reconstructed ALL tokens
        """
        B = encoded_visible.shape[0]

        # Project to decoder dimension (skip CLS token, use only visible patches)
        visible_decoded = self.encoder_to_decoder(encoded_visible[:, 1:])  # [B, num_visible, decoder_dim]

        # Create full sequence with mask tokens (match dtype for AMP compatibility)
        full_tokens = torch.zeros(
            B, self.num_tokens, self.decoder_dim,
            device=mask.device, dtype=visible_decoded.dtype
        )

        for i in range(B):
            visible_indices = (~mask[i]).nonzero(as_tuple=True)[0]
            masked_indices = mask[i].nonzero(as_tuple=True)[0]

            # Place visible tokens
            full_tokens[i, visible_indices] = visible_decoded[i, :len(visible_indices)]

            # Place mask tokens (cast to match dtype)
            full_tokens[i, masked_indices] = self.mask_token.squeeze(0).expand(len(masked_indices), -1).to(visible_decoded.dtype)

        # Add positional embedding (cast to match dtype)
        full_tokens = full_tokens + self.decoder_pos_embed.to(visible_decoded.dtype)

        # Decode
        decoded = self.decoder(full_tokens)
        decoded = self.norm(decoded)

        # Project back to embed_dim
        reconstructed = self.output_proj(decoded)  # [B, num_tokens, embed_dim]

        return reconstructed


class ContrastiveHead(nn.Module):
    """
    Projection head for contrastive learning.

    Projects embeddings to a space where contrastive loss is applied.
    Following SimCLR: encoder -> projection -> contrastive loss
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings for contrastive learning.

        Args:
            x: [B, D] embeddings (typically CLS token)

        Returns:
            projected: [B, output_dim] normalized projections
        """
        x = self.projector(x)
        x = F.normalize(x, dim=-1)
        return x


class PretrainWrapper(nn.Module):
    """
    Wrapper for self-supervised pretraining of backbones.

    Supports two pretraining modes:
    1. "masked": Masked autoencoding - mask tokens BEFORE encoder, reconstruct pre-encoder values
    2. "contrastive": Contrastive learning - pull augmented views together

    Key difference from naive implementation:
    - Masking happens BEFORE the transformer encoder (not after)
    - Target is pre-encoder token values (not post-encoder)
    - Only visible tokens go through encoder (efficiency + better learning)

    After pretraining, extract the backbone and wrap with CPEAWrapper
    for few-shot learning.

    Usage:
        # Create backbone and pretraining wrapper
        backbone = MFTChannelBackbone(hsi_channels=144, ...)
        pretrain_model = PretrainWrapper(backbone, mode="masked")

        # Pretrain
        for batch in dataloader:
            loss = pretrain_model(hsi, aux)
            loss.backward()

        # Extract backbone for few-shot
        trained_backbone = pretrain_model.backbone
        fewshot_model = CPEAWrapper(trained_backbone)
    """

    def __init__(
        self,
        backbone: BackboneBase,
        mode: str = "masked",
        mask_ratio: float = 0.5,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        contrastive_dim: int = 128,
        temperature: float = 0.1
    ):
        super().__init__()

        assert mode in ["masked", "contrastive"], f"Unknown mode: {mode}"

        self.backbone = backbone
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.temperature = temperature

        self.num_tokens = backbone.num_tokens
        self.embed_dim = backbone.embed_dim

        if mode == "masked":
            self.decoder = MAEDecoder(
                embed_dim=self.embed_dim,
                num_tokens=self.num_tokens,
                decoder_dim=decoder_dim,
                decoder_depth=decoder_depth
            )
        else:  # contrastive
            self.contrastive_head = ContrastiveHead(
                embed_dim=self.embed_dim,
                hidden_dim=decoder_dim,
                output_dim=contrastive_dim
            )

    def generate_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate random mask for tokens.

        Args:
            batch_size: Batch size
            device: Device

        Returns:
            mask: [B, num_tokens] boolean mask (True = masked, to be reconstructed)
        """
        num_mask = int(self.num_tokens * self.mask_ratio)

        # Generate same mask pattern for all samples in batch (simpler, works well)
        mask_indices = random.sample(range(self.num_tokens), num_mask)
        mask = torch.zeros(self.num_tokens, dtype=torch.bool, device=device)
        mask[mask_indices] = True

        # Expand for batch
        return mask.unsqueeze(0).expand(batch_size, -1)

    def forward_masked(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for masked autoencoding.

        Proper MAE flow:
        1. Extract pre-encoder tokens (target for reconstruction)
        2. Mask some tokens
        3. Encode only visible tokens
        4. Decode and reconstruct ALL tokens
        5. Compute loss only on masked positions

        Args:
            hsi: [B, C, H, W] HSI data
            aux: [B, C_aux, H, W] auxiliary data

        Returns:
            loss: Reconstruction loss (MSE on masked tokens only)
            reconstructed: [B, num_tokens, D] reconstructed tokens
            target: [B, num_tokens, D] original pre-encoder tokens
        """
        B = hsi.shape[0]

        # Generate mask
        mask = self.generate_mask(B, hsi.device)  # [B, num_tokens]

        # Forward with masking - encoder only sees visible tokens
        # Returns: encoded_visible [B, num_visible+1, D], pre_encoder_tokens [B, num_tokens, D]
        encoded_visible, target, mask = self.backbone.forward_with_mask(hsi, aux, mask)

        # Decode - reconstruct all tokens from visible encodings
        reconstructed = self.decoder(encoded_visible, mask)  # [B, num_tokens, D]

        # Compute loss only on MASKED positions
        # This is the key - we only penalize reconstruction of masked tokens
        masked_recon = reconstructed[mask]  # [num_masked_total, D]
        masked_target = target[mask]  # [num_masked_total, D]

        loss = F.mse_loss(masked_recon, masked_target)

        return loss, reconstructed, target

    def forward_contrastive(
        self,
        hsi1: torch.Tensor,
        aux1: torch.Tensor,
        hsi2: torch.Tensor,
        aux2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for contrastive learning.

        Assumes hsi1/aux1 and hsi2/aux2 are two augmented views of the same samples.

        Args:
            hsi1, aux1: First view [B, C, H, W]
            hsi2, aux2: Second view [B, C, H, W]

        Returns:
            loss: InfoNCE contrastive loss
        """
        # Extract CLS embeddings from both views (full forward, no masking)
        _, cls1 = self.backbone.forward_features(hsi1, aux1)
        _, cls2 = self.backbone.forward_features(hsi2, aux2)

        # Project for contrastive loss
        z1 = self.contrastive_head(cls1)  # [B, contrastive_dim]
        z2 = self.contrastive_head(cls2)  # [B, contrastive_dim]

        # InfoNCE loss
        B = z1.shape[0]

        # Similarity matrix
        sim = torch.mm(z1, z2.T) / self.temperature  # [B, B]

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=z1.device)

        # Cross-entropy loss (both directions)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

        return loss

    def forward(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor,
        hsi2: Optional[torch.Tensor] = None,
        aux2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for pretraining.

        For masked mode: pass single view (hsi, aux)
        For contrastive mode: pass two augmented views (hsi, aux, hsi2, aux2)

        Returns:
            loss: Pretraining loss
        """
        if self.mode == "masked":
            loss, _, _ = self.forward_masked(hsi, aux)
            return loss
        else:
            if hsi2 is None or aux2 is None:
                raise ValueError("Contrastive mode requires two views (hsi2, aux2)")
            return self.forward_contrastive(hsi, aux, hsi2, aux2)

    def get_backbone(self) -> BackboneBase:
        """Extract the trained backbone for downstream tasks."""
        return self.backbone

"""
Abstract base class for backbone models.

All backbones must implement:
- forward_features() -> (patch_emb, cls_emb)
- Configurable number of output tokens
- freeze_backbone() / unfreeze_backbone() methods
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional


class BackboneBase(nn.Module, ABC):
    """
    Abstract base class for all backbone models.

    Backbones are pure feature extractors that output:
    - patch_emb: [B, N, D] - patch/token embeddings for dense similarity
    - cls_emb: [B, D] - class-aware embedding for CPEA adaptation

    # Alternative interface (commented for reference):
    # - patch_emb: [B, N, D]
    # - cls_emb: [B, D]
    # - aux_emb: [B, D] - auxiliary modality embedding
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_tokens: int = 8,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens  # Number of patch tokens (excluding CLS)

    @abstractmethod
    def forward_features(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from multimodal input.

        Args:
            hsi: [B, C, H, W] hyperspectral data
            aux: [B, C_aux, H, W] auxiliary data (e.g., LiDAR)

        Returns:
            patch_emb: [B, N, D] patch embeddings (N = num_tokens)
            cls_emb: [B, D] class-aware embedding

        # Alternative return (for compatibility with original MFTCPEA):
        # return patch_emb, cls_emb, aux_emb
        """
        pass

    def forward_tokens_only(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract tokens BEFORE transformer encoder (for MAE pretraining).

        This is the target for masked autoencoding - we want to reconstruct
        these pre-encoder token embeddings.

        Args:
            hsi: [B, C, H, W] hyperspectral data
            aux: [B, C_aux, H, W] auxiliary data

        Returns:
            hsi_tokens: [B, N, D] HSI tokens (before encoder)
            cls_token: [B, 1, D] CLS token (before encoder)

        Note: Subclasses should override this if they have different tokenization.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("Subclass must implement forward_tokens_only for MAE pretraining")

    # Alternative interface for backward compatibility
    # def forward_features_full(
    #     self,
    #     hsi: torch.Tensor,
    #     aux: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Extract features with separate aux embedding.
    #
    #     Returns:
    #         patch_emb: [B, N, D] patch embeddings
    #         cls_emb: [B, D] class-aware embedding
    #         aux_emb: [B, D] auxiliary modality embedding
    #     """
    #     pass

    def forward(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default forward calls forward_features."""
        return self.forward_features(hsi, aux)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_param_count(self) -> Dict[str, int]:
        """Get count of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }

    def get_output_info(self) -> Dict[str, int]:
        """Get information about output dimensions."""
        return {
            'num_tokens': self.num_tokens,
            'embed_dim': self.embed_dim,
            'similarity_input_size': self.num_tokens * self.num_tokens
        }

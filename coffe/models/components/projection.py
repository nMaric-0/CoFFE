"""Projection head for embedding transformation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ProjectionHead(nn.Module):
    """
    Projection head following SimCLR/BYOL best practices for few-shot learning.

    Design choices based on recent research (ICLR 2024, arXiv:2212.11491):
    - Multi-layer MLP with bottleneck expansion for richer representations
    - BatchNorm for stability during contrastive/reconstruction pretraining
    - Optional L2 normalization for metric-based few-shot learning
    - Configurable depth and width for different use cases

    For few-shot HSI classification:
    - L2 normalization is recommended for cosine similarity metrics
    - 2-layer design balances expressiveness and overfitting prevention
    - Hidden expansion (4x) captures spectral-spatial correlations
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        use_bn: bool = True,
        l2_normalize: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension (default: 4x embed_dim)
            output_dim: Output dimension (default: same as embed_dim)
            num_layers: Number of linear layers (1, 2, or 3)
            use_bn: Use BatchNorm (True) or LayerNorm (False)
            l2_normalize: Apply L2 normalization to output (recommended for metric learning)
            dropout: Dropout rate between layers (default: 0.0)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 4  # Standard expansion factor
        if output_dim is None:
            output_dim = embed_dim

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.l2_normalize = l2_normalize
        self.num_layers = num_layers

        # Build MLP layers
        layers = []
        in_dim = embed_dim

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_dim = output_dim if is_last else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if not is_last:
                # Normalization (handles both 2D and 3D inputs)
                if use_bn:
                    layers.append(_BatchNorm1dFor3D(out_dim))
                else:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., D] input embeddings (supports 2D [B, D] or 3D [B, N, D])
        Returns:
            [..., D_out] projected embeddings, optionally L2 normalized
        """
        x = self.net(x)

        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x


class _BatchNorm1dFor3D(nn.Module):
    """
    BatchNorm1d wrapper that handles both 2D [B, D] and 3D [B, N, D] inputs.

    For 3D inputs, reshapes to [B*N, D], applies BatchNorm, then reshapes back.
    This is necessary for patch embeddings in vision transformers.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.bn(x)
        elif x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
            x = self.bn(x)
            x = x.reshape(B, N, D)
            return x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")


class ModalitySpecificProjection(nn.Module):
    """
    Separate projection heads for different modalities.

    For multimodal few-shot learning, using separate projectors allows
    each modality to learn its own optimal representation space before
    fusion or similarity computation.

    Recommended for HSI + LiDAR/SAR fusion where modalities have
    different characteristics.
    """

    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 2,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        shared_output_space: bool = True,
        **kwargs
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            num_modalities: Number of separate projectors (e.g., 2 for HSI + aux)
            hidden_dim: Hidden dimension for each projector
            output_dim: Output dimension
            shared_output_space: If True, all projectors output to same dim
            **kwargs: Additional args passed to ProjectionHead
        """
        super().__init__()

        self.num_modalities = num_modalities
        self.projectors = nn.ModuleList([
            ProjectionHead(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs
            )
            for _ in range(num_modalities)
        ])

    def forward(
        self,
        *inputs: torch.Tensor,
        modality_idx: Optional[int] = None
    ) -> tuple:
        """
        Args:
            *inputs: Variable number of tensors, one per modality
            modality_idx: If provided, only project that modality

        Returns:
            Tuple of projected tensors (same order as inputs)
        """
        if modality_idx is not None:
            return self.projectors[modality_idx](inputs[0])

        if len(inputs) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} inputs, got {len(inputs)}"
            )

        return tuple(proj(x) for proj, x in zip(self.projectors, inputs))

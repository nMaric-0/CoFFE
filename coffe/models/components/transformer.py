"""Transformer encoder with optional cross-patch attention."""

import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    """Single transformer layer."""

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pre-LayerNorm self-attention + MLP, both residual."""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply every layer in turn (D=128, 2 heads, 2 layers in CoFFE)."""
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

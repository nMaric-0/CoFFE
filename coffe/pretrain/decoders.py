"""
Decoder architectures for masked autoencoder pretraining.

Two options:
1. TransformerDecoder: Full transformer decoder with cross-attention (like MAE)
2. SimpleMLP: Lightweight 1-2 layer MLP decoder

The choice of decoder affects what the encoder learns:
- Lightweight decoder forces encoder to learn better representations
- Full decoder can model more complex reconstruction but may "offload" work from encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with self-attention and cross-attention.

    Following MAE design:
    1. Self-attention on decoder tokens (including mask tokens)
    2. Cross-attention to encoder output (optional, for more context)
    3. MLP block
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cross_attention: bool = False
    ):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(embed_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, num_heads,
                dropout=dropout,
                batch_first=True
            )

        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] decoder input (with mask tokens)
            encoder_output: [B, N_enc, D] encoder output for cross-attention
            attn_mask: Optional attention mask

        Returns:
            x: [B, N, D] decoder output
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # Cross-attention (if enabled and encoder output provided)
        if self.use_cross_attention and encoder_output is not None:
            x_norm = self.norm_cross(x)
            cross_out, _ = self.cross_attn(x_norm, encoder_output, encoder_output)
            x = x + cross_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerDecoder(nn.Module):
    """
    Full transformer decoder for masked autoencoder pretraining.

    This is a more powerful decoder that can model complex reconstruction.
    Used as baseline to compare against simpler MLP decoders.

    Architecture:
    - Learnable mask tokens
    - Position embeddings for decoder
    - Stack of transformer layers
    - Linear projection to output dimension
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_tokens: int,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cross_attention: bool = False
    ):
        """
        Args:
            embed_dim: Decoder embedding dimension
            output_dim: Output dimension (e.g., num HSI channels)
            num_tokens: Number of spatial tokens
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            use_cross_attention: Whether to use cross-attention to encoder
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cross_attention=use_cross_attention
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        visible_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            visible_tokens: [B, N_vis, D] visible token embeddings from encoder
            ids_restore: [B, N] indices to restore original order
            encoder_output: [B, N_enc, D] full encoder output for cross-attention

        Returns:
            pred: [B, N, output_dim] predicted values for all tokens
        """
        B, N_vis, D = visible_tokens.shape
        N = ids_restore.shape[1]
        N_mask = N - N_vis

        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, N_mask, -1)

        # Concatenate visible and mask tokens
        tokens = torch.cat([visible_tokens, mask_tokens], dim=1)  # [B, N, D]

        # Restore original spatial order
        tokens = torch.gather(
            tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # Add position embeddings
        tokens = tokens + self.pos_embed

        # Transformer decoder
        for layer in self.layers:
            tokens = layer(tokens, encoder_output)

        tokens = self.norm(tokens)

        # Output projection
        pred = self.output_proj(tokens)  # [B, N, output_dim]

        return pred


class SimpleMLPDecoder(nn.Module):
    """
    Simple 1-layer MLP decoder.

    Minimal decoder that forces encoder to learn meaningful representations.
    Just a single linear projection from embed_dim to output_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_tokens: int
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            output_dim: Output dimension (e.g., num HSI channels)
            num_tokens: Number of spatial tokens
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Single layer projection
        self.decoder = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        visible_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            visible_tokens: [B, N_vis, D] visible token embeddings
            ids_restore: [B, N] indices to restore original order
            encoder_output: Ignored (for API compatibility)

        Returns:
            pred: [B, N, output_dim] predicted values
        """
        B, N_vis, D = visible_tokens.shape
        N = ids_restore.shape[1]
        N_mask = N - N_vis

        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, N_mask, -1)

        # Concatenate and restore order
        tokens = torch.cat([visible_tokens, mask_tokens], dim=1)
        tokens = torch.gather(
            tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # Simple projection
        pred = self.decoder(tokens)

        return pred


class TwoLayerMLPDecoder(nn.Module):
    """
    Two-layer MLP decoder (hidden layer + output).

    Slightly more capacity than 1-layer but still lightweight.
    This is the original design from the masked_modeling.py.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_tokens: int,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_tokens: Number of spatial tokens
            dropout: Dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Two-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        visible_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            visible_tokens: [B, N_vis, D] visible token embeddings
            ids_restore: [B, N] indices to restore original order
            encoder_output: Ignored (for API compatibility)

        Returns:
            pred: [B, N, output_dim] predicted values
        """
        B, N_vis, D = visible_tokens.shape
        N = ids_restore.shape[1]
        N_mask = N - N_vis

        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, N_mask, -1)

        # Concatenate and restore order
        tokens = torch.cat([visible_tokens, mask_tokens], dim=1)
        tokens = torch.gather(
            tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # Two-layer MLP
        pred = self.decoder(tokens)

        return pred


def build_decoder(
    decoder_type: str,
    embed_dim: int,
    output_dim: int,
    num_tokens: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    use_cross_attention: bool = False
) -> nn.Module:
    """
    Factory function to build decoder based on type.

    Args:
        decoder_type: One of "transformer", "mlp_1layer", "mlp_2layer"
        embed_dim: Embedding dimension
        output_dim: Output dimension
        num_tokens: Number of spatial tokens
        hidden_dim: Hidden dimension for MLP decoders
        num_layers: Number of layers for transformer decoder
        num_heads: Number of heads for transformer decoder
        dropout: Dropout rate
        use_cross_attention: Whether transformer decoder uses cross-attention

    Returns:
        Decoder module
    """
    if decoder_type == "transformer":
        return TransformerDecoder(
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_cross_attention=use_cross_attention
        )
    elif decoder_type == "mlp_1layer":
        return SimpleMLPDecoder(
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_tokens=num_tokens
        )
    elif decoder_type == "mlp_2layer":
        return TwoLayerMLPDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

"""
MFT-specific building blocks adapted from:
https://github.com/srinadh99/Transformer-Models-for-Multimodal-Remote-Sensing-Data

Components:
- HetConv: Heterogeneous convolution for HSI feature extraction
- MCrossAttention: Multi-head cross attention (Q from first token)
- LearnableTokenizer: Attention-based token generation
- MFTBlock: Transformer block with MCrossAttention
- MFTEncoder: Stack of MFT blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HetConv(nn.Module):
    """
    Heterogeneous Convolution combining groupwise and pointwise convolutions.

    This is more parameter-efficient than standard convolution for
    processing high-dimensional HSI data.

    From MFT paper: Combines spatial (groupwise) and spectral (pointwise) processing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 64,
        padding: int = 1,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size for groupwise conv
            groups: Number of groups (should divide in_channels)
            padding: Padding for groupwise conv
        """
        super().__init__()

        # Adjust groups to be valid divisor
        self.groups = min(groups, in_channels)
        while in_channels % self.groups != 0:
            self.groups -= 1

        # Groupwise convolution (spatial processing)
        self.gwconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.groups,
            bias=False,
        )

        # Pointwise convolution (channel mixing)
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels, H, W]
        """
        return self.gwconv(x) + self.pwconv(x)


class MCrossAttention(nn.Module):
    """
    Multi-head Cross Attention where queries come from the first token only.

    This is the key innovation from MFT - the CLS/fusion token attends to
    all other tokens, but other tokens don't self-attend.

    Q: from first token [B, 1, D]
    K, V: from all tokens [B, N, D]
    Output: updated first token [B, 1, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Separate projections for Q (first token) and KV (all tokens)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens (first token is CLS/fusion token)

        Returns:
            [B, N, D] with first token updated via cross-attention
        """
        B, N, D = x.shape

        # Q from first token only
        q = self.q_proj(x[:, 0:1, :])  # [B, 1, D]
        q = q.reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # K, V from all tokens
        k = self.k_proj(x)  # [B, N, D]
        v = self.v_proj(x)  # [B, N, D]
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention: [B, heads, 1, head_dim] @ [B, heads, head_dim, N] -> [B, heads, 1, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = attn @ v  # [B, heads, 1, head_dim]
        out = out.transpose(1, 2).reshape(B, 1, D)  # [B, 1, D]
        out = self.proj(out)
        out = self.proj_drop(out)

        # Only update first token, keep others unchanged
        result = x.clone()
        result[:, 0:1, :] = out

        return result


class StandardSelfAttention(nn.Module):
    """
    Standard multi-head self-attention for comparison.
    All tokens attend to all tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, mlp_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4

        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(dropout)

        # Xavier initialization as in MFT
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two-layer feed-forward block (Xavier-initialised, as in MFT)."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MFTBlock(nn.Module):
    """
    Transformer block with configurable attention type.

    Structure: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_dim: int | None = None,
        dropout: float = 0.1,
        attention_type: str = "mcross",  # "mcross" or "standard"
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn: nn.Module
        if attention_type == "mcross":
            self.attn = MCrossAttention(
                dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout
            )
        else:
            self.attn = StandardSelfAttention(
                dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout
            )

        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm attention + MLP with residual connections."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MFTEncoder(nn.Module):
    """
    Stack of MFT blocks forming the transformer encoder.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 8,
        mlp_dim: int | None = None,
        dropout: float = 0.1,
        attention_type: str = "mcross",
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [MFTBlock(dim, num_heads, mlp_dim, dropout, attention_type) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D] encoded tokens
        """
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class LearnableTokenizer(nn.Module):
    """
    MFT-style learnable tokenization via attention.

    Converts spatial features into a fixed number of tokens using
    learnable attention weights.

    Formula:
        A = softmax(X @ token_wA.T)  # Attention weights
        V = X @ token_wV             # Value projection
        tokens = A.T @ V             # Weighted aggregation
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_tokens: int,
        token_type: str = "channel",  # "channel" or "pixel"
    ):
        """
        Args:
            input_dim: Input feature dimension
            embed_dim: Output embedding dimension
            num_tokens: Number of output tokens
            token_type: Tokenization strategy
                - "channel": attention over channels
                - "pixel": attention over spatial positions
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.token_type = token_type

        # Learnable attention and value weights
        if token_type == "channel":
            # Attention over feature dimension
            self.token_wA = nn.Parameter(torch.empty(num_tokens, input_dim))
            self.token_wV = nn.Parameter(torch.empty(input_dim, embed_dim))
        else:  # pixel
            # Attention over spatial dimension (single weight per token)
            self.token_wA = nn.Parameter(torch.empty(num_tokens, 1))
            self.token_wV = nn.Parameter(torch.empty(input_dim, embed_dim))

        # Initialize
        nn.init.xavier_uniform_(self.token_wA)
        nn.init.xavier_uniform_(self.token_wV)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] flattened spatial features
               N = H*W (spatial positions)
               D = input_dim (feature channels)

        Returns:
            tokens: [B, num_tokens, embed_dim]
        """
        B, N, D = x.shape

        if self.token_type == "channel":
            # Attention over feature dimension
            # [B, N, D] @ [D, num_tokens] -> [B, N, num_tokens]
            attn = torch.matmul(x, self.token_wA.T)
            attn = F.softmax(attn, dim=1)  # Softmax over spatial positions

            # [B, N, D] @ [D, embed_dim] -> [B, N, embed_dim]
            values = torch.matmul(x, self.token_wV)

            # [B, num_tokens, N] @ [B, N, embed_dim] -> [B, num_tokens, embed_dim]
            tokens = torch.matmul(attn.transpose(1, 2), values)

        else:  # pixel
            # Attention with spatial-agnostic weights
            # Broadcast attention weights across spatial dimension
            # [B, N, 1] @ [1, num_tokens] -> [B, N, num_tokens]
            attn = F.softmax(self.token_wA.T.expand(B, N, -1), dim=1)

            # [B, N, D] @ [D, embed_dim] -> [B, N, embed_dim]
            values = torch.matmul(x, self.token_wV)

            # [B, num_tokens, N] @ [B, N, embed_dim] -> [B, num_tokens, embed_dim]
            tokens = torch.matmul(attn.transpose(1, 2), values)

        return tokens


class HSI3DConv(nn.Module):
    """
    3D convolution for HSI feature extraction as used in MFT.

    Processes spectral-spatial information jointly.
    """

    def __init__(
        self, in_channels: int, out_channels: int, spectral_kernel: int = 7, spatial_kernel: int = 3
    ):
        """
        Args:
            in_channels: Number of HSI bands (spectral dimension)
            out_channels: Output feature channels
            spectral_kernel: Kernel size along spectral dimension
            spatial_kernel: Kernel size along spatial dimensions
        """
        super().__init__()

        # 3D conv: (in_channels, D, H, W) -> (out_channels, D', H', W')
        # We treat spectral as depth dimension
        self.conv3d = nn.Conv3d(
            1,
            out_channels,
            kernel_size=(spectral_kernel, spatial_kernel, spatial_kernel),
            padding=(spectral_kernel // 2, spatial_kernel // 2, spatial_kernel // 2),
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] HSI data (C = spectral bands)

        Returns:
            [B, out_channels, C, H, W] 3D features
        """
        # Add channel dimension for 3D conv: [B, 1, C, H, W]
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.act(x)
        return x

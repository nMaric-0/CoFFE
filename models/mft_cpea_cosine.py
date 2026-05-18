"""
MFT-CPEA-Cosine: Multimodal Fusion Transformer with Cosine Similarity

Variant of MFT-CPEA that uses prototypical networks with cosine similarity
instead of DenseSimilarity MLP. This version is ideal for zero-shot evaluation
(no finetuning) as it doesn't rely on learnable similarity parameters.

Key differences from MFTCPEA:
- No DenseSimilarity MLP (no trainable params in similarity computation)
- Uses prototypical networks with cosine similarity
- Better for pretrained models without finetuning
- Simpler and faster inference

Usage:
    # For zero-shot evaluation with pretrained encoder
    model = MFTCPEACosine(
        hsi_channels=144,
        aux_channels=1,
        embed_dim=128,
        ...
    )

    # Load pretrained weights (same as MFTCPEA)
    model.load_state_dict(checkpoint)

    # Evaluate directly (no finetuning needed)
    logits = model.forward_episode(...)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .components.tokenizers import ChannelTokenizer, SpatialTokenizer
from .components.transformer import TransformerEncoder
from .components.projection import ProjectionHead
from utils.spatial_weights import make_center_weights, center_weighted_pool


class MFTCPEACosine(nn.Module):
    """
    Multimodal Fusion Transformer with Cosine Similarity for Few-Shot Learning.

    This variant uses prototypical networks with cosine similarity instead of
    the DenseSimilarity MLP, making it ideal for zero-shot evaluation without
    finetuning. All pretrained MFTCPEA weights are compatible.

    Combines:
    - MFT: Multimodal fusion via mCrossPA with auxiliary data as CLS
    - CPEA: Class-aware patch embedding adaptation
    - Prototypical Networks: Cosine similarity to class prototypes

    Args:
        hsi_channels: Number of HSI spectral bands
        aux_channels: Number of auxiliary data channels
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        patch_size: Spatial patch size
        lambda_factor: Class-aware adaptation factor
        dropout: Dropout rate
        use_projection: Whether to use projection head
        proj_hidden_dim: Projection head hidden dimension (default: 4x embed_dim)
        proj_num_layers: Number of projection head layers (1, 2, or 3)
        proj_l2_normalize: Whether to L2 normalize projection output
        distance_metric: "cosine" or "euclidean" for prototype matching
        temperature: Temperature scaling for cosine similarity (default: 10.0)
        prototype_mode: "mean_features" (average features then distance) or
                       "mean_distances" (distance to each, then average)
    """

    def __init__(
        self,
        hsi_channels: int,
        aux_channels: int = 1,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        patch_size: int = 11,
        lambda_factor: float = 2.0,
        dropout: float = 0.1,
        use_projection: bool = True,
        proj_hidden_dim: Optional[int] = None,
        proj_num_layers: int = 2,
        proj_l2_normalize: bool = True,
        distance_metric: str = "cosine",
        temperature: float = 10.0,
        prototype_mode: str = "mean_features",
        pool_sigma: Optional[float] = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.lambda_factor = lambda_factor
        self.num_tokens = patch_size * patch_size
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.prototype_mode = prototype_mode
        self.pool_sigma = pool_sigma

        # Center-weighted spatial pooling (None = uniform mean pooling)
        if pool_sigma is not None:
            self.register_buffer(
                '_center_pool_weights',
                make_center_weights(patch_size, pool_sigma, normalize=True)
            )

        # Track channel counts (HSI + aux concatenated at input)
        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels
        self.total_channels = hsi_channels + aux_channels

        # ========== Unified Tokenizer ==========
        # HSI and auxiliary bands are concatenated along the channel dim and
        # embedded by a single channel+spatial tokenizer. One token per pixel
        # carries information from both HSI bands and aux (e.g., LiDAR) bands.
        self.channel_tokenizer = ChannelTokenizer(
            in_channels=self.total_channels,
            embed_dim=embed_dim
        )
        self.spatial_tokenizer = SpatialTokenizer(
            embed_dim=embed_dim
        )

        # ========== Class-Agnostic Embedding (CPEA) ==========
        self.class_agnostic_emb = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )

        # ========== Position Embedding ==========
        # +1 for CLS token, + num_tokens for the unified (HSI+aux) patch tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_tokens, embed_dim) * 0.02
        )

        # ========== Transformer Encoder ==========
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ========== Projection Head ==========
        if use_projection:
            self.projection = ProjectionHead(
                embed_dim=embed_dim,
                hidden_dim=proj_hidden_dim,
                output_dim=embed_dim,
                num_layers=proj_num_layers,
                use_bn=False,
                l2_normalize=proj_l2_normalize
            )
        else:
            self.projection = nn.Identity()

        # NOTE: No DenseSimilarity - we use cosine similarity directly

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_trainable_param_count(self) -> Dict[str, int]:
        """Get count of trainable parameters."""
        counts = {
            'total_trainable': 0,
            'total': 0,
        }

        for name, param in self.named_parameters():
            counts['total'] += param.numel()
            if param.requires_grad:
                counts['total_trainable'] += param.numel()

        return counts

    def tokenize(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> torch.Tensor:
        """
        Tokenize multimodal input.

        HSI and auxiliary bands are concatenated along the channel dimension,
        producing one token per spatial position that carries both modalities.

        Args:
            hsi: [B, C_hsi, H, W] hyperspectral data
            aux: [B, C_aux, H, W] auxiliary data

        Returns:
            patch_tokens: [B, N, D] unified patch tokens (one per pixel)
        """
        combined = torch.cat([hsi, aux], dim=1)         # [B, C_hsi + C_aux, H, W]
        tokens = self.channel_tokenizer(combined)       # [B, D, H, W]
        tokens = self.spatial_tokenizer(tokens)         # [B, N, D]
        return tokens

    def forward_features(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from multimodal input.

        Args:
            hsi: [B, C, H, W] hyperspectral data
            aux: [B, C_aux, H, W] auxiliary data

        Returns:
            patch_emb: [B, N, D] patch embeddings
            cls_emb: [B, D] class-aware embedding
            aux_emb: [B, D] auxiliary embedding (alias for cls_emb in unified tokenizer)
        """
        B = hsi.shape[0]

        # Tokenize (unified: HSI+aux concatenated as extra bands on each pixel)
        patch_tokens = self.tokenize(hsi, aux)  # [B, N, D]

        # Expand class-agnostic embedding
        cls_token = self.class_agnostic_emb.expand(B, -1, -1)

        # Concatenate: [cls(1), patch_tokens(N)]
        tokens = torch.cat([cls_token, patch_tokens], dim=1)

        # Add position embedding
        tokens = tokens + self.pos_embed

        # Transformer encoding
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # Split outputs
        cls_emb = tokens[:, 0]          # [B, D]
        patch_emb = tokens[:, 1:]       # [B, N, D]

        # Project
        cls_emb = self.projection(cls_emb)
        patch_emb = self.projection(patch_emb)

        # Backward-compatible return signature: aux_emb is no longer a distinct
        # modality token, so return cls_emb as a stand-in for downstream code.
        return patch_emb, cls_emb, cls_emb

    def adapt_embeddings(
        self,
        patch_emb: torch.Tensor,
        cls_emb: torch.Tensor,
        lambda_factor: Optional[float] = None,
        renormalize: bool = True
    ) -> torch.Tensor:
        """
        Class-aware patch embedding adaptation.

        z̄ᵢ = zᵢ + λ · z_class

        Args:
            patch_emb: [B, N, D] original patch embeddings
            cls_emb: [B, D] class-aware embedding
            lambda_factor: adaptation strength
            renormalize: Re-apply L2 normalization after adaptation

        Returns:
            adapted: [B, N, D] class-relevant embeddings
        """
        if lambda_factor is None:
            lambda_factor = self.lambda_factor

        adapted = patch_emb + lambda_factor * cls_emb.unsqueeze(1)

        if renormalize and hasattr(self.projection, 'l2_normalize') and self.projection.l2_normalize:
            adapted = F.normalize(adapted, p=2, dim=-1)

        return adapted

    def compute_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes by averaging features.

        Args:
            features: [N*K, D] feature vectors
            labels: [N*K] class labels (0 to N-1)

        Returns:
            prototypes: [N, D] class prototypes
        """
        num_classes = labels.max().item() + 1
        # Vectorized prototype computation using scatter
        prototypes = torch.zeros(num_classes, features.shape[-1], device=features.device)
        counts = torch.zeros(num_classes, device=features.device)
        prototypes.scatter_add_(0, labels.unsqueeze(-1).expand_as(features), features)
        counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
        prototypes = prototypes / counts.unsqueeze(-1).clamp(min=1)

        return prototypes

    def forward_episode(
        self,
        support_hsi: torch.Tensor,
        support_aux: torch.Tensor,
        support_labels: torch.Tensor,
        query_hsi: torch.Tensor,
        query_aux: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for few-shot episode using prototypical networks.

        Uses cosine similarity (or Euclidean distance) to class prototypes.
        No learnable similarity parameters - ideal for zero-shot evaluation.

        Two prototype modes:
        - "mean_features": Average support features first, then compute distance (standard)
        - "mean_distances": Compute distance to each support, then average distances

        Args:
            support_hsi: [N*K, C, H, W] support HSI
            support_aux: [N*K, C_aux, H, W] support auxiliary
            support_labels: [N*K] support labels (0 to N-1)
            query_hsi: [N*Q, C, H, W] query HSI
            query_aux: [N*Q, C_aux, H, W] query auxiliary

        Returns:
            logits: [N*Q, N] classification logits
        """
        # Extract features
        s_patch, s_cls, _ = self.forward_features(support_hsi, support_aux)
        q_patch, q_cls, _ = self.forward_features(query_hsi, query_aux)

        # Adapt embeddings
        s_adapted = self.adapt_embeddings(s_patch, s_cls)
        q_adapted = self.adapt_embeddings(q_patch, q_cls)

        # Spatial pooling over patch tokens (center-weighted or uniform)
        if self.pool_sigma is not None:
            s_features = center_weighted_pool(s_adapted, self._center_pool_weights)  # [N*K, D]
            q_features = center_weighted_pool(q_adapted, self._center_pool_weights)  # [N*Q, D]
        else:
            s_features = s_adapted.mean(dim=1)  # [N*K, D]
            q_features = q_adapted.mean(dim=1)  # [N*Q, D]

        # Normalize for cosine similarity (if not already normalized)
        if self.distance_metric == "cosine":
            s_features = F.normalize(s_features, p=2, dim=-1)
            q_features = F.normalize(q_features, p=2, dim=-1)

        if self.prototype_mode == "mean_distances":
            # Compute distance to each support example, then average per class
            logits = self._forward_mean_distances(s_features, support_labels, q_features)
        else:
            # Standard: average features first, then compute distance
            logits = self._forward_mean_features(s_features, support_labels, q_features)

        return logits

    def _forward_mean_features(
        self,
        s_features: torch.Tensor,
        support_labels: torch.Tensor,
        q_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard prototypical networks: average features, then distance.

        Args:
            s_features: [N*K, D] support features
            support_labels: [N*K] support labels
            q_features: [N*Q, D] query features

        Returns:
            logits: [N*Q, N] classification logits
        """
        # Compute class prototypes (mean of features)
        prototypes = self.compute_prototypes(s_features, support_labels)  # [N, D]

        # Compute similarity/distance to prototypes
        if self.distance_metric == "cosine":
            # Cosine similarity (already normalized)
            # [N*Q, D] @ [D, N] -> [N*Q, N]
            logits = torch.matmul(q_features, prototypes.T) * self.temperature
        else:  # euclidean
            # Negative squared Euclidean distance
            dists = torch.cdist(q_features, prototypes, p=2)
            logits = -dists.pow(2)

        return logits

    def _forward_mean_distances(
        self,
        s_features: torch.Tensor,
        support_labels: torch.Tensor,
        q_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean distances: compute distance to each support, then average per class.

        This computes similarity/distance to each individual support example,
        then averages the similarities/distances within each class.

        Args:
            s_features: [N*K, D] support features
            support_labels: [N*K] support labels
            q_features: [N*Q, D] query features

        Returns:
            logits: [N*Q, N] classification logits
        """
        num_classes = support_labels.max().item() + 1
        num_queries = q_features.shape[0]
        device = q_features.device

        # Compute similarity/distance to ALL support examples
        if self.distance_metric == "cosine":
            # Cosine similarity: [N*Q, D] @ [D, N*K] -> [N*Q, N*K]
            all_similarities = torch.matmul(q_features, s_features.T) * self.temperature
        else:  # euclidean
            # Negative squared Euclidean distance: [N*Q, N*K]
            all_distances = torch.cdist(q_features, s_features, p=2)
            all_similarities = -all_distances.pow(2)

        # Average similarities per class (vectorized)
        logits = torch.zeros(num_queries, num_classes, device=device)
        for c in range(num_classes):
            mask = (support_labels == c)
            logits[:, c] = all_similarities[:, mask].mean(dim=1)

        return logits

    def forward(
        self,
        hsi: torch.Tensor,
        aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard forward pass (for pretraining or feature extraction).

        Args:
            hsi: [B, C, H, W] hyperspectral data
            aux: [B, C_aux, H, W] auxiliary data

        Returns:
            patch_emb, cls_emb, aux_emb
        """
        return self.forward_features(hsi, aux)

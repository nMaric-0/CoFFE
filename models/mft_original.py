"""
Original MFT (Multimodal Fusion Transformer) — the architectural control.

Faithful port of the Multimodal Fusion Transformer (Roy et al., 2023,
"Multimodal Fusion Transformer for Remote Sensing Image Classification",
arXiv:2203.16952; github.com/srinadh99/Transformer-Models-for-Multimodal-Remote-Sensing-Data),
wired as the *architectural control* against CoFFE: same masked objectives
(:mod:`pretrain.mft_mae` for MAE, :mod:`pretrain.mft_spatial_mae` for SimMIM
token), same frozen-encoder nearest-class-mean protocol as ``CoFFE`` and
``HyperSIGMAFewShot`` (``scripts/evaluate.py``), but MFT's external fusion token
instead of CoFFE's input-level fusion.

This is the **channel-tokenization** variant: the auxiliary modality (LiDAR/DSM)
is aggregated into a single external CLS token via the learnable channel
tokenizer, and the transformer fuses it with the HSI tokens through multihead
cross-patch attention (mCrossPA), exactly as in the MFT paper.

Spatial-token MAE reconciliation
--------------------------------
Standard MAE (He et al., 2022) masks a high fraction of *spatial* tokens and
reconstructs their pixels; the encoder sees only the visible subset. The MFT's
``LearnableTokenizer`` compresses HSI into only a handful of aggregate tokens,
which is degenerate under 75% masking. So here the HSI is represented as the
**121 per-pixel spatial tokens** produced by the MFT Conv3D+HetConv front-end
(the un-compressed limit of MFT tokenization) — the fusion (LiDAR→CLS channel
tokenization + mCrossPA) stays faithful, while the token geometry now matches
``CoFFE`` (121 tokens) so the canonical MAE machinery and the shared
evaluator apply unchanged.

Eval feature = the CLS token
----------------------------
mCrossPA updates only the CLS token (token 0); the HSI patch tokens pass through
unchanged. The discriminative fused representation is therefore the encoded CLS,
so :meth:`forward_features` packs it as a single patch token
(``patch_emb = cls_emb.unsqueeze(1)``) — exactly how ``HyperSIGMAFewShot`` packs
its fused feature — and the shared eval loop's ``patch_emb.mean(dim=1)`` recovers
it. No class-token folding: :meth:`eval_patch_embeddings` is a no-op (clean original-MFT
baseline).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .components.mft_blocks import HetConv, LearnableTokenizer, MFTEncoder


class MFTOriginal(nn.Module):
    """
    Original MFT (channel-tokenization) encoder, evaluated by nearest class mean.

    Args:
        hsi_channels: Number of HSI spectral bands.
        aux_channels: Number of auxiliary (LiDAR/DSM) channels.
        use_aux: Must be True — MFT's external CLS token is derived from the
            auxiliary modality. Kept for interface symmetry with the other
            models; an HSI-only MFT is degenerate and not supported.
        embed_dim: Transformer embedding dimension. Faithful MFT: dim = FM*4 = 64
            (the authors use FM=16).
        num_heads: Number of attention heads (faithful MFT: 8).
        num_layers: Number of mCrossPA transformer layers / depth (faithful: 2).
        mlp_dim: Transformer feed-forward width (faithful MFT: a fixed 512, i.e.
            dim*8 at dim=64 — NOT dim*4).
        patch_size: Input patch spatial size (token count = patch_size ** 2).
        dropout: Dropout rate.
        attention_type: "mcross" (faithful MFT) or "standard" (self-attention,
            for ablation).
        distance_metric: "cosine" or "euclidean" for prototype matching.
        temperature: Temperature scaling for cosine similarity.
        prototype_mode: "mean_features" or "mean_distances".
    """

    def __init__(
        self,
        hsi_channels: int,
        aux_channels: int = 1,
        use_aux: bool = True,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_dim: int = 512,
        patch_size: int = 11,
        dropout: float = 0.1,
        attention_type: str = "mcross",
        distance_metric: str = "cosine",
        temperature: float = 10.0,
        prototype_mode: str = "mean_features",
        pool_sigma: Optional[float] = None,
    ):
        super().__init__()

        if not use_aux:
            raise ValueError(
                "MFTOriginal requires use_aux=True: the MFT CLS token is "
                "derived from the auxiliary modality. HSI-only MFT is degenerate."
            )

        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels
        self.use_aux = use_aux
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_tokens = patch_size * patch_size  # 121 spatial HSI tokens
        self.attention_type = attention_type
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.prototype_mode = prototype_mode
        # The eval loop pools patch tokens; MFT packs the fused CLS as a single
        # token, so center-weighted pooling is not applicable (kept None).
        self.pool_sigma = pool_sigma

        self.mlp_dim = mlp_dim

        # ===== HSI feature extraction (faithful MFT front-end) =====
        # Conv3d(1, 8, (9,3,3)) exactly as the authors: the spectral kernel (9) is
        # VALID (no spectral padding) so the spectral dim goes NC -> NC-8, while the
        # spatial 3x3 is padded so the 11x11 grid is PRESERVED (giving 121 spatial
        # tokens — the agreed deviation; this preserves the grid, it does not enlarge
        # the patch). The 8*(NC-8) channels are then mixed by a HetConv to embed_dim,
        # matching MFT's `HetConv(8*(NC-8), FM*4)`.
        self._spec_out = hsi_channels - 8  # spectral dim after the valid 9-kernel
        if self._spec_out < 1:
            raise ValueError(
                f"hsi_channels={hsi_channels} too small for the MFT 3D conv "
                f"(needs > 8 spectral bands)."
            )
        self.hsi_conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(9, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )
        self.hsi_hetconv = HetConv(
            in_channels=8 * self._spec_out,   # after reshape from 3D
            out_channels=embed_dim,           # = FM*4
            kernel_size=3,
            groups=8,                         # 8 = the 3D-conv feature maps; divides
            padding=1,                        # both 8*(NC-8) and embed_dim for all datasets
        )

        # ===== Auxiliary (LiDAR) -> single external CLS token (channel tok) =====
        # Conv2d(NCLidar, FM*4, 3, 1, 1) then channel tokenization to 1 CLS token.
        self.aux_conv = nn.Sequential(
            nn.Conv2d(aux_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.aux_tokenizer = LearnableTokenizer(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_tokens=1,           # single fusion / CLS token
            token_type="channel",
        )

        # ===== Positional embedding: +1 for the CLS token =====
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_tokens, embed_dim) * 0.02
        )

        # ===== mCrossPA transformer encoder (faithful: 8 heads, depth 2, mlp 512) =====
        self.encoder = MFTEncoder(
            dim=embed_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Tokenization (used by both the MAE wrapper and forward_features)
    # ------------------------------------------------------------------
    def tokenize(self, hsi: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        """HSI -> [B, num_tokens(=121), embed_dim] spatial tokens.

        ``aux`` is accepted (and ignored) only so the call signature matches the
        ``encoder.tokenize(hsi, aux)`` contract used by the MAE wrapper; the
        external CLS is built separately via :meth:`make_cls`.
        """
        B, C, H, W = hsi.shape
        x = hsi.unsqueeze(1)                # [B, 1, C, H, W] (spectral = depth)
        x = self.hsi_conv3d(x)              # [B, 8, C-8, H, W] (spectral valid)
        x = x.reshape(B, -1, H, W)          # [B, 8*(C-8), H, W]
        x = self.hsi_hetconv(x)             # [B, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)    # [B, H*W, embed_dim]
        return x

    def make_cls(self, aux: torch.Tensor) -> torch.Tensor:
        """LiDAR/DSM -> [B, 1, embed_dim] external CLS token (channel tokenization)."""
        x = self.aux_conv(aux)              # [B, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)    # [B, H*W, embed_dim]
        cls = self.aux_tokenizer(x)         # [B, 1, embed_dim]
        return cls

    # ------------------------------------------------------------------
    # Few-shot eval contract (mirrors CoFFE / HyperSIGMAFewShot)
    # ------------------------------------------------------------------
    def forward_features(
        self,
        hsi: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            patch_emb: [B, 1, D] — the encoded CLS packed as a single token.
            cls_emb:   [B, D]    — the encoded CLS (fused) feature.
            aux_emb:   [B, D]    — alias of cls_emb (interface compatibility).
        """
        B = hsi.shape[0]
        hsi_tokens = self.tokenize(hsi)         # [B, N, D]
        cls_token = self.make_cls(aux)          # [B, 1, D]

        tokens = torch.cat([cls_token, hsi_tokens], dim=1)  # [B, 1+N, D]
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        cls_emb = tokens[:, 0]                  # [B, D]
        patch_emb = cls_emb.unsqueeze(1)        # [B, 1, D] (single-token packing)
        return patch_emb, cls_emb, cls_emb

    def eval_patch_embeddings(
        self,
        patch_emb: torch.Tensor,
        cls_emb: torch.Tensor,
        cls_token_weight: Optional[float] = None,
        renormalize: bool = True,
    ) -> torch.Tensor:
        """No class-token folding in the MFT control — patch tokens pass through
        unchanged, so the CoFFE-only adaptation of PAPER_CANON §8 D3 cannot
        touch this route (its eval configs record ``lambda_factor: None``)."""
        return patch_emb

    def compute_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Average features per class to form prototypes (vectorized scatter)."""
        num_classes = labels.max().item() + 1
        prototypes = torch.zeros(num_classes, features.shape[-1], device=features.device)
        counts = torch.zeros(num_classes, device=features.device)
        prototypes.scatter_add_(0, labels.unsqueeze(-1).expand_as(features), features)
        counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
        prototypes = prototypes / counts.unsqueeze(-1).clamp(min=1)
        return prototypes

    def _forward_mean_features(
        self,
        s_features: torch.Tensor,
        support_labels: torch.Tensor,
        q_features: torch.Tensor,
    ) -> torch.Tensor:
        """Nearest class mean: average the support features, then measure distance."""
        prototypes = self.compute_prototypes(s_features, support_labels)  # [N, D]
        if self.distance_metric == "cosine":
            logits = torch.matmul(q_features, prototypes.T) * self.temperature
        else:  # euclidean
            dists = torch.cdist(q_features, prototypes, p=2)
            logits = -dists.pow(2)
        return logits

    def _forward_mean_distances(
        self,
        s_features: torch.Tensor,
        support_labels: torch.Tensor,
        q_features: torch.Tensor,
    ) -> torch.Tensor:
        """Distance to each support example, then averaged per class."""
        num_classes = support_labels.max().item() + 1
        num_queries = q_features.shape[0]
        device = q_features.device
        if self.distance_metric == "cosine":
            all_similarities = torch.matmul(q_features, s_features.T) * self.temperature
        else:  # euclidean
            all_distances = torch.cdist(q_features, s_features, p=2)
            all_similarities = -all_distances.pow(2)
        logits = torch.zeros(num_queries, num_classes, device=device)
        for c in range(num_classes):
            mask = (support_labels == c)
            logits[:, c] = all_similarities[:, mask].mean(dim=1)
        return logits

    def get_trainable_param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "total_trainable": trainable}

    def forward(
        self,
        hsi: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_features(hsi, aux)

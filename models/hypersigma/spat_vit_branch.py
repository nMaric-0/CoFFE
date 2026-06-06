"""SpatViT-B (``SpatViT_fusion_patch``) wrapper for small HSI patches.

The wrapper:

* constructs the upstream ``SpatViT`` with ``patch_size=3`` and
  ``img_size=12`` (so ``num_patches = 16``) and ``use_abs_pos_emb=True``;
  reflection-pads the 11x11 input to 12x12 on the fly,
* loads the released SpatViT-B checkpoint, stripping the keys that no
  longer fit (``patch_embed.proj.*``, ``pos_embed``, ``fpn*``,
  ``cls*``, ``classifier*``) and asserting nothing important was
  silently discarded,
* re-initializes the dropped components randomly (``patch_embed.proj``
  becomes ``Conv2d(in_chans, 768, k=3, s=3)``, ``pos_embed`` becomes
  ``(1, 16, 768)`` trunc-normal), and
* freezes the transformer body when ``freeze_body=True``, leaving only
  ``patch_embed.proj`` and ``pos_embed`` trainable.

NOTE: ``SpatViT_fusion_patch`` does **not** use a CLS token. The
upstream ``pos_embed`` shape is ``(1, num_patches, embed_dim)`` with no
``+1`` for CLS.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.HyperSIGMA.ImageClassification.model import SpatViT_fusion_patch

logger = logging.getLogger(__name__)


# Keys we deliberately drop from the released SpatViT-B checkpoint before
# loading into our model. Anything outside this set that ends up missing
# or unexpected is a real problem and we log it.
_EXPECTED_DROPPED_KEYS = (
    "patch_embed.proj.",  # kernel changes from 16 -> 3
    "pos_embed",          # token count changes from 14*14 to 4*4
    "fpn1.", "fpn2.", "fpn3.", "fpn4.",  # k=3 fpn is Identity, no params
    "cls",                # cls_token / cls Conv2d (downstream-only)
    "classifier.", "classifier1.",       # downstream classifier heads
)


def _looks_like_dropped_key(key: str) -> bool:
    return any(key.startswith(p) or key == p for p in _EXPECTED_DROPPED_KEYS)


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip ``module.`` / ``encoder.`` prefixes if present."""
    if not state_dict:
        return state_dict
    first = next(iter(state_dict))
    if first.startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    if all(k.startswith("encoder.") for k in state_dict):
        state_dict = {k[len("encoder."):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(ckpt: dict) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("state_dict", "model", "encoder"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


class SpatViTBranch(nn.Module):
    """HyperSIGMA SpatViT branch adapted for 11x11 patches with k=3."""

    def __init__(
        self,
        pretrained_path: str,
        in_chans: int = 3,
        new_patch_size: int = 3,
        pad_to: int = 12,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        out_indices: Tuple[int, ...] = (3, 5, 7, 11),
        freeze_body: bool = True,
        log_dropped_keys: bool = True,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.new_patch_size = new_patch_size
        self.pad_to = pad_to
        self.embed_dim = embed_dim
        self.out_indices = tuple(out_indices)

        # Construct upstream model directly at the target geometry.
        self.model = SpatViT_fusion_patch.SpatViT(
            img_size=pad_to,
            in_chans=in_chans,
            patch_size=new_patch_size,
            drop_path_rate=0.1,
            out_indices=list(out_indices),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            use_checkpoint=False,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8,
        )

        # Sanity check on the freshly-built encoder.
        assert self.model.patch_embed.num_patches == (pad_to // new_patch_size) ** 2, (
            "SpatViT patch_embed num_patches mismatch"
        )
        assert self.model.pos_embed is not None and self.model.pos_embed.shape == (
            1, self.model.patch_embed.num_patches, embed_dim,
        ), f"unexpected pos_embed shape {tuple(self.model.pos_embed.shape)}"

        self._dropped_keys: List[str] = []
        self._unexpected_keys: List[str] = []
        self._missing_keys: List[str] = []
        self.pos_embed_source = "reinit"

        if pretrained_path is not None:
            self._load_checkpoint(pretrained_path)

        if log_dropped_keys:
            self._log_load_summary()

        if freeze_body:
            self._freeze_body()

    def _load_checkpoint(self, path: str) -> None:
        # weights_only=False: the upstream HyperSIGMA checkpoint pickles an
        # argparse.Namespace alongside the state dict, which PyTorch 2.6's
        # default safe-unpickler rejects. The file is trusted (WHU-Sigma HF).
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(ckpt)
        state_dict = _strip_state_dict_prefix(state_dict)

        # Inspect pos_embed BEFORE dropping it.
        ckpt_pos_embed = state_dict.get("pos_embed")

        kept: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if _looks_like_dropped_key(k):
                self._dropped_keys.append(k)
                continue
            kept[k] = v

        missing, unexpected = self.model.load_state_dict(kept, strict=False)
        self._missing_keys = [k for k in missing if not _looks_like_dropped_key(k)]
        self._unexpected_keys = list(unexpected)

        # Re-init patch_embed.proj (kernel changed).
        nn.init.trunc_normal_(self.model.patch_embed.proj.weight, std=0.02)
        if self.model.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.model.patch_embed.proj.bias)

        # pos_embed: load-then-interp-then-reinit decision tree.
        self._load_pos_embed(ckpt_pos_embed)

    def _load_pos_embed(self, ckpt_pos_embed) -> None:
        target = self.model.pos_embed  # [1, num_patches, embed_dim]
        if ckpt_pos_embed is None:
            nn.init.trunc_normal_(target, std=0.02)
            self.pos_embed_source = "reinit"
            return

        if ckpt_pos_embed.shape == tuple(target.shape):
            with torch.no_grad():
                target.copy_(ckpt_pos_embed)
            self.pos_embed_source = "loaded"
            return

        # 2-D bicubic interpolate if the embed dim matches and both grids
        # are square. SpatViT has no CLS token, so no extra-token strip.
        if (
            ckpt_pos_embed.dim() == 3
            and ckpt_pos_embed.shape[0] == 1
            and ckpt_pos_embed.shape[2] == target.shape[2]
        ):
            orig_n = ckpt_pos_embed.shape[1]
            orig_side = int(round(orig_n ** 0.5))
            new_n = target.shape[1]
            new_side = int(round(new_n ** 0.5))
            if orig_side * orig_side == orig_n and new_side * new_side == new_n:
                pos = ckpt_pos_embed.reshape(1, orig_side, orig_side, -1).permute(0, 3, 1, 2)
                pos = F.interpolate(pos, size=(new_side, new_side), mode="bicubic", align_corners=False)
                pos = pos.permute(0, 2, 3, 1).reshape(1, new_n, -1)
                with torch.no_grad():
                    target.copy_(pos)
                self.pos_embed_source = f"interpolated_{orig_side}x{orig_side}->{new_side}x{new_side}"
                return

        nn.init.trunc_normal_(target, std=0.02)
        self.pos_embed_source = "reinit"

    def _log_load_summary(self) -> None:
        logger.info(
            "[HyperSIGMA] SpatViT load summary: dropped=%d, missing=%d (post-reinit), unexpected=%d, pos_embed=%s",
            len(self._dropped_keys), len(self._missing_keys), len(self._unexpected_keys),
            self.pos_embed_source,
        )
        if self._missing_keys:
            logger.warning(
                "[HyperSIGMA] SpatViT missing keys after re-init (first 10): %s",
                self._missing_keys[:10],
            )
        if self._unexpected_keys:
            logger.warning(
                "[HyperSIGMA] SpatViT unexpected keys (first 10): %s",
                self._unexpected_keys[:10],
            )

    def _freeze_body(self) -> None:
        # Train: patch_embed.proj (kernel changed), pos_embed (token count
        # changed), and any fpn ops that carry learnable params (k=1 uses
        # ConvTranspose / MaxPool; k=3 uses Identity, so fpn* contribute
        # no params and the requires_grad flip is a no-op there).
        trainable_prefixes = ("patch_embed.proj.", "pos_embed", "fpn1.", "fpn2.", "fpn3.", "fpn4.")
        for name, p in self.model.named_parameters():
            if any(name.startswith(prefix) or name == prefix for prefix in trainable_prefixes):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    @property
    def trainable_parameter_names(self) -> List[str]:
        return [n for n, p in self.model.named_parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Pad to ``pad_to`` and return the four FPN feature stages.

        Args:
            x: ``[B, in_chans, 11, 11]`` (or whatever ``pad_to - 1``).

        Returns:
            list of four ``[B, embed_dim, Hp, Wp]`` feature maps. With
            ``pad_to=12`` / ``new_patch_size=3`` each is ``[B, 768, 4, 4]``.
        """
        # forward_features returns [original_img, feat_a, feat_b, feat_c, feat_d]
        pad_right = self.pad_to - x.shape[-1]
        pad_bottom = self.pad_to - x.shape[-2]
        if pad_right or pad_bottom:
            x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="reflect")
        feats = self.model.forward_features(x, self.new_patch_size)
        return list(feats[1:])

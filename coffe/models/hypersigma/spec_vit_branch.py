"""SpecViT-B (``SpectralVisionTransformer``) wrapper for small HSI patches.

The wrapper:

* constructs upstream's ``SpectralVisionTransformer`` with
  ``img_size=11``, ``in_chans=144`` (Houston raw bands, no spectral
  PCA), ``NUM_TOKENS=100``, ``use_abs_pos_emb=True``;
* loads the SpecViT-B checkpoint, dropping the keys that don't match the
  small-patch deployment (``spat_map`` because ``img_size`` differs from
  HyperGlobal's 64; ``conv_q/k/v`` and ``l1`` because they were trained
  for downstream tasks);
* handles ``pos_embed`` with a **load-then-interp-then-reinit** decision
  tree (see ``_load_pos_embed``);
* freezes the transformer body when ``freeze_body=True``, leaving
  ``spat_map``, ``pos_embed``, and (intentionally) ``l1`` trainable —
  ``l1`` is applied to ``features[0]`` inside the upstream
  ``forward_features`` and we need to keep it consistent with the
  matrix dimensions the SEM consumes.

NOTE: ``SpectralVisionTransformer`` does **not** use a CLS token.
``pos_embed.shape == (1, NUM_TOKENS, embed_dim)``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.HyperSIGMA.ImageClassification.model import SpecViT_fusion

from ._input_fit import fit_input

logger = logging.getLogger(__name__)


# Keys we deliberately drop from the released SpecViT-B checkpoint
# before loading. Anything outside this set that ends up missing or
# unexpected is logged.
_DROPPED_PREFIXES = (
    "spat_map.",  # img_size dependent; re-init
    "pos_embed",  # token count dependent; handled explicitly
    "conv_q.",
    "conv_k.",
    "conv_v.",  # downstream auxiliary layers
    "l1.",  # downstream auxiliary 768 -> 128 projection
    "cls",  # any CLS tokens (none expected, defensive)
    "classifier.",  # downstream classifier head
)


# Native-geometry ablation: build SpecViT at its pretrained img_size=64 so
# ``spat_map`` (Linear(4096, 768)) and ``pos_embed`` LOAD from the released
# MAE checkpoint (no reinit). Keep those two; drop only the genuine
# downstream-aux layers and the MAE-decoder / mask token extras.
_NATIVE_DROPPED_PREFIXES = (
    "conv_q.",
    "conv_k.",
    "conv_v.",
    "l1.",
    "cls",
    "classifier.",
    "decoder_blocks.",
    "decoder_embed.",
    "decoder_norm.",
    "decoder_pred.",
    "decoder_pos_embed",
    "mask_token",
)


def _looks_like_dropped_key(key: str, drop_keys: tuple[str, ...] = _DROPPED_PREFIXES) -> bool:
    return any(key.startswith(p) or key == p for p in drop_keys)


def _strip_state_dict_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first = next(iter(state_dict))
    if first.startswith("module."):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    if all(k.startswith("encoder.") for k in state_dict):
        state_dict = {k[len("encoder.") :]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(ckpt: dict) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("state_dict", "model", "encoder"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


class SpecViTBranch(nn.Module):
    """HyperSIGMA SpecViT branch adapted for 11x11 / 144-band patches."""

    def __init__(
        self,
        pretrained_path: str | None,
        in_chans: int = 144,
        img_size: int = 11,
        num_tokens: int = 100,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        out_indices: tuple[int, ...] = (3, 5, 7, 11),
        freeze_body: bool = True,
        log_dropped_keys: bool = True,
        native_geometry: bool = False,
        native_img_size: int = 64,
        input_fit: str = "upscale",
        pad_anchor: str = "center",
        interp_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.out_indices = tuple(out_indices)

        # Native geometry: keep the encoder at its pretrained img_size=64 so
        # spat_map (Linear(4096, 768)) + pos_embed load from the checkpoint,
        # and resize the 11x11 input up to 64x64 instead.
        self.native_geometry = native_geometry
        self.input_fit = input_fit
        self.pad_anchor = pad_anchor
        self.interp_mode = interp_mode
        eff_img = native_img_size if native_geometry else img_size
        self.img_size = eff_img

        self.model = SpecViT_fusion.SpectralVisionTransformer(
            NUM_TOKENS=num_tokens,
            img_size=eff_img,
            in_chans=in_chans,
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

        # Sanity check.
        assert self.model.spat_map.weight.shape == (embed_dim, eff_img * eff_img), (
            f"unexpected spat_map shape {tuple(self.model.spat_map.weight.shape)}"
        )
        assert self.model.pos_embed is not None and self.model.pos_embed.shape == (
            1,
            num_tokens,
            embed_dim,
        ), f"unexpected pos_embed shape {tuple(self.model.pos_embed.shape)}"

        self._dropped_keys: list[str] = []
        self._unexpected_keys: list[str] = []
        self._missing_keys: list[str] = []
        self.pos_embed_source: str = "reinit"
        self.spat_map_source: str = "reinit"

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

        # Inspect pos_embed BEFORE we drop it.
        ckpt_pos_embed = state_dict.get("pos_embed")

        drop_keys = _NATIVE_DROPPED_PREFIXES if self.native_geometry else _DROPPED_PREFIXES
        kept: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if _looks_like_dropped_key(k, drop_keys):
                self._dropped_keys.append(k)
                continue
            kept[k] = v

        missing, unexpected = self.model.load_state_dict(kept, strict=False)
        self._missing_keys = [k for k in missing if not _looks_like_dropped_key(k, drop_keys)]
        self._unexpected_keys = list(unexpected)

        if self.native_geometry:
            # Native geometry matches the checkpoint: spat_map + pos_embed are
            # kept above and loaded by load_state_dict — do NOT reinit them.
            sm_bad = any("spat_map" in k for k in (self._missing_keys + self._unexpected_keys))
            self.spat_map_source = "reinit" if sm_bad else "loaded"
            self._load_pos_embed(ckpt_pos_embed)
            self._assert_native_loaded()
        else:
            # Re-init spat_map (img_size-dependent linear projection).
            nn.init.trunc_normal_(self.model.spat_map.weight, std=0.02)
            if self.model.spat_map.bias is not None:
                nn.init.zeros_(self.model.spat_map.bias)
            self.spat_map_source = "reinit"
            # pos_embed: load-then-interp-then-reinit.
            self._load_pos_embed(ckpt_pos_embed)

    def _assert_native_loaded(self) -> None:
        """Fail loudly if the native geometry did not actually load the
        pretrained spat_map / position embedding."""
        if self.pos_embed_source != "loaded":
            raise RuntimeError(
                "[HyperSIGMA][native] SpecViT pos_embed did not load cleanly "
                f"(source={self.pos_embed_source!r}); num_tokens likely does not "
                f"match the checkpoint (pos_embed shape {tuple(self.model.pos_embed.shape)})."
            )
        if self.spat_map_source != "loaded":
            raise RuntimeError(
                "[HyperSIGMA][native] SpecViT spat_map did not load cleanly; "
                "native_img_size must satisfy native_img_size**2 == checkpoint "
                "spat_map in_features (4096 -> img_size 64) "
                f"(got img_size={self.img_size}, spat_map.weight shape "
                f"{tuple(self.model.spat_map.weight.shape)})."
            )

    def _load_pos_embed(self, ckpt_pos_embed: torch.Tensor | None) -> None:
        target_shape = (1, self.num_tokens, self.embed_dim)
        if ckpt_pos_embed is None:
            nn.init.trunc_normal_(self.model.pos_embed, std=0.02)
            self.pos_embed_source = "reinit"
            return

        if ckpt_pos_embed.shape == target_shape:
            with torch.no_grad():
                self.model.pos_embed.copy_(ckpt_pos_embed)
            self.pos_embed_source = "loaded"
            return

        # Some upstream checkpoints include a CLS token even if the
        # downstream model doesn't use one. Try stripping it.
        if (
            ckpt_pos_embed.dim() == 3
            and ckpt_pos_embed.shape[0] == 1
            and ckpt_pos_embed.shape[2] == self.embed_dim
            and ckpt_pos_embed.shape[1] == self.num_tokens + 1
        ):
            with torch.no_grad():
                self.model.pos_embed.copy_(ckpt_pos_embed[:, 1:, :])
            self.pos_embed_source = "loaded_strip_cls"
            return

        if (
            ckpt_pos_embed.dim() == 3
            and ckpt_pos_embed.shape[0] == 1
            and ckpt_pos_embed.shape[2] == self.embed_dim
        ):
            # 1-D bicubic interpolate the token axis: treat as a 1xT
            # sequence and resample to 1xnum_tokens.
            T = ckpt_pos_embed.shape[1]
            x = ckpt_pos_embed.permute(0, 2, 1).unsqueeze(-1)  # [1, embed, T, 1]
            x = F.interpolate(x, size=(self.num_tokens, 1), mode="bicubic", align_corners=False)
            x = x.squeeze(-1).permute(0, 2, 1)  # [1, num_tokens, embed]
            with torch.no_grad():
                self.model.pos_embed.copy_(x)
            self.pos_embed_source = f"interpolated_{T}->{self.num_tokens}"
            return

        # Last resort.
        nn.init.trunc_normal_(self.model.pos_embed, std=0.02)
        self.pos_embed_source = "reinit"

    def _log_load_summary(self) -> None:
        logger.info(
            "[HyperSIGMA] SpecViT load summary: "
            "dropped=%d, missing=%d, unexpected=%d, pos_embed=%s",
            len(self._dropped_keys),
            len(self._missing_keys),
            len(self._unexpected_keys),
            self.pos_embed_source,
        )
        if self._missing_keys:
            logger.warning(
                "[HyperSIGMA] SpecViT missing keys (first 10): %s",
                self._missing_keys[:10],
            )
        if self._unexpected_keys:
            logger.warning(
                "[HyperSIGMA] SpecViT unexpected keys (first 10): %s",
                self._unexpected_keys[:10],
            )

    def _freeze_body(self) -> None:
        trainable_prefixes = ("spat_map.", "pos_embed", "l1.")
        for name, p in self.model.named_parameters():
            if any(name.startswith(prefix) or name == prefix for prefix in trainable_prefixes):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    @property
    def trainable_parameter_names(self) -> list[str]:
        """Names of the SpecViT parameters left unfrozen by this wrapper."""
        return [n for n, p in self.model.named_parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the full features list from the upstream encoder.

        ``forward_features`` returns ``[features_layer3, features_layer5,
        features_layer7, features_layer11]``. ``features[0]`` has had the
        ``l1`` projection applied (768 -> 128); the rest are the raw
        post-block embeddings of shape ``[B, num_tokens, embed_dim]``.
        """
        if self.native_geometry:
            # Resize the small patch up to the native input size (64x64) so
            # spat_map's 4096 in-features match.
            x = fit_input(
                x,
                self.img_size,
                self.input_fit,
                interp_mode=self.interp_mode,
                pad_anchor=self.pad_anchor,
            )
        return list(self.model.forward_features(x))

"""Spatial input-fit helpers for the native-geometry HyperSIGMA ablation.

The native-geometry ablation keeps each encoder at its pretrained input
size (SpatViT 64x64/patch-8, SpecViT 64x64) and transforms the small
11x11 patch *up to* that size, instead of shrinking the encoder. Two
strategies:

* ``"upscale"`` — bicubic interpolation 11x11 -> size x size,
* ``"pad"``     — constant (zero) padding, centered, 11x11 -> size x size.

Reflection padding is intentionally unsupported here: ``F.pad`` reflect
requires every pad amount < the input dim, but 11 -> 64 needs pads far
larger than 10, so reflect is impossible at these ratios.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_input(
    x: torch.Tensor,
    size: int,
    input_fit: str = "upscale",
    interp_mode: str = "bicubic",
    pad_anchor: str = "center",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Resize/pad ``[B, C, H, W]`` to ``[B, C, size, size]``.

    Args:
        x: input tensor ``[B, C, H, W]``.
        size: target spatial size (square).
        input_fit: ``"upscale"`` (interpolate) or ``"pad"`` (zero-pad).
        interp_mode: interpolation mode for ``"upscale"`` (default bicubic).
        pad_anchor: ``"center"`` (default) or ``"top-left"`` for ``"pad"``.
        pad_value: constant fill value for ``"pad"`` (default 0).
    """
    h, w = x.shape[-2], x.shape[-1]

    if input_fit == "upscale":
        if (h, w) == (size, size):
            return x
        return F.interpolate(
            x, size=(size, size), mode=interp_mode, align_corners=False
        )

    if input_fit == "pad":
        pad_h = size - h
        pad_w = size - w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"fit_input(pad): target size {size} smaller than input {(h, w)}"
            )
        if pad_anchor == "center":
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
        elif pad_anchor == "top-left":
            top, left = 0, 0
            bottom, right = pad_h, pad_w
        else:
            raise ValueError(f"unknown pad_anchor={pad_anchor!r}")
        return F.pad(x, (left, right, top, bottom), mode="constant", value=pad_value)

    raise ValueError(f"unknown input_fit={input_fit!r} (expected 'upscale' or 'pad')")

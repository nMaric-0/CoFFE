"""Checkpoint utilities for flexible key loading."""
from __future__ import annotations

from typing import Dict, Tuple, Optional
import torch


def load_checkpoint_state(
    checkpoint_path: str,
    device: str = "cpu"
) -> Tuple[Dict, Optional[Dict]]:
    """Load a checkpoint and return the state dict plus optional config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = ckpt.get("config")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        config = ckpt.get("config")
    elif "encoder_state_dict" in ckpt:
        state_dict = ckpt["encoder_state_dict"]
        config = ckpt.get("config")
    else:
        state_dict = ckpt
        config = None

    return state_dict, config


def fix_state_dict_keys(state_dict: Dict, model_state: Dict) -> Dict:
    """Fix key mismatches between checkpoint and model."""
    fixed_state = {}

    skip_prefixes = (
        "spatial_masking", "spectral_masking", "lidar_masking",
        "spatial_decoder", "spectral_decoder", "lidar_decoder", "denoise_decoder",
        "enc_to_dec", "mask_token", "null_lidar", "noise_augmentation",
        "decoder", "contrastive_head",
    )

    model_has_encoder_encoder = any(
        key.startswith("encoder.encoder.") for key in model_state.keys()
    )

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        if any(new_key.startswith(prefix) for prefix in skip_prefixes):
            continue

        if new_key.startswith("encoder.encoder."):
            new_key = new_key[8:]
        elif new_key.startswith("encoder.") and not model_has_encoder_encoder:
            if new_key not in model_state and new_key[8:] in model_state:
                new_key = new_key[8:]

        if new_key.startswith("layers.") and "encoder.layers.0.norm1.weight" in model_state:
            new_key = f"encoder.{new_key}"

        fixed_state[new_key] = value

    return fixed_state


def filter_compatible_state(
    state_dict: Dict,
    model_state: Dict,
    channel_dependent_keys: Optional[set] = None
) -> Tuple[Dict, list]:
    """Filter state dict to only include matching shapes."""
    compatible_state = {}
    size_mismatches = []

    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if value.shape == model_state[key].shape:
            compatible_state[key] = value
        else:
            size_mismatches.append((key, value.shape, model_state[key].shape))
            if channel_dependent_keys and key in channel_dependent_keys:
                continue

    return compatible_state, size_mismatches

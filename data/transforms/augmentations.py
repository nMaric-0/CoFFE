"""Data augmentation for multimodal EO data."""
import torch
import numpy as np
from typing import Dict, List, Callable


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomFlip:
    """Random horizontal and vertical flip."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if np.random.random() < self.p:
            sample["hsi"] = torch.flip(sample["hsi"], dims=[-1])
            sample["aux"] = torch.flip(sample["aux"], dims=[-1])
        
        if np.random.random() < self.p:
            sample["hsi"] = torch.flip(sample["hsi"], dims=[-2])
            sample["aux"] = torch.flip(sample["aux"], dims=[-2])
        
        return sample


class RandomRotation:
    """Random 90-degree rotation."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            sample["hsi"] = torch.rot90(sample["hsi"], k, dims=[-2, -1])
            sample["aux"] = torch.rot90(sample["aux"], k, dims=[-2, -1])
        
        return sample


class SpectralJitter:
    """Add spectral noise to HSI."""
    
    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noise = torch.randn_like(sample["hsi"]) * self.sigma
        sample["hsi"] = sample["hsi"] + noise
        return sample


class GaussianNoise:
    """Add Gaussian noise to both modalities."""

    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample["hsi"] = sample["hsi"] + torch.randn_like(sample["hsi"]) * self.sigma
        sample["aux"] = sample["aux"] + torch.randn_like(sample["aux"]) * self.sigma
        return sample


class RandomBandWindow:
    """
    Select a random contiguous window of spectral bands.

    Following original HyperSIGMA: randomly select a contiguous window of
    `window_size` bands from the full spectrum for each sample.
    If the HSI has fewer bands than window_size, all bands are kept.

    Args:
        window_size: Number of contiguous bands to keep (default: 100)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hsi = sample["hsi"]  # [C, H, W]
        C = hsi.shape[0]

        if C <= self.window_size:
            return sample

        # Random start position
        start = np.random.randint(0, C - self.window_size + 1)
        sample["hsi"] = hsi[start:start + self.window_size]
        return sample


class RandomResizedCropHSI:
    """
    Random resized crop for HSI patches, following torchvision.

    Randomly crops a region with scale in [scale_min, scale_max] of the
    original area, then resizes to target_size.

    Args:
        target_size: Output spatial size
        scale: Tuple of (min_scale, max_scale) for area fraction
    """

    def __init__(self, target_size: int = 11, scale: tuple = (0.5, 1.0)):
        self.target_size = target_size
        self.scale = scale

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hsi = sample["hsi"]  # [C, H, W]
        aux = sample["aux"]  # [C_aux, H, W]

        C, H, W = hsi.shape

        # Random scale
        area = H * W
        target_area = np.random.uniform(self.scale[0], self.scale[1]) * area

        # Compute crop size (square)
        crop_size = int(np.sqrt(target_area))
        crop_size = max(1, min(crop_size, min(H, W)))

        # Random position
        top = np.random.randint(0, H - crop_size + 1)
        left = np.random.randint(0, W - crop_size + 1)

        # Crop
        hsi_crop = hsi[:, top:top+crop_size, left:left+crop_size]
        aux_crop = aux[:, top:top+crop_size, left:left+crop_size]

        # Resize to target
        if crop_size != self.target_size:
            hsi_crop = torch.nn.functional.interpolate(
                hsi_crop.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False
            ).squeeze(0)
            aux_crop = torch.nn.functional.interpolate(
                aux_crop.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False
            ).squeeze(0)

        sample["hsi"] = hsi_crop
        sample["aux"] = aux_crop
        return sample

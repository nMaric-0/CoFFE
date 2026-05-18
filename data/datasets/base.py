"""Base dataset class for multimodal Earth observation data."""
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import scipy.io as sio


class MultimodalEODataset(ABC, Dataset):
    """
    Abstract base class for multimodal Earth observation datasets.
    
    Handles HSI + auxiliary modality (LiDAR/SAR/DSM) data loading.
    """
    
    def __init__(
        self,
        data_root: str,
        patch_size: int = 11,
        normalize: bool = True,
        train: bool = True
    ):
        self.data_root = data_root
        self.patch_size = patch_size
        self.normalize = normalize
        self.train = train
        
        # Load data
        self.hsi, self.aux, self.labels = self._load_data()
        
        # Normalize
        if self.normalize:
            self.hsi = self._normalize(self.hsi)
            self.aux = self._normalize(self.aux)
        
        # Build index
        self.class_indices = self._build_class_indices()
        
    @abstractmethod
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load HSI, auxiliary data, and labels. Must be implemented by subclass."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        pass
    
    @property
    @abstractmethod
    def hsi_channels(self) -> int:
        """Number of HSI spectral bands."""
        pass
    
    @property
    @abstractmethod
    def aux_channels(self) -> int:
        """Number of auxiliary data channels."""
        pass
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalization per channel."""
        data = data.astype(np.float32)
        for i in range(data.shape[-1]):
            channel = data[..., i]
            min_val, max_val = channel.min(), channel.max()
            if max_val - min_val > 0:
                data[..., i] = (channel - min_val) / (max_val - min_val)
        return data
    
    def _build_class_indices(self) -> Dict[int, list]:
        """Build index of valid pixel coordinates per class."""
        pad = self.patch_size // 2
        H, W = self.labels.shape
        
        class_indices = {}
        unique_classes = np.unique(self.labels)
        unique_classes = unique_classes[unique_classes > 0]  # Exclude background
        
        for c in unique_classes:
            ys, xs = np.where(self.labels == c)
            # Filter edge pixels
            valid = (
                (ys >= pad) & (ys < H - pad) &
                (xs >= pad) & (xs < W - pad)
            )
            class_indices[int(c)] = list(zip(ys[valid], xs[valid]))
        
        return class_indices
    
    def extract_patch(self, y: int, x: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract HSI and auxiliary patches centered at (y, x)."""
        pad = self.patch_size // 2
        
        hsi_patch = self.hsi[y-pad:y+pad+1, x-pad:x+pad+1, :]
        aux_patch = self.aux[y-pad:y+pad+1, x-pad:x+pad+1, :]
        
        # Convert to [C, H, W] format
        hsi_tensor = torch.from_numpy(hsi_patch).permute(2, 0, 1).float()
        aux_tensor = torch.from_numpy(aux_patch).permute(2, 0, 1).float()
        
        return hsi_tensor, aux_tensor
    
    def __len__(self) -> int:
        return sum(len(indices) for indices in self.class_indices.values())
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Flatten all indices
        all_indices = []
        all_labels = []
        for label, indices in self.class_indices.items():
            all_indices.extend(indices)
            all_labels.extend([label] * len(indices))
        
        y, x = all_indices[idx]
        label = all_labels[idx]
        
        hsi, aux = self.extract_patch(y, x)
        
        return {
            "hsi": hsi,
            "aux": aux,
            "label": torch.tensor(label, dtype=torch.long),
            "coords": torch.tensor([y, x], dtype=torch.long)
        }

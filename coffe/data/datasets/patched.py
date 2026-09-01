"""Base dataset class for pre-patched MFT format data."""
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import scipy.io as sio
import os


class PatchedMultimodalDataset(ABC, Dataset):
    """
    Base class for pre-patched multimodal Earth observation datasets.
    
    This is for data in MFT repository format where patches are already extracted:
    - HSI_Tr.mat, HSI_Te.mat: Pre-extracted HSI patches [N, H, W, C] or [N, C, H, W]
    - LiDAR_Tr.mat, LiDAR_Te.mat: Pre-extracted LiDAR patches
    - TrLabel.mat, TeLabel.mat: Labels for each patch [N] or [N, 1]
    """
    
    def __init__(
        self,
        data_root: str,
        patch_size: int = 11,
        split: str = "train",  # "train", "test", or "all"
        normalize: bool = True
    ):
        """
        Args:
            data_root: Root directory containing dataset folder
            patch_size: Expected patch size (for validation, should be 11)
            split: Which split to use - "train", "test", or "all"
            normalize: Whether to normalize data
        """
        self.data_root = data_root
        self.patch_size = patch_size
        self.split = split
        self.normalize = normalize
        
        # Load data
        self.hsi, self.aux, self.labels = self._load_data()
        
        # Normalize if requested
        if self.normalize:
            self.hsi = self._normalize(self.hsi)
            self.aux = self._normalize(self.aux)
        
        # Convert to torch tensors
        self.hsi = torch.from_numpy(self.hsi).float()
        self.aux = torch.from_numpy(self.aux).float()
        self.labels = torch.from_numpy(self.labels).long()
        
        # Build class indices for episodic sampling
        self.class_indices = self._build_class_indices()
    
    @property
    @abstractmethod
    def folder_name(self) -> str:
        """Name of the dataset folder (e.g., 'Houston11x11')."""
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
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load pre-patched data from .mat files."""
        base_path = os.path.join(self.data_root, self.folder_name)
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f"Dataset folder not found: {base_path}\n"
                f"Expected structure: {self.data_root}/{self.folder_name}/"
            )
        
        if self.split == "train":
            return self._load_split(base_path, "Tr")
        elif self.split == "test":
            return self._load_split(base_path, "Te")
        else:  # "all" - combine train and test
            hsi_tr, aux_tr, labels_tr = self._load_split(base_path, "Tr")
            hsi_te, aux_te, labels_te = self._load_split(base_path, "Te")
            
            hsi = np.concatenate([hsi_tr, hsi_te], axis=0)
            aux = np.concatenate([aux_tr, aux_te], axis=0)
            labels = np.concatenate([labels_tr, labels_te], axis=0)
            
            return hsi, aux, labels
    
    def _load_split(
        self,
        base_path: str,
        suffix: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single split (train or test)."""
        # Load HSI
        hsi_path = os.path.join(base_path, f"HSI_{suffix}.mat")
        hsi_data = sio.loadmat(hsi_path)
        hsi = self._extract_array(hsi_data).astype(np.float32)
        
        # Load LiDAR
        lidar_path = os.path.join(base_path, f"LIDAR_{suffix}.mat")
        lidar_data = sio.loadmat(lidar_path)
        aux = self._extract_array(lidar_data).astype(np.float32)
        
        # Load labels
        label_path = os.path.join(base_path, f"{suffix}Label.mat")
        label_data = sio.loadmat(label_path)
        labels = self._extract_array(label_data).astype(np.int64)
        
        # Ensure correct shapes
        hsi = self._ensure_nchw(hsi)
        aux = self._ensure_nchw(aux)
        labels = self._ensure_1d(labels)
        
        return hsi, aux, labels
    
    def _extract_array(self, mat_dict: dict) -> np.ndarray:
        """Extract the data array from a .mat file dictionary."""
        for key, value in mat_dict.items():
            if not key.startswith('__') and isinstance(value, np.ndarray):
                return value
        raise ValueError(f"No data array found. Keys: {list(mat_dict.keys())}")
    
    def _ensure_nchw(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is in [N, C, H, W] format."""
        if data.ndim == 3:
            # [N, H, W] -> [N, 1, H, W]
            return data[:, np.newaxis, :, :]
        elif data.ndim == 4:
            # Check if [N, H, W, C] or [N, C, H, W]
            # Assume H == W == patch_size
            if data.shape[2] == data.shape[3] == self.patch_size:
                # Already [N, C, H, W]
                return data
            elif data.shape[1] == data.shape[2] == self.patch_size:
                # [N, H, W, C] -> [N, C, H, W]
                return np.transpose(data, (0, 3, 1, 2))
            else:
                # Try to infer from channel count
                if data.shape[1] <= 5:  # Small number, likely channels
                    return data
                else:
                    return np.transpose(data, (0, 3, 1, 2))
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
    
    def _ensure_1d(self, labels: np.ndarray) -> np.ndarray:
        """Ensure labels are 1D array."""
        return labels.squeeze()
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data per channel across all samples."""
        # data: [N, C, H, W]
        data = data.copy()
        N, C, H, W = data.shape

        for c in range(C):
            channel_data = data[:, c, :, :]
            min_val = channel_data.min()
            max_val = channel_data.max()
            if max_val - min_val > 1e-8:
                data[:, c, :, :] = (channel_data - min_val) / (max_val - min_val)

        return data
    
    def _build_class_indices(self) -> Dict[int, list]:
        """Build index mapping class -> list of sample indices."""
        class_indices = {}
        unique_classes = torch.unique(self.labels)
        
        for c in unique_classes:
            indices = torch.where(self.labels == c)[0].tolist()
            class_indices[c.item()] = indices
        
        return class_indices
    
    def extract_patch(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get HSI and aux patches by sample index."""
        return self.hsi[idx], self.aux[idx]
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "hsi": self.hsi[idx],
            "aux": self.aux[idx],
            "label": self.labels[idx],
            "index": torch.tensor(idx, dtype=torch.long)
        }


class HoustonPatchedDataset(PatchedMultimodalDataset):
    """
    Houston 2013 dataset in pre-patched MFT format.
    
    Expected structure:
    data_root/Houston11x11/
        HSI_Tr.mat, HSI_Te.mat
        LiDAR_Tr.mat, LiDAR_Te.mat
        TrLabel.mat, TeLabel.mat
    """
    
    @property
    def folder_name(self) -> str:
        return "Houston11x11"
    
    @property
    def num_classes(self) -> int:
        return 15
    
    @property
    def hsi_channels(self) -> int:
        return 144
    
    @property
    def aux_channels(self) -> int:
        return 1


class TrentoPatchedDataset(PatchedMultimodalDataset):
    """
    Trento dataset in pre-patched MFT format.
    
    Expected structure:
    data_root/Trento11x11/
        HSI_Tr.mat, HSI_Te.mat
        LiDAR_Tr.mat, LiDAR_Te.mat
        TrLabel.mat, TeLabel.mat
    """
    
    @property
    def folder_name(self) -> str:
        return "Trento11x11"
    
    @property
    def num_classes(self) -> int:
        return 6
    
    @property
    def hsi_channels(self) -> int:
        return 63
    
    @property
    def aux_channels(self) -> int:
        return 1


class MUUFLPatchedDataset(PatchedMultimodalDataset):
    """
    MUUFL Gulfport dataset in pre-patched MFT format.
    
    Expected structure:
    data_root/MUUFL11x11/
        HSI_Tr.mat, HSI_Te.mat
        LiDAR_Tr.mat, LiDAR_Te.mat
        TrLabel.mat, TeLabel.mat
    """
    
    @property
    def folder_name(self) -> str:
        return "MUUFL11x11"
    
    @property
    def num_classes(self) -> int:
        return 11
    
    @property
    def hsi_channels(self) -> int:
        return 64
    
    @property
    def aux_channels(self) -> int:
        return 2  # MUUFL has 2 LiDAR returns

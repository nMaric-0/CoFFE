"""MUUFL Gulfport dataset."""
import os
import numpy as np
import scipy.io as sio
from .base import MultimodalEODataset


class MUUFLDataset(MultimodalEODataset):
    """
    MUUFL Gulfport dataset.
    
    HSI: 64 bands (72 original, 8 removed due to noise)
    LiDAR: 2 bands (elevation rasters)
    Classes: 11
    Size: 325 × 220 pixels
    """
    
    @property
    def num_classes(self) -> int:
        return 11
    
    @property
    def hsi_channels(self) -> int:
        return 64
    
    @property
    def aux_channels(self) -> int:
        return 2
    
    def _load_data(self):
        """Load MUUFL HSI and LiDAR data."""
        mft_path = os.path.join(self.data_root, "MUUFL11x11")
        if os.path.exists(mft_path):
            return self._load_mft_format(mft_path)
        
        # Standard format loading
        data_path = os.path.join(self.data_root, "muufl")
        
        hsi = sio.loadmat(os.path.join(data_path, "HSI.mat"))
        lidar = sio.loadmat(os.path.join(data_path, "LIDAR.mat"))
        gt = sio.loadmat(os.path.join(data_path, "GT.mat"))
        
        hsi = self._get_array(hsi)
        lidar = self._get_array(lidar)
        labels = self._get_array(gt)
        
        return hsi, lidar, labels
    
    def _load_mft_format(self, path: str):
        """Load MFT format data."""
        data = sio.loadmat(os.path.join(path, "MUUFL.mat"))
        hsi = self._get_array(data, 'hsi')
        lidar = self._get_array(data, 'lidar')
        labels = self._get_array(data, 'gt')
        return hsi, lidar, labels
    
    def _get_array(self, mat_dict, hint=None):
        """Extract array from mat file."""
        if hint and hint in mat_dict:
            return mat_dict[hint]
        for key, value in mat_dict.items():
            if not key.startswith('__') and isinstance(value, np.ndarray):
                return value
        raise ValueError("No array found in mat file")

"""Houston 2013 dataset."""

import os

import numpy as np
import scipy.io as sio

from .base import MultimodalEODataset


class HoustonDataset(MultimodalEODataset):
    """
    Houston 2013 dataset (IEEE GRSS Data Fusion Contest).

    HSI: 144 bands (0.38-1.05 μm)
    LiDAR: 1 band (elevation)
    Classes: 15
    Size: 349 × 1905 pixels
    GSD: 2.5m
    """

    @property
    def num_classes(self) -> int:
        return 15

    @property
    def hsi_channels(self) -> int:
        return 144

    @property
    def aux_channels(self) -> int:
        return 1

    def _load_data(self):
        """Load Houston HSI and LiDAR data."""
        # Expected file structure:
        # data_root/houston/Houston.mat or similar

        hsi_path = os.path.join(self.data_root, "houston", "Houston.mat")
        lidar_path = os.path.join(self.data_root, "houston", "LIDAR.mat")
        gt_path = os.path.join(self.data_root, "houston", "Houston_gt.mat")

        # Try alternative structures
        if not os.path.exists(hsi_path):
            # Try Houston11x11 format from MFT
            alt_path = os.path.join(self.data_root, "Houston11x11")
            if os.path.exists(alt_path):
                return self._load_mft_format(alt_path)

        # Load standard format
        hsi_data = sio.loadmat(hsi_path)
        lidar_data = sio.loadmat(lidar_path)
        gt_data = sio.loadmat(gt_path)

        # Extract arrays (key names may vary)
        hsi = self._extract_array(hsi_data, ["HSI", "hsi", "data", "X"])
        lidar = self._extract_array(lidar_data, ["LiDAR", "lidar", "DSM", "data"])
        labels = self._extract_array(gt_data, ["gt", "GT", "labels", "Y"])

        # Ensure LiDAR has channel dimension
        if lidar.ndim == 2:
            lidar = lidar[:, :, np.newaxis]

        return hsi, lidar, labels

    def _load_mft_format(self, path: str):
        """Load data in MFT repository format."""
        data = sio.loadmat(os.path.join(path, "Houston.mat"))

        # MFT uses patches, we need to reconstruct or use directly
        # This is a placeholder - adapt based on actual MFT data format
        hsi = data.get("hsi", data.get("HSI"))
        lidar = data.get("lidar", data.get("LiDAR"))
        labels = data.get("gt", data.get("Y"))

        if lidar.ndim == 2:
            lidar = lidar[:, :, np.newaxis]

        return hsi, lidar, labels

    def _extract_array(self, mat_dict, possible_keys):
        """Extract array from .mat file trying multiple possible keys."""
        for key in possible_keys:
            if key in mat_dict:
                return mat_dict[key]
        # Return first non-metadata array
        for key, value in mat_dict.items():
            if not key.startswith("__") and isinstance(value, np.ndarray):
                return value
        raise KeyError(f"Could not find data array. Keys: {list(mat_dict.keys())}")

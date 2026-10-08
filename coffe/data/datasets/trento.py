"""Trento dataset."""

import os

import numpy as np
import scipy.io as sio

from .base import MultimodalEODataset


class TrentoDataset(MultimodalEODataset):
    """
    Trento dataset.

    HSI: 63 bands (0.42-0.99 μm)
    LiDAR: 1 band (elevation)
    Classes: 6
    Size: 600 × 166 pixels
    GSD: 1m
    """

    @property
    def num_classes(self) -> int:
        return 6

    @property
    def hsi_channels(self) -> int:
        return 63

    @property
    def aux_channels(self) -> int:
        return 1

    def _load_data(self):
        """Load Trento HSI and LiDAR data."""
        # Similar structure to Houston
        data_path = os.path.join(self.data_root, "trento")

        # Try MFT format first
        mft_path = os.path.join(self.data_root, "Trento11x11")
        if os.path.exists(mft_path):
            return self._load_mft_format(mft_path)

        # Standard format
        hsi_path = os.path.join(data_path, "HSI.mat")
        lidar_path = os.path.join(data_path, "LiDAR.mat")
        gt_path = os.path.join(data_path, "GT.mat")

        hsi = sio.loadmat(hsi_path)
        lidar = sio.loadmat(lidar_path)
        gt = sio.loadmat(gt_path)

        hsi = self._get_array(hsi)
        lidar = self._get_array(lidar)
        labels = self._get_array(gt)

        if lidar.ndim == 2:
            lidar = lidar[:, :, np.newaxis]

        return hsi, lidar, labels

    def _load_mft_format(self, path: str):
        """Load MFT format data."""
        data = sio.loadmat(os.path.join(path, "Trento.mat"))
        hsi = self._get_array(data, "hsi")
        lidar = self._get_array(data, "lidar")
        labels = self._get_array(data, "gt")

        if lidar.ndim == 2:
            lidar = lidar[:, :, np.newaxis]

        return hsi, lidar, labels

    def _get_array(self, mat_dict, hint=None):
        """Extract array from mat file."""
        if hint and hint in mat_dict:
            return mat_dict[hint]
        for key, value in mat_dict.items():
            if not key.startswith("__") and isinstance(value, np.ndarray):
                return value
        raise ValueError("No array found in mat file")

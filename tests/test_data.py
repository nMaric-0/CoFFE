"""Tests for data loading."""
import pytest
import torch
import numpy as np
from coffe.data.samplers import EpisodeSampler


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self):
        self.hsi = np.random.randn(100, 100, 64).astype(np.float32)
        self.aux = np.random.randn(100, 100, 1).astype(np.float32)
        self.labels = np.zeros((100, 100), dtype=np.int64)
        
        # Create mock classes
        for i in range(5):
            self.labels[i*20:(i+1)*20, :20] = i + 1
        
        self.patch_size = 11
        self.class_indices = self._build_indices()
    
    def _build_indices(self):
        pad = self.patch_size // 2
        indices = {}
        for c in range(1, 6):
            ys, xs = np.where(self.labels == c)
            valid = (ys >= pad) & (ys < 100-pad) & (xs >= pad) & (xs < 100-pad)
            indices[c] = list(zip(ys[valid], xs[valid]))
        return indices
    
    def extract_patch(self, y, x):
        pad = self.patch_size // 2
        hsi = self.hsi[y-pad:y+pad+1, x-pad:x+pad+1, :]
        aux = self.aux[y-pad:y+pad+1, x-pad:x+pad+1, :]
        return (
            torch.from_numpy(hsi).permute(2, 0, 1),
            torch.from_numpy(aux).permute(2, 0, 1)
        )


class TestEpisodeSampler:
    def test_sample_episode(self):
        dataset = MockDataset()
        sampler = EpisodeSampler(
            dataset, n_way=3, k_shot=5, k_query=10, num_episodes=10
        )
        
        episode = sampler.sample_episode()
        
        assert episode["support_hsi"].shape == (15, 64, 11, 11)
        assert episode["support_aux"].shape == (15, 1, 11, 11)
        assert episode["support_labels"].shape == (15,)
        assert episode["query_hsi"].shape == (30, 64, 11, 11)
        assert episode["query_labels"].shape == (30,)

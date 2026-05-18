"""Episode sampler for few-shot multimodal EO classification."""
import numpy as np
import torch
from typing import Dict, Optional, List
from ..datasets.base import MultimodalEODataset


class EpisodeSampler:
    """
    Samples N-way K-shot episodes from multimodal EO datasets.
    
    Each episode contains:
    - Support set: N classes × K samples
    - Query set: N classes × Q samples
    """
    
    def __init__(
        self,
        dataset: MultimodalEODataset,
        n_way: int = 5,
        k_shot: int = 5,
        k_query: int = 15,
        num_episodes: int = 1000,
        seed: Optional[int] = None
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_episodes = num_episodes

        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        # Filter classes with enough samples
        self.available_classes = [
            c for c, indices in dataset.class_indices.items()
            if len(indices) >= k_shot + k_query
        ]
        
        if len(self.available_classes) < n_way:
            raise ValueError(
                f"Not enough classes with sufficient samples. "
                f"Need {n_way}, found {len(self.available_classes)}"
            )
    
    def sample_episode(self) -> Dict[str, torch.Tensor]:
        """Sample a single episode."""
        # Select N classes randomly
        selected_classes = self.rng.choice(
            self.available_classes, self.n_way, replace=False
        )
        
        support_hsi, support_aux, support_labels = [], [], []
        query_hsi, query_aux, query_labels = [], [], []
        
        for new_label, original_class in enumerate(selected_classes):
            indices = self.dataset.class_indices[original_class]
            
            # Sample K+Q indices
            sampled_idx = self.rng.choice(
                len(indices),
                self.k_shot + self.k_query,
                replace=False
            )
            
            for i, idx in enumerate(sampled_idx):
                y, x = indices[idx]
                hsi, aux = self.dataset.extract_patch(y, x)
                
                if i < self.k_shot:
                    support_hsi.append(hsi)
                    support_aux.append(aux)
                    support_labels.append(new_label)
                else:
                    query_hsi.append(hsi)
                    query_aux.append(aux)
                    query_labels.append(new_label)
        
        return {
            "support_hsi": torch.stack(support_hsi),
            "support_aux": torch.stack(support_aux),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_hsi": torch.stack(query_hsi),
            "query_aux": torch.stack(query_aux),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
        }
    
    def __iter__(self):
        for _ in range(self.num_episodes):
            yield self.sample_episode()
    
    def __len__(self):
        return self.num_episodes

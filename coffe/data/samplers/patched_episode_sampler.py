"""Episode sampler for pre-patched multimodal EO datasets (MFT format)."""

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchedEpisodeSampler:
    """
    Samples N-way K-shot episodes from pre-patched multimodal EO datasets.

    Works with datasets where patches are already extracted (MFT format):
    - HoustonPatchedDataset
    - TrentoPatchedDataset
    - MUUFLPatchedDataset

    Each episode contains:
    - Support set: N classes × K samples
    - Query set: N classes × Q samples
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int | None = None,
        k_shot: int = 5,
        k_query: int = 15,
        num_episodes: int = 600,
        seed: int | None = None,
        fixed_support: bool = False,
    ):
        """
        Args:
            dataset: PatchedMultimodalDataset instance
            n_way: Number of classes per episode. ``None`` means "use all
                available classes" (those that survive the k_shot+k_query
                filter) — i.e. C-way evaluation.
            k_shot: Number of support samples per class
            k_query: Number of query samples per class
            num_episodes: Number of episodes to generate
            seed: Random seed for reproducibility
            fixed_support: If True, reuse the same support samples per class
        """
        self.dataset = dataset
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_episodes = num_episodes
        self.fixed_support = fixed_support

        self.rng = np.random.RandomState(seed)

        # Build class indices from dataset
        self.class_indices = self._build_class_indices()

        # Filter classes with enough samples
        self.available_classes = [
            c for c, indices in self.class_indices.items() if len(indices) >= k_shot + k_query
        ]

        if n_way is None:
            self.n_way = len(self.available_classes)
        else:
            self.n_way = n_way

        if len(self.available_classes) < self.n_way:
            raise ValueError(
                f"Not enough classes with sufficient samples. "
                f"Need {self.n_way} classes with >= {k_shot + k_query} samples each, "
                f"found {len(self.available_classes)} classes. "
                f"Class sample counts: {[(c, len(idx)) for c, idx in self.class_indices.items()]}"
            )

        self.support_indices_by_class = {}
        if self.fixed_support:
            self.support_indices_by_class = self._select_fixed_support_indices()

    def _select_fixed_support_indices(self) -> dict[int, list[int]]:
        """Select fixed support indices per class for all episodes."""
        fixed_support = {}
        for cls in self.available_classes:
            indices = self.class_indices[cls]
            if len(indices) < self.k_shot + self.k_query:
                raise ValueError(
                    f"Class {cls} has {len(indices)} samples; "
                    f"requires >= {self.k_shot + self.k_query} for fixed support."
                )
            fixed_support[cls] = self.rng.choice(indices, self.k_shot, replace=False).tolist()
        return fixed_support

    def _build_class_indices(self) -> dict[int, list[int]]:
        """Build mapping from class label to sample indices."""
        class_indices: dict[int, list[int]] = {}

        # Handle different dataset types
        if hasattr(self.dataset, "class_indices"):
            # Dataset already has class_indices (list of sample indices per class)
            return self.dataset.class_indices

        if hasattr(self.dataset, "labels"):
            # Build from labels tensor/array
            labels = self.dataset.labels
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()

            for idx in range(len(labels)):
                label = int(labels[idx])
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
        else:
            # Iterate through dataset to build indices
            for idx in range(len(self.dataset)):  # type: ignore[arg-type]
                sample = self.dataset[idx]
                label = int(
                    sample["label"].item()
                    if isinstance(sample["label"], torch.Tensor)
                    else sample["label"]
                )
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)

        return class_indices

    def sample_episode(self) -> dict[str, torch.Tensor]:
        """
        Sample a single N-way K-shot episode.

        Returns:
            Dictionary containing:
            - support_hsi: [N*K, C, H, W]
            - support_aux: [N*K, C_aux, H, W]
            - support_labels: [N*K] (relabeled 0 to N-1)
            - query_hsi: [N*Q, C, H, W]
            - query_aux: [N*Q, C_aux, H, W]
            - query_labels: [N*Q] (relabeled 0 to N-1)
            - original_classes: [N] original class labels
        """
        # Select N classes randomly
        selected_classes = self.rng.choice(self.available_classes, self.n_way, replace=False)

        support_hsi, support_aux, support_labels = [], [], []
        query_hsi, query_aux, query_labels = [], [], []

        for new_label, original_class in enumerate(selected_classes):
            indices = self.class_indices[original_class]

            if self.fixed_support:
                support_idx = self.support_indices_by_class[original_class]
                remaining = [idx for idx in indices if idx not in support_idx]
                if len(remaining) < self.k_query:
                    raise ValueError(
                        f"Not enough query samples for class {original_class}. "
                        f"Need {self.k_query}, have {len(remaining)} after fixed support."
                    )
                query_idx = self.rng.choice(remaining, self.k_query, replace=False).tolist()
                sampled_idx = support_idx + query_idx
            else:
                # Sample K+Q indices without replacement
                sampled_idx = self.rng.choice(
                    indices, self.k_shot + self.k_query, replace=False
                ).tolist()

            for i, idx in enumerate(sampled_idx):
                sample = self.dataset[idx]
                hsi = sample["hsi"]
                aux = sample["aux"]

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
            "original_classes": torch.tensor(selected_classes, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate through episodes."""
        for _ in range(self.num_episodes):
            yield self.sample_episode()

    def __len__(self) -> int:
        return self.num_episodes

    def get_class_info(self) -> dict:
        """Get information about available classes."""
        return {
            "available_classes": self.available_classes,
            "num_available": len(self.available_classes),
            "samples_per_class": {c: len(self.class_indices[c]) for c in self.available_classes},
            "min_samples": min(len(self.class_indices[c]) for c in self.available_classes),
        }


class CrossDatasetEpisodeSampler:
    """
    Samples episodes for cross-dataset evaluation.

    Support set comes from source dataset(s), query from target dataset.
    Useful for testing generalization.
    """

    def __init__(
        self,
        source_datasets: list[Dataset],
        target_dataset: Dataset,
        n_way: int = 5,
        k_shot: int = 5,
        k_query: int = 15,
        num_episodes: int = 600,
        seed: int | None = None,
    ):
        """
        Args:
            source_datasets: List of datasets for support set
            target_dataset: Dataset for query set
            n_way, k_shot, k_query, num_episodes, seed: Same as PatchedEpisodeSampler
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_episodes = num_episodes
        self.rng = np.random.RandomState(seed)

        # Build combined source sampler
        # For simplicity, concatenate source datasets
        from torch.utils.data import ConcatDataset

        self.source_dataset = (
            ConcatDataset(source_datasets) if len(source_datasets) > 1 else source_datasets[0]
        )
        self.target_dataset = target_dataset

        # Build class indices for both
        self.source_class_indices = self._build_class_indices(self.source_dataset)
        self.target_class_indices = self._build_class_indices(self.target_dataset)

        # Find common classes
        source_classes = set(self.source_class_indices.keys())
        target_classes = set(self.target_class_indices.keys())
        common_classes = source_classes & target_classes

        # Filter by sample count
        self.available_classes = [
            c
            for c in common_classes
            if (
                len(self.source_class_indices.get(c, [])) >= k_shot
                and len(self.target_class_indices.get(c, [])) >= k_query
            )
        ]

        if len(self.available_classes) < n_way:
            raise ValueError(
                f"Not enough common classes. Need {n_way}, found {len(self.available_classes)}"
            )

    def _build_class_indices(self, dataset) -> dict[int, list[int]]:
        """Build class indices for a dataset."""
        class_indices: dict[int, list[int]] = {}

        if hasattr(dataset, "class_indices"):
            return dataset.class_indices

        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = int(
                sample["label"].item()
                if isinstance(sample["label"], torch.Tensor)
                else sample["label"]
            )
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        return class_indices

    def sample_episode(self) -> dict[str, torch.Tensor]:
        """Sample cross-dataset episode."""
        selected_classes = self.rng.choice(self.available_classes, self.n_way, replace=False)

        support_hsi, support_aux, support_labels = [], [], []
        query_hsi, query_aux, query_labels = [], [], []

        for new_label, original_class in enumerate(selected_classes):
            # Support from source
            source_indices = self.source_class_indices[original_class]
            support_idx = self.rng.choice(source_indices, self.k_shot, replace=False)

            for idx in support_idx:
                sample = self.source_dataset[idx]
                support_hsi.append(sample["hsi"])
                support_aux.append(sample["aux"])
                support_labels.append(new_label)

            # Query from target
            target_indices = self.target_class_indices[original_class]
            query_idx = self.rng.choice(target_indices, self.k_query, replace=False)

            for idx in query_idx:
                sample = self.target_dataset[idx]
                query_hsi.append(sample["hsi"])
                query_aux.append(sample["aux"])
                query_labels.append(new_label)

        return {
            "support_hsi": torch.stack(support_hsi),
            "support_aux": torch.stack(support_aux),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_hsi": torch.stack(query_hsi),
            "query_aux": torch.stack(query_aux),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
            "original_classes": torch.tensor(selected_classes, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for _ in range(self.num_episodes):
            yield self.sample_episode()

    def __len__(self) -> int:
        return self.num_episodes

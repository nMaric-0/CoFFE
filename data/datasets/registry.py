"""Single source of truth for per-dataset specs in the HyperSIGMA pipeline.

Band counts, class counts, aux-channel counts, and the conventional
on-disk locations for PCA artifacts and adapted checkpoints all live
here so the adapt config, eval script, PCA fitter, and the
``preprocessing`` module cannot drift out of sync. Switching the whole
adapt/eval pipeline to a different dataset is then a single
``dataset`` field (config) or ``--dataset`` flag (CLI).

Adding a new dataset = add its patched ``Dataset`` class (in
``data/datasets``) and one ``DatasetSpec`` entry below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from .patched import (
    PatchedMultimodalDataset,
    HoustonPatchedDataset,
    TrentoPatchedDataset,
    MUUFLPatchedDataset,
)

# Path conventions shared across fit_pca / adapt / eval. Keep these in
# lockstep with the defaults baked into the scripts and configs.
DEFAULT_PCA_DIR = "checkpoints/hypersigma"
DEFAULT_ADAPT_DIR = "checkpoints/hypersigma_adapted"
# The SpatViT branch was pretrained on 3-channel inputs, so the spatial
# PCA always reduces to 3 components regardless of dataset band count.
SPAT_COMPONENTS = 3


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable description of one dataset for the HyperSIGMA pipeline."""

    name: str
    patched_cls: Type[PatchedMultimodalDataset]
    hsi_channels: int
    aux_channels: int
    num_classes: int
    spat_components: int = SPAT_COMPONENTS

    def pca_spat_path(self, root: str = DEFAULT_PCA_DIR) -> str:
        return f"{root}/pca_{self.name}_{self.spat_components}band.pkl"

    def pca_stats_path(self, root: str = DEFAULT_PCA_DIR) -> str:
        return f"{root}/pca_{self.name}_{self.spat_components}band_stats.pkl"

    def adapt_ckpt_dir(self, spat_patch_k: int = 3, root: str = DEFAULT_ADAPT_DIR) -> str:
        return f"{root}/{self.name}_k{spat_patch_k}"


# NOTE: band/class/aux counts here mirror the corresponding patched
# Dataset class properties (data/datasets/{houston,trento,muufl}.py).
DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "houston": DatasetSpec("houston", HoustonPatchedDataset, hsi_channels=144, aux_channels=1, num_classes=15),
    "trento": DatasetSpec("trento", TrentoPatchedDataset, hsi_channels=63, aux_channels=1, num_classes=6),
    "muufl": DatasetSpec("muufl", MUUFLPatchedDataset, hsi_channels=64, aux_channels=2, num_classes=11),
}


def get_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name`` or raise a clear error."""
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset '{name}'. Known datasets: {sorted(DATASET_REGISTRY)}"
        ) from None

"""HyperSIGMA dual-branch encoder + evaluation wrappers.

Vendored upstream sources live under ``third_party/HyperSIGMA``. The
modules in this package wrap those encoders with the small adaptations
needed for Houston few-shot evaluation:

* spatial PCA preprocessor (144 -> 3 bands) for the SpatViT branch
* k=3 patch_embed for the 11x11 input geometry
* random-init spat_map / pos_embeds for the SpecViT branch
* a four-stage gated SEM fusion module (output 512-d)
"""

from .preprocessing import (
    PCAPreprocessor,
    DATASET_PCA_CONFIG,
    fit_dataset_pca,
    load_pca,
)
from .sem import SEM
from .spat_vit_branch import SpatViTBranch
from .spec_vit_branch import SpecViTBranch
from .hypersigma_dual import HyperSIGMADual
from .few_shot import HyperSIGMAFewShot

__all__ = [
    "PCAPreprocessor",
    "DATASET_PCA_CONFIG",
    "fit_dataset_pca",
    "load_pca",
    "SEM",
    "SpatViTBranch",
    "SpecViTBranch",
    "HyperSIGMADual",
    "HyperSIGMAFewShot",
]


# Legacy class names (PAPER_CANON §1) still resolve, with a DeprecationWarning,
# so notebooks and scripts written before the rename keep importing. The alias
# table lives in one place: coffe_compat.LEGACY_CLASSES.
_LEGACY_ALIASES = {"HyperSIGMACosine": "HyperSIGMAFewShot"}


def __getattr__(name):
    canonical = _LEGACY_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import coffe_compat

    return getattr(coffe_compat, name)


def __dir__():
    return sorted(set(__all__) | set(_LEGACY_ALIASES))

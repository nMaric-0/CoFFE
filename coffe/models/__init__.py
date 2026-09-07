"""Model architectures."""

from typing import Any

from .coffe import CoFFE
from .coffe_aux_token import CoFFEAuxToken
from .mft_original import MFTOriginal

__all__ = [
    "CoFFE",
    "CoFFEAuxToken",
    "MFTOriginal",
]


# Legacy class names (PAPER_CANON §1) still resolve, with a DeprecationWarning,
# so notebooks and scripts written before the rename keep importing. The alias
# table lives in one place: coffe_compat.LEGACY_CLASSES.
_LEGACY_ALIASES = {"MFTCPEACosine": "CoFFE", "MFTOriginalCosine": "MFTOriginal"}


def __getattr__(name: str) -> Any:
    canonical = _LEGACY_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from coffe import compat as coffe_compat

    return getattr(coffe_compat, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LEGACY_ALIASES))

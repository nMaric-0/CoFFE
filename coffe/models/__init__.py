"""Model architectures."""
from .coffe import CoFFE
from .mft_original import MFTOriginal

__all__ = [
    "CoFFE",
    "MFTOriginal",
]


# Legacy class names (PAPER_CANON §1) still resolve, with a DeprecationWarning,
# so notebooks and scripts written before the rename keep importing. The alias
# table lives in one place: coffe_compat.LEGACY_CLASSES.
_LEGACY_ALIASES = {"MFTCPEACosine": "CoFFE", "MFTOriginalCosine": "MFTOriginal"}


def __getattr__(name):
    canonical = _LEGACY_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from coffe import compat as coffe_compat

    return getattr(coffe_compat, name)


def __dir__():
    return sorted(set(__all__) | set(_LEGACY_ALIASES))

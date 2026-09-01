"""CoFFE — a compact in-domain fusion encoder for few-shot HSI-LiDAR classification.

Public code release for *"A Compact In-Domain Fusion Encoder versus a
Hyperspectral Foundation Model for Few-Shot HSI-LiDAR Land-Cover
Classification"* (N. Marić & D. Kocev). ``PAPER_CANON.md`` at the repo root is
the source of truth for names, protocol constants and results.

The three routes compared in the paper:

- ``coffe.models.CoFFE``      — the compact encoder (input-level HSI+LiDAR fusion)
- ``coffe.models.MFTOriginal``— the MFT architectural control (external fusion token)
- ``coffe.models.hypersigma`` — the HyperSIGMA foundation-model route

Sub-packages: :mod:`coffe.models`, :mod:`coffe.pretrain`, :mod:`coffe.data`,
:mod:`coffe.eval`, :mod:`coffe.runners`, :mod:`coffe.utils`, :mod:`coffe.compat`.

Imports here stay lazy: ``import coffe`` must not pull in torch.
"""

__version__ = "0.9.0"

__all__ = ["__version__", "CoFFE", "MFTOriginal"]


def __getattr__(name):
    # Lazy re-export of the two headline encoders, so that `import coffe` is
    # cheap and torch is only imported when a model is actually requested.
    if name in ("CoFFE", "MFTOriginal"):
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

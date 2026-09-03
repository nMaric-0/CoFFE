"""CoFFE — a compact in-domain fusion encoder for few-shot HSI-LiDAR classification.

Public code release for *"A Compact In-Domain Fusion Encoder versus a
Hyperspectral Foundation Model for Few-Shot HSI-LiDAR Land-Cover
Classification"* (N. Marić & D. Kocev). ``PAPER_CANON.md`` at the repo root is
the source of truth for names, protocol constants and results.

The three routes compared in the paper, and the entry point each is driven by:

===================  ==========================  ==============================
Route                Encoder                     Runner
===================  ==========================  ==============================
CoFFE                :class:`CoFFE`              :func:`run_pretrain`
MFT control          :class:`MFTOriginal`        :func:`run_pretrain`
HyperSIGMA           :class:`HyperSIGMAFewShot`  :func:`run_adapt_hypersigma`
===================  ==========================  ==============================

All three are then scored by the same frozen-encoder protocol — 5-shot, N-way
(every class in the scene), 1000 episodes, Euclidean nearest-class-mean — via
:func:`run_evaluation` (CoFFE and the MFT control) or
:func:`run_hypersigma_evaluation`.

Sub-packages: :mod:`coffe.models`, :mod:`coffe.pretrain`, :mod:`coffe.data`,
:mod:`coffe.eval`, :mod:`coffe.runners`, :mod:`coffe.utils`, :mod:`coffe.compat`.

Everything exported here is resolved lazily: ``import coffe`` must stay cheap
and must not pull in torch.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.9.0"

#: Exported name -> (defining module, attribute), resolved on first access.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Encoders — one per route (PAPER_CANON §1).
    "CoFFE": ("coffe.models", "CoFFE"),
    "MFTOriginal": ("coffe.models", "MFTOriginal"),
    "HyperSIGMAFewShot": ("coffe.models.hypersigma", "HyperSIGMAFewShot"),
    # Runner entry points — the experiment-tree wrappers the notebooks and the
    # thin CLIs under scripts/ both call.
    "run_pretrain": ("coffe.runners.pretrain_runner", "run_pretrain"),
    "run_adapt_hypersigma": ("coffe.runners.adapt_runner", "run_adapt_hypersigma"),
    "run_evaluation": ("coffe.runners.eval_runner", "run_evaluation"),
    "run_hypersigma_evaluation": ("coffe.runners.eval_runner", "run_hypersigma_evaluation"),
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted(__all__)

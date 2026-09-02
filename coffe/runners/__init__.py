"""Experiment-tree runners: the entry points notebooks and the thin CLIs share.

Each ``run_*`` function creates (or reuses) a directory under
``experiments/<name>/``, redirects checkpoints, logs and results into it, and
writes a finalization record — so a run is reconstructible from its own
directory. :mod:`coffe.runners.experiments` holds the ``ExperimentLogger`` /
``PretrainExperiment`` / ``EvalRun`` classes that define that layout.

- :func:`run_pretrain` — per-scene SimMIM/MAE pretraining of CoFFE or the MFT
  control (PAPER_CANON §3)
- :func:`run_adapt_hypersigma` — label-free HyperSIGMA adaptation (§5)
- :func:`run_evaluation`, :func:`run_hypersigma_evaluation` — the frozen-encoder
  5-shot N-way Euclidean nearest-class-mean protocol (§4)

Importing this package stays torch-free; the runners import torch when called.
"""

from .adapt_runner import run_adapt_hypersigma
from .eval_runner import run_evaluation, run_hypersigma_evaluation
from .experiments import (
    EvalRun,
    ExperimentLogger,
    PretrainExperiment,
    load_all_evaluations,
)
from .pretrain_runner import run_pretrain

__all__ = [
    "EvalRun",
    "ExperimentLogger",
    "PretrainExperiment",
    "load_all_evaluations",
    "run_adapt_hypersigma",
    "run_evaluation",
    "run_hypersigma_evaluation",
    "run_pretrain",
]

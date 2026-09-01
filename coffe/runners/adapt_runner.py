"""Notebook-friendly wrapper around coffe.pretrain.hypersigma_adapt.run_adapt.

Mirrors :mod:`coffe.runners.pretrain_runner` so HyperSIGMA Level-2 MAE adaptation
runs land under ``experiments/<name>/`` with the same packaging:

- ``pretrain_config.yaml``   — frozen merged config (base YAML + overrides)
- ``pretrain_overrides.yaml`` — just the overrides dict, for diffing
- ``pretrain_metadata.json``  — timestamp, git SHA, resolved device,
                                wallclock, final/best loss, history
- ``README.md``               — name + description
- ``pretrain.log``            — full training log (file handler)
- ``checkpoints/``            — `checkpoint_epoch_*.pth` + `checkpoint.pth`
                                + `checkpoint_final.pth`

Pass an ``overrides`` dict to tweak hyperparameters without editing the
YAML, e.g. ``{"pretrain": {"lr": 5e-5}, "model": {"adapt_mode": "spatial_only"}}``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .experiments import (
    DEFAULT_EXPERIMENTS_ROOT,
    ExperimentLogger,
    PretrainExperiment,
    attach_file_logger,
    detach_file_logger,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

def _load_config(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(config, dict):
        return config
    with open(config) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_adapt_hypersigma(
    name: str,
    description: str,
    config: Union[str, Path, Dict[str, Any]],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    experiments_root: Union[str, Path] = DEFAULT_EXPERIMENTS_ROOT,
    overwrite: bool = False,
) -> PretrainExperiment:
    """Run HyperSIGMA Level-2 MAE adaptation end-to-end under ``experiments/<name>/``.

    Args:
        name: Short experiment identifier (becomes the directory name).
        description: One-paragraph description, written into README.md.
        config: Path to a YAML config OR a dict matching the YAML schema.
            Typical: ``configs/hypersigma/houston_patchnative_joint_sem.yaml``.
        overrides: Optional deep-merged overrides applied to the loaded
            config (handy for sweeps — e.g.
            ``{"pretrain": {"lr": 5e-5, "epochs": 200}}``,
            ``{"model": {"adapt_mode": "spatial_only"}}``,
            ``{"hardware": {"device": "cuda:1"}}``).
        experiments_root: Where to place the experiment directory.
        overwrite: If True, reuse an existing experiment directory.

    Returns the ``PretrainExperiment`` handle; its ``.metadata`` contains
    the final ``history`` summary after the run.
    """
    from coffe.pretrain.hypersigma_adapt import run_adapt as _run_adapt

    base_config_path = str(config) if isinstance(config, (str, Path)) else None
    cfg = _load_config(config)
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    logger_ = ExperimentLogger(experiments_root=experiments_root, repo_root=REPO_ROOT)
    exp = logger_.start_pretrain(
        name,
        description,
        cfg,
        overwrite=overwrite,
        overrides=overrides or {},
        base_config_path=base_config_path,
    )

    handler = attach_file_logger(exp)
    # Make sure INFO logs from the adapt script reach the file handler the
    # logger above just attached. `basicConfig` is a no-op once the root
    # logger has handlers, so we set the level explicitly.
    logging.getLogger().setLevel(logging.INFO)
    log = logging.getLogger(__name__)
    log.info(f"Starting HyperSIGMA adapt experiment '{name}' at {exp.root}")

    try:
        history = _run_adapt(
            cfg,
            checkpoint_dir=str(exp.checkpoints_dir),
            log_dir=str(exp.root),
        )
        # The adapt script catches KeyboardInterrupt internally, saves
        # checkpoint_interrupted.pth, and returns a history with
        # interrupted=True. Surface that as `status: "interrupted"` in
        # metadata instead of `status: "complete"`.
        if history.get("interrupted"):
            exp.finalize(history=history, status="interrupted")
            log.warning(
                "HyperSIGMA adapt experiment '%s' interrupted after %d/%d epochs. "
                "checkpoint_interrupted.pth saved.",
                name, history.get("epochs_run", 0), history.get("epochs_requested", 0),
            )
        else:
            exp.finalize(history=history)
            log.info(f"HyperSIGMA adapt experiment '{name}' complete.")
        return exp
    except Exception as e:
        log.exception("HyperSIGMA adapt run failed")
        exp.mark_failed(str(e))
        raise
    finally:
        detach_file_logger(handler)

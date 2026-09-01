"""Notebook-friendly wrapper around coffe.pretrain.loop.run_pretrain.

The runner creates a directory under experiments/<name>/, redirects the
trainer's checkpoint/log output there, attaches a file logger so the run's
console output lands in experiments/<name>/pretrain.log, and writes a
finalization record (best loss, wallclock, etc.) when the run finishes.
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


def run_pretrain(
    name: str,
    description: str,
    config: Union[str, Path, Dict[str, Any]],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    resume: Optional[str] = None,
    experiments_root: Union[str, Path] = DEFAULT_EXPERIMENTS_ROOT,
    overwrite: bool = False,
) -> PretrainExperiment:
    """Run a masked-pretraining experiment end-to-end.

    Args:
        name: Short experiment identifier (becomes the directory name).
        description: One-paragraph description, written into README.md.
        config: Path to a YAML config OR a dict matching the YAML schema.
        overrides: Optional deep-merged overrides applied to the loaded config
            (handy for sweeps from a notebook — e.g. `{"pretrain": {"lr": 3e-4}}`).
        resume: Optional checkpoint path to resume from.
        experiments_root: Where to place the experiment directory.
        overwrite: If True, reuse an existing experiment directory.

    Returns the PretrainExperiment handle; its `.metadata` contains the final
    `history` summary after the run.
    """
    from coffe.pretrain.loop import run_pretrain as _run

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger(__name__)
    log.info(f"Starting pretrain experiment '{name}' at {exp.root}")

    try:
        history = _run(
            config=cfg,
            checkpoint_dir=str(exp.checkpoints_dir),
            log_dir=str(exp.root),
            resume=resume,
        )
        exp.finalize(history=history)
        log.info(f"Pretrain experiment '{name}' complete.")
        return exp
    except Exception as e:
        log.exception("Pretrain run failed")
        exp.mark_failed(str(e))
        raise
    finally:
        detach_file_logger(handler)


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

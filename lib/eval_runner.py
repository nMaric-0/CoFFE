"""Notebook-friendly wrapper around scripts.evaluate_cosine.run_evaluation.

The runner creates an evaluation subdir under an existing experiment, points
the evaluator at it for `results.json` and the plots dir, and attaches a file
logger so the run's console output lands in `eval.log`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .experiments import (
    DEFAULT_EXPERIMENTS_ROOT,
    EvalRun,
    ExperimentLogger,
    attach_file_logger,
    detach_file_logger,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Keys read from the pretrain experiment's saved config to seed eval defaults.
# Eval-time overrides in `eval_params` still win.
_ARCH_KEYS_FROM_MODEL = (
    "embed_dim",
    "num_heads",
    "num_layers",
    "lambda_factor",
    "dropout",
    "use_projection",
    "proj_hidden_dim",
    "proj_num_layers",
    "proj_l2_normalize",
)
_ARCH_KEYS_FROM_DATA = ("patch_size",)


def load_pretrain_config(
    experiment_name: str,
    *,
    experiments_root: Union[str, Path] = DEFAULT_EXPERIMENTS_ROOT,
) -> Dict[str, Any]:
    """Read experiments/<name>/pretrain_config.yaml as a plain dict."""
    logger_ = ExperimentLogger(experiments_root=experiments_root, repo_root=REPO_ROOT)
    exp_dir = logger_.get_experiment(experiment_name)
    cfg_path = exp_dir / "pretrain_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No pretrain_config.yaml under {exp_dir}")
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


def _arch_defaults_from_pretrain(pretrain_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract architecture-shaped keys from a pretrain config dict."""
    model_cfg = pretrain_cfg.get("model", {}) or {}
    data_cfg = pretrain_cfg.get("data", {}) or {}
    out: Dict[str, Any] = {}
    for k in _ARCH_KEYS_FROM_MODEL:
        if k in model_cfg:
            out[k] = model_cfg[k]
    for k in _ARCH_KEYS_FROM_DATA:
        if k in data_cfg:
            out[k] = data_cfg[k]
    return out


def find_checkpoint(
    experiment_name: str,
    *,
    epoch: Optional[int] = None,
    experiments_root: Union[str, Path] = DEFAULT_EXPERIMENTS_ROOT,
) -> str:
    """Locate a checkpoint inside experiments/<name>/checkpoints/.

    If `epoch` is None, returns the highest-numbered checkpoint
    (or the file named `checkpoint_final.pth` if present).
    """
    logger_ = ExperimentLogger(experiments_root=experiments_root, repo_root=REPO_ROOT)
    exp_dir = logger_.get_experiment(experiment_name)
    ckpts = exp_dir / "checkpoints"
    if not ckpts.exists():
        raise FileNotFoundError(f"No checkpoints/ under {exp_dir}")

    if epoch is None:
        final = ckpts / "checkpoint_final.pth"
        if final.exists():
            return str(final)
        candidates = sorted(ckpts.glob("checkpoint_epoch_*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint files found in {ckpts}")
        return str(candidates[-1])

    target = ckpts / f"checkpoint_epoch_{epoch}.pth"
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found: {target}")
    return str(target)


def run_hypersigma_evaluation(
    experiment_name: str,
    eval_name: str,
    *,
    adapted_checkpoint: Optional[str] = None,
    eval_params: Optional[Dict[str, Any]] = None,
    experiments_root: Union[str, Path] = DEFAULT_EXPERIMENTS_ROOT,
    overwrite: bool = False,
) -> EvalRun:
    """Run HyperSIGMA few-shot evaluation under ``experiments/<name>/``.

    Mirrors :func:`run_evaluation` (MFT-CPEA-Cosine) but dispatches to
    ``scripts.evaluate_hypersigma_cosine.run_evaluation``. The
    ``adapted_checkpoint`` argument can be a path or the literal string
    ``"none"`` (the unadapted ablation).
    """
    from scripts.evaluate_hypersigma_cosine import run_evaluation as _run

    params = dict(eval_params or {})
    if "dataset" not in params:
        raise ValueError("eval_params must include 'dataset'")
    if adapted_checkpoint is not None:
        params["adapted_checkpoint"] = adapted_checkpoint

    logger_ = ExperimentLogger(experiments_root=experiments_root, repo_root=REPO_ROOT)

    config_for_logging = {
        "adapted_checkpoint": params.get("adapted_checkpoint"),
        "mode": params.get("mode"),
        "spat_patch_k": params.get("spat_patch_k"),
        **params,
    }

    eval_run = logger_.start_eval(
        experiment_name, eval_name, config_for_logging, overwrite=overwrite,
    )

    handler = attach_file_logger(eval_run)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger(__name__)
    log.info(
        f"Starting HyperSIGMA eval '{eval_name}' for experiment '{experiment_name}'"
    )

    eval_run.plots_dir.mkdir(parents=True, exist_ok=True)
    forwarded = dict(params)
    dataset = forwarded.pop("dataset")
    forwarded["output"] = str(eval_run.results_path)
    forwarded["output_dir"] = str(eval_run.plots_dir)

    try:
        results = _run(dataset=dataset, **forwarded)
        if isinstance(results, list):
            summary = results[-1]
        else:
            summary = results
        # ``summary`` from the HyperSIGMA eval has separate cosine/euclidean
        # blocks; pick the primary metric for the EvalRun.finalize summary.
        primary_metric = forwarded.get("distance_metric", "cosine")
        finalize_payload = summary.get(primary_metric, summary) if isinstance(summary, dict) else summary
        eval_run.finalize(results=finalize_payload)
        log.info(f"HyperSIGMA eval '{eval_name}' complete.")
        return eval_run
    except Exception as e:
        log.exception("HyperSIGMA eval run failed")
        eval_run.mark_failed(str(e))
        raise
    finally:
        detach_file_logger(handler)


def run_evaluation(
    experiment_name: str,
    eval_name: str,
    *,
    epoch: Optional[int] = None,
    checkpoint: Optional[str] = None,
    eval_params: Optional[Dict[str, Any]] = None,
    experiments_root: Union[str, Path] = DEFAULT_EXPERIMENTS_ROOT,
    overwrite: bool = False,
) -> EvalRun:
    """Run cosine few-shot evaluation against a saved experiment.

    Args:
        experiment_name: Name of the pretrain experiment to evaluate.
        eval_name: Short identifier for this evaluation run.
        epoch: Which checkpoint epoch to load (None = latest / final).
        checkpoint: Optional explicit checkpoint path (overrides `epoch`).
            Use the string "random" to evaluate an untrained encoder.
        eval_params: Dict of evaluation hyperparameters mirroring the CLI
            flags of `scripts/evaluate_cosine.py`. Must include at least
            `dataset` (one of "houston", "trento", "muufl").
        experiments_root: Directory containing experiments.
        overwrite: If True, reuse an existing eval directory.

    Returns the EvalRun handle with finalized metadata.
    """
    from scripts.evaluate_cosine import run_evaluation as _run

    user_params = dict(eval_params or {})
    if "dataset" not in user_params:
        raise ValueError("eval_params must include 'dataset'")

    if checkpoint is None:
        if user_params.get("checkpoint") in ("random", "none", "null"):
            checkpoint = user_params.pop("checkpoint")
        else:
            checkpoint = find_checkpoint(
                experiment_name, epoch=epoch, experiments_root=experiments_root,
            )

    # Pull architecture defaults from the pretraining experiment so the
    # notebook doesn't have to re-specify them (and can't silently drift).
    # User-supplied keys win.
    pretrain_cfg = load_pretrain_config(experiment_name, experiments_root=experiments_root)
    arch_defaults = _arch_defaults_from_pretrain(pretrain_cfg)
    params: Dict[str, Any] = {**arch_defaults, **user_params}

    arch_summary = ", ".join(f"{k}={params[k]}" for k in arch_defaults if k in params)
    logging.getLogger(__name__).info(
        f"Loaded architecture from experiments/{experiment_name}/pretrain_config.yaml: "
        f"{arch_summary or '(none)'}"
    )

    logger_ = ExperimentLogger(experiments_root=experiments_root, repo_root=REPO_ROOT)

    config_for_logging = {
        "checkpoint": checkpoint,
        "epoch": epoch,
        "arch_from_pretrain": arch_defaults,
        **params,
    }

    eval_run = logger_.start_eval(
        experiment_name, eval_name, config_for_logging, overwrite=overwrite,
    )

    handler = attach_file_logger(eval_run)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger(__name__)
    log.info(f"Starting eval '{eval_name}' for experiment '{experiment_name}'")
    log.info(f"Checkpoint: {checkpoint}")

    eval_run.plots_dir.mkdir(parents=True, exist_ok=True)
    forwarded = dict(params)
    forwarded.pop("checkpoint", None)
    forwarded.pop("dataset", None)
    dataset = params["dataset"]
    forwarded["output"] = str(eval_run.results_path)
    forwarded["output_dir"] = str(eval_run.plots_dir)

    try:
        results = _run(
            checkpoint=checkpoint,
            dataset=dataset,
            **forwarded,
        )
        if isinstance(results, list):
            summary = results[-1]
        else:
            summary = results
        eval_run.finalize(results=summary)
        log.info(f"Eval '{eval_name}' complete.")
        return eval_run
    except Exception as e:
        log.exception("Eval run failed")
        eval_run.mark_failed(str(e))
        raise
    finally:
        detach_file_logger(handler)

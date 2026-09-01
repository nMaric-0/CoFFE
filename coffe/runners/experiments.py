"""File-based experiment tracking.

Directory layout (per pretrain experiment):

    experiments/<experiment_name>/
        README.md                   name + short description (human-written)
        pretrain_config.yaml        frozen training config
        pretrain_metadata.json      timestamp, git SHA, dataset, final/best loss, wallclock
        pretrain.log                training stdout/stderr
        checkpoints/
            checkpoint_epoch_*.pth
            checkpoint_final.pth
        evaluations/<eval_name>/
            eval_config.json        checkpoint epoch, n_way, k_shot, metric, ...
            eval_metadata.json      timestamp, git SHA, num_episodes
            results.json            OA / AA / Kappa + per-class
            eval.log
            plots/                  optional, written by the eval runner

`ExperimentLogger.start_pretrain` and `start_eval` return small handle objects
that own a directory and expose path/metadata-update helpers. They do not
import torch — keep this module dependency-light so notebooks can import it
without spinning up CUDA.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_EXPERIMENTS_ROOT = "experiments"


def _git_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _slugify(name: str) -> str:
    bad = ' /\\:?*"<>|\t\n'
    out = "".join("_" if c in bad else c for c in name.strip())
    if not out:
        raise ValueError("Experiment name resolves to an empty slug")
    return out


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str, sort_keys=True)


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


@dataclass
class PretrainExperiment:
    """Handle to a single pretraining experiment directory."""

    name: str
    root: Path
    description: str
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _started_at: float = field(default_factory=time.monotonic)

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def log_file(self) -> Path:
        return self.root / "pretrain.log"

    @property
    def config_path(self) -> Path:
        return self.root / "pretrain_config.yaml"

    @property
    def metadata_path(self) -> Path:
        return self.root / "pretrain_metadata.json"

    @property
    def readme_path(self) -> Path:
        return self.root / "README.md"

    @property
    def evaluations_dir(self) -> Path:
        return self.root / "evaluations"

    def finalize(self, history: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        wall = time.monotonic() - self._started_at
        update: Dict[str, Any] = {
            "status": "complete",
            "finished_at": _now_iso(),
            "wallclock_seconds": round(wall, 2),
        }
        if history is not None:
            update["history"] = {
                k: history[k]
                for k in ("best_val_loss", "final_train_loss", "epochs_run")
                if k in history
            }
            for k in ("resolved_device", "cuda_device_name"):
                if k in history:
                    update[k] = history[k]
        update.update(extra)
        self.metadata.update(update)
        _write_json(self.metadata_path, self.metadata)

    def mark_failed(self, error: str) -> None:
        self.metadata.update({
            "status": "failed",
            "finished_at": _now_iso(),
            "error": error,
        })
        _write_json(self.metadata_path, self.metadata)


@dataclass
class EvalRun:
    """Handle to a single evaluation under an experiment."""

    experiment_name: str
    eval_name: str
    root: Path
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _started_at: float = field(default_factory=time.monotonic)

    @property
    def config_path(self) -> Path:
        return self.root / "eval_config.json"

    @property
    def metadata_path(self) -> Path:
        return self.root / "eval_metadata.json"

    @property
    def results_path(self) -> Path:
        return self.root / "results.json"

    @property
    def log_file(self) -> Path:
        return self.root / "eval.log"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    def finalize(self, results: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        wall = time.monotonic() - self._started_at
        update: Dict[str, Any] = {
            "status": "complete",
            "finished_at": _now_iso(),
            "wallclock_seconds": round(wall, 2),
        }
        if results is not None:
            summary = {}
            for key in ("OA", "AA", "Kappa"):
                if key in results and isinstance(results[key], dict):
                    summary[key] = {k: results[key].get(k) for k in ("mean", "std", "ci_95")}
            if "num_episodes" in results:
                summary["num_episodes"] = results["num_episodes"]
            update["summary"] = summary
        update.update(extra)
        self.metadata.update(update)
        _write_json(self.metadata_path, self.metadata)

    def mark_failed(self, error: str) -> None:
        self.metadata.update({
            "status": "failed",
            "finished_at": _now_iso(),
            "error": error,
        })
        _write_json(self.metadata_path, self.metadata)


class ExperimentLogger:
    """Top-level entry point — creates and locates experiment directories.

    The same `ExperimentLogger` can be reused across multiple runs in one
    notebook; each `start_*` call writes a fresh subtree.
    """

    def __init__(
        self,
        experiments_root: os.PathLike | str = DEFAULT_EXPERIMENTS_ROOT,
        repo_root: Optional[os.PathLike | str] = None,
    ) -> None:
        self.root = Path(experiments_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._repo_root = Path(repo_root).resolve() if repo_root else Path.cwd()

    def _git_sha(self) -> Optional[str]:
        return _git_sha(self._repo_root)

    def _experiment_dir(self, name: str) -> Path:
        return self.root / _slugify(name)

    def start_pretrain(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        *,
        overwrite: bool = False,
        overrides: Optional[Dict[str, Any]] = None,
        base_config_path: Optional[str] = None,
    ) -> PretrainExperiment:
        """Create a new pretraining experiment directory.

        Raises FileExistsError if the directory exists and `overwrite` is
        False, so accidentally rerunning the same notebook cell doesn't
        clobber a previous run.

        `overrides` (the raw dict the user passed in the notebook, *not* the
        merged config) and `base_config_path` (the YAML the overrides were
        applied on top of) are persisted alongside the frozen merged config
        so the lineage of the run is recoverable.
        """
        exp_dir = self._experiment_dir(name)
        if exp_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Experiment '{name}' already exists at {exp_dir}. "
                "Pass overwrite=True to reuse, or pick a new name."
            )

        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "evaluations").mkdir(exist_ok=True)

        overrides = dict(overrides or {})
        metadata = {
            "name": name,
            "description": description,
            "status": "running",
            "started_at": _now_iso(),
            "git_sha": self._git_sha(),
            "base_config_path": base_config_path,
            "overrides": overrides,
        }
        _write_json(exp_dir / "pretrain_metadata.json", metadata)
        _write_yaml(exp_dir / "pretrain_config.yaml", config)
        _write_yaml(exp_dir / "pretrain_overrides.yaml", overrides)
        (exp_dir / "README.md").write_text(
            f"# {name}\n\n{description}\n\n"
            f"Created: {metadata['started_at']}\n"
            f"Git SHA: {metadata['git_sha']}\n"
            f"Base config: {base_config_path}\n"
        )

        return PretrainExperiment(
            name=name,
            root=exp_dir,
            description=description,
            config=config,
            metadata=metadata,
        )

    def start_eval(
        self,
        experiment_name: str,
        eval_name: str,
        config: Dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> EvalRun:
        """Create a new evaluation directory under an existing experiment.

        The experiment must already exist (pretraining must have produced a
        checkpoint). The eval_name is slugified into a subdir name.
        """
        exp_dir = self._experiment_dir(experiment_name)
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_name}' not found at {exp_dir}"
            )

        eval_dir = exp_dir / "evaluations" / _slugify(eval_name)
        if eval_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Evaluation '{eval_name}' already exists at {eval_dir}. "
                "Pass overwrite=True to reuse, or pick a new name."
            )
        eval_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "experiment": experiment_name,
            "eval_name": eval_name,
            "status": "running",
            "started_at": _now_iso(),
            "git_sha": self._git_sha(),
        }
        _write_json(eval_dir / "eval_metadata.json", metadata)
        _write_json(eval_dir / "eval_config.json", config)

        return EvalRun(
            experiment_name=experiment_name,
            eval_name=eval_name,
            root=eval_dir,
            config=config,
            metadata=metadata,
        )

    def get_experiment(self, name: str) -> Path:
        exp_dir = self._experiment_dir(name)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment '{name}' not found at {exp_dir}")
        return exp_dir

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Return one summary dict per experiment under the root."""
        out: List[Dict[str, Any]] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / "pretrain_metadata.json"
            if not meta_path.exists():
                continue
            meta = _read_json(meta_path)
            out.append({
                "name": meta.get("name", child.name),
                "description": meta.get("description", ""),
                "status": meta.get("status", "unknown"),
                "started_at": meta.get("started_at"),
                "finished_at": meta.get("finished_at"),
                "path": str(child),
                "history": meta.get("history", {}),
            })
        return out

    def list_evaluations(self, experiment_name: str) -> List[Dict[str, Any]]:
        exp_dir = self.get_experiment(experiment_name)
        evals_dir = exp_dir / "evaluations"
        if not evals_dir.exists():
            return []
        out: List[Dict[str, Any]] = []
        for child in sorted(evals_dir.iterdir()):
            if not child.is_dir():
                continue
            cfg_path = child / "eval_config.json"
            meta_path = child / "eval_metadata.json"
            res_path = child / "results.json"
            if not cfg_path.exists():
                continue
            entry = {
                "eval_name": child.name,
                "config": _read_json(cfg_path),
                "metadata": _read_json(meta_path) if meta_path.exists() else {},
                "results": _read_json(res_path) if res_path.exists() else None,
                "path": str(child),
            }
            out.append(entry)
        return out


def load_all_evaluations(
    experiments_root: os.PathLike | str = DEFAULT_EXPERIMENTS_ROOT,
) -> List[Dict[str, Any]]:
    """Walk the experiments tree and return a flat list of every evaluation run.

    Each entry contains the experiment name + description, the eval config,
    and the summary metrics — enough to feed into a pandas DataFrame for
    cross-experiment comparison.
    """
    logger = ExperimentLogger(experiments_root)
    rows: List[Dict[str, Any]] = []
    for exp in logger.list_experiments():
        for ev in logger.list_evaluations(exp["name"]):
            cfg = ev["config"]
            res = ev.get("results") or {}
            row = {
                "experiment": exp["name"],
                "description": exp["description"],
                "eval_name": ev["eval_name"],
                "checkpoint": cfg.get("checkpoint"),
                "dataset": cfg.get("dataset"),
                "n_way": cfg.get("n_way"),
                "k_shot": cfg.get("k_shot"),
                "k_query": cfg.get("k_query"),
                "distance_metric": cfg.get("distance_metric"),
                "temperature": cfg.get("temperature"),
                "prototype_mode": cfg.get("prototype_mode"),
                "num_episodes": res.get("num_episodes"),
                "OA_mean": (res.get("OA") or {}).get("mean") if isinstance(res.get("OA"), dict) else res.get("OA"),
                "OA_ci": (res.get("OA") or {}).get("ci_95") if isinstance(res.get("OA"), dict) else None,
                "AA_mean": (res.get("AA") or {}).get("mean") if isinstance(res.get("AA"), dict) else res.get("AA"),
                "AA_ci": (res.get("AA") or {}).get("ci_95") if isinstance(res.get("AA"), dict) else None,
                "Kappa_mean": (res.get("Kappa") or {}).get("mean") if isinstance(res.get("Kappa"), dict) else res.get("Kappa"),
                "Kappa_ci": (res.get("Kappa") or {}).get("ci_95") if isinstance(res.get("Kappa"), dict) else None,
                "path": ev["path"],
            }
            rows.append(row)
    return rows


def attach_file_logger(handle, log_file: Optional[Path] = None) -> logging.Handler:
    """Add a FileHandler routing the root logger to handle.log_file.

    Returns the handler so the caller can remove it after the run (avoids
    duplicate output when the same notebook kicks off several runs).
    """
    target = Path(log_file) if log_file else handle.log_file
    target.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(target, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


def detach_file_logger(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()

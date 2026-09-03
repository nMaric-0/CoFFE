"""Regression tests for ``scripts/compile_results.py``.

Two phase-6 gate decisions are pinned here.

**The modality classifier must accept both frozen on-disk conventions for "no
LiDAR".** The canonical Table 2 runs are named ``*_no_lidar``; the 5-seed
significance runs are named ``<scene>_hsi_only_<variant>_seed<s>`` and
``<scene>_enhanced_mae_hsi_only_seed<s>``
(``scripts/reproduce/sig_significance_config.py``). Both are DO-NOT-RENAME
(PAPER_CANON §7.3). Until the phase-6 gate the compiler matched only
``no_lidar``, so every significance HSI-only run was labelled ``HSI+LiDAR``.

**The compiler must not overwrite its committed artifact by accident.** It used
to take no arguments and ignore ``argv``, so a bare ``--help`` during CLI smoke
regenerated ``docs/presentation/RESULTS.json`` — which happened twice, in
phases 5 and 6.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import compile_results

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "compile_results.py"


def _result(dataset: str = "houston", model_type: str = "CoFFE") -> dict:
    """The minimum a ``results.json`` needs for :func:`classify`."""
    return {"dataset": dataset, "model_type": model_type}


@pytest.mark.parametrize(
    "experiment",
    [
        # canonical Table 2 HSI-only dirs
        "houston_enhanced_spatial_no_lidar",
        "trento_enhanced_spectral_no_lidar",
        "muufl_enhanced_spectral_spatial_no_lidar",
        # 5-seed significance HSI-only dirs (the ones that used to be mislabelled)
        "houston_hsi_only_spatial_seed42",
        "trento_hsi_only_spectral_seed123",
        "muufl_hsi_only_both_seed1011",
        "houston_enhanced_mae_hsi_only_seed789",
    ],
)
def test_hsi_only_run_dirs_are_labelled_hsi_only(experiment: str) -> None:
    dataset = experiment.split("_")[0]
    _model, _regime, modality, _reason = compile_results.classify(
        experiment, f"{dataset}_15way_5shot", _result(dataset)
    )
    assert modality == "HSI-only", f"{experiment} was classified {modality!r}"


@pytest.mark.parametrize(
    "experiment",
    [
        "houston_enhanced_spatial_run1",
        "trento_enhanced_mae_lidar_seed42",
        "muufl_enhanced_spectral_spatial_run2",
    ],
)
def test_fused_run_dirs_stay_hsi_plus_lidar(experiment: str) -> None:
    dataset = experiment.split("_")[0]
    _model, _regime, modality, _reason = compile_results.classify(
        experiment, f"{dataset}_15way_5shot", _result(dataset)
    )
    assert modality == "HSI+LiDAR"


def test_both_hsi_only_conventions_are_declared() -> None:
    # The tuple is the single place the two frozen conventions are written down.
    assert compile_results.HSI_ONLY_MARKERS == ("no_lidar", "hsi_only")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=120,
    )


def test_help_does_no_work() -> None:
    """``--help`` must print usage and exit, touching nothing."""
    proc = _run("--help")
    assert proc.returncode == 0
    assert "--force" in proc.stdout and "--out" in proc.stdout
    assert "Wrote" not in proc.stdout


def test_existing_output_is_refused_without_force(tmp_path: Path) -> None:
    """An existing --out is refused, and left byte-identical."""
    out = tmp_path / "RESULTS.json"
    out.write_text('{"sentinel": true}\n')

    proc = _run("--out", str(out))

    assert proc.returncode != 0
    assert "refusing to overwrite" in (proc.stderr + proc.stdout)
    assert out.read_text() == '{"sentinel": true}\n'


def test_default_output_is_the_committed_artifact() -> None:
    """The guard protects the real artifact, not some other path."""
    assert compile_results.DEFAULT_OUT == REPO / "docs" / "presentation" / "RESULTS.json"

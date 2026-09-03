"""Determinism contract and shared fixtures for the equivalence harness.

Determinism rules applied to everything in this package:

* **CPU only.** No test here touches CUDA; configs pin ``hardware.device: cpu``
  and ``device: "cpu"``.
* ``torch.manual_seed`` / ``numpy.random.seed`` / ``random.seed`` all set.
* ``torch.use_deterministic_algorithms(True)``.
* ``num_workers=0`` in every DataLoader (via ``data.num_workers``).
* No AMP (``use_amp: False``).
* Fixed episode seed (``seed: 42``, matching the frozen paper eval configs).

The torch determinism switch is global process state, so ``_determinism`` is
``scope="package"``: it is entered when the first test in this package runs and
torn down when the last one finishes, so ``tests/unit`` and
``tests/integration`` never inherit the flag from a full-suite run (a
session-scoped fixture kept it on for the rest of the process - phase-7 gate
decision 7).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ._harness import SCENES, build_all_scenes, set_determinism

#: Thread count the goldens were generated at. Not recorded in ``meta.json``
#: (phase 2 did not know it mattered) and not expressible as a pip pin, but
#: load-bearing: BLAS reduction order follows it. Measured at the phase-8 gate
#: on the reference machine (64 cores, torch's default of 32) — the same clone
#: and the same torch gives 33 passed at 32 threads, 31 at 4, 31 at 2, 26 at 1.
REFERENCE_THREADS = 32


def _golden_env() -> dict:
    """The environment ``golden/meta.json`` records, or ``{}`` if unreadable."""
    meta_path = Path(__file__).parent / "golden" / "meta.json"
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):  # pragma: no cover - a broken checkout
        return {}


def _release(version: str) -> str:
    """``2.11.0+cu128`` -> ``2.11.0``.

    CPU and CUDA builds of the same torch release produce identical goldens —
    measured at the phase-8 gate: a fresh clone on ``torch==2.11.0+cpu``
    reproduces every numeric fingerprint. Only the build string differs, so the
    comparison is release-level.
    """
    return version.split("+", 1)[0]


def environment_mismatch() -> str | None:
    """Why this environment cannot reproduce the goldens, or ``None``.

    The goldens are bit-level fingerprints of float32 arithmetic, so they hold
    on the environment they were generated on and drift on any other. Rather
    than reporting that drift as a behaviour change — which is what a bare
    failure looks like, and what the phase-8 release gate first saw in a fresh
    clone — the package skips itself and says what differs.
    """
    meta = _golden_env()
    if not meta:
        return None  # nothing to compare against; let the tests speak
    reasons = []
    if _release(meta.get("torch", "")) != _release(torch.__version__):
        reasons.append(f"torch {torch.__version__} != goldens' {meta['torch']}")
    if _release(meta.get("numpy", "")) != _release(np.__version__):
        reasons.append(f"numpy {np.__version__} != goldens' {meta['numpy']}")
    threads = torch.get_num_threads()
    if threads != REFERENCE_THREADS:
        reasons.append(f"torch.get_num_threads() {threads} != reference {REFERENCE_THREADS}")
    if not reasons:
        return None
    return (
        "environment differs from the one golden/meta.json records, so these "
        "fingerprints cannot be reproduced here: " + "; ".join(reasons) + ". "
        'Reconstruct it with `pip install -e ".[dev]" -c constraints/verification.txt` '
        "on a machine with the reference thread count, or read this as "
        "'not checked here' — it is not evidence about the code. "
        "See tests/equivalence/README.md."
    )


# Tolerances. CPU deterministic runs are bit-exact on a fixed environment; the
# tolerance exists only to absorb BLAS reduction-order noise, not to paper over
# behaviour changes. Anything looser than this must be justified in a comment
# at the assertion site.
RTOL = 1e-6
ATOL = 1e-8


@pytest.fixture(scope="package", autouse=True)
def _reference_environment():
    """Skip the whole package when the environment cannot reproduce the goldens."""
    reason = environment_mismatch()
    if reason:
        pytest.skip(reason)


@pytest.fixture(scope="package", autouse=True)
def _determinism():
    previous = torch.are_deterministic_algorithms_enabled()
    set_determinism()
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.fixture(scope="session")
def scene_root(tmp_path_factory) -> Path:
    """All three mini-scenes written as ``.mat`` files under one data root.

    Session-scoped: the scenes are a deterministic function of their seeds, so
    one build serves every test.
    """
    root = tmp_path_factory.mktemp("mini_scenes") / "raw"
    build_all_scenes(root)
    return root


@pytest.fixture(scope="session")
def houston_mini():
    return SCENES["houston_mini"]

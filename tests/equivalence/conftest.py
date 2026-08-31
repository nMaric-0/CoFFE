"""Determinism contract and shared fixtures for the equivalence harness.

Determinism rules applied to everything in this package:

* **CPU only.** No test here touches CUDA; configs pin ``hardware.device: cpu``
  and ``device: "cpu"``.
* ``torch.manual_seed`` / ``numpy.random.seed`` / ``random.seed`` all set.
* ``torch.use_deterministic_algorithms(True)``.
* ``num_workers=0`` in every DataLoader (via ``data.num_workers``).
* No AMP (``use_amp: False``).
* Fixed episode seed (``seed: 42``, matching the frozen paper eval configs).

The torch determinism switch is global process state, so it is restored on
teardown: the rest of ``tests/`` must not inherit it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ._harness import SCENES, build_all_scenes, set_determinism

# Tolerances. CPU deterministic runs are bit-exact on a fixed environment; the
# tolerance exists only to absorb BLAS reduction-order noise, not to paper over
# behaviour changes. Anything looser than this must be justified in a comment
# at the assertion site.
RTOL = 1e-6
ATOL = 1e-8


@pytest.fixture(scope="session", autouse=True)
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

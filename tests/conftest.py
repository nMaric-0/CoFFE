"""Fixtures shared by ``tests/unit`` and ``tests/integration``.

Everything here is synthetic: no dataset under ``data/raw/`` and no HyperSIGMA
checkpoint is ever needed (PAPER_CANON §7.5). The synthetic-scene builders
are reused from the equivalence harness rather than re-implemented, so there is
exactly one description of the on-disk scene layout in the test tree
(``tests/equivalence/_harness.py``).

Nothing here is ``autouse``: the equivalence package keeps its own determinism
contract in ``tests/equivalence/conftest.py``, and these fixtures must not
change what that package sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.equivalence._harness import SCENES, SceneSpec, build_all_scenes

#: Scene keys in the order the paper's tables use them.
SCENE_KEYS = ("houston", "trento", "muufl")

#: PAPER_CANON §5 Table 1: HSI bands, aux channels, class count per scene.
#: Duplicated from the paper (not imported from the code) on purpose — this is
#: the constant the code is checked *against*.
CANON_DATASET_SPECS: dict[str, dict[str, int]] = {
    "houston": {"hsi_channels": 144, "aux_channels": 1, "num_classes": 15},
    "trento": {"hsi_channels": 63, "aux_channels": 1, "num_classes": 6},
    "muufl": {"hsi_channels": 64, "aux_channels": 2, "num_classes": 11},
}


def spec_for(key: str) -> SceneSpec:
    """The mini-scene spec whose shapes match the real scene ``key``."""
    return SCENES[f"{key}_mini"]


@pytest.fixture(scope="session")
def mini_scene_root(tmp_path_factory) -> Path:
    """All three shape-faithful mini-scenes written as ``.mat`` files.

    Session-scoped because the scenes are a deterministic function of their
    seeds. Writing them to disk (rather than building tensors) means the real
    ``PatchedMultimodalDataset`` loading, shape-normalising and min-max
    normalising code runs unmodified.
    """
    root = tmp_path_factory.mktemp("unit_scenes") / "raw"
    build_all_scenes(root)
    return root

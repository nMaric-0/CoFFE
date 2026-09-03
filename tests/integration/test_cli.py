"""The CLI surface: ``--help`` on every entry point, and one e2e per route.

Two jobs:

1. **Every ``argparse`` entry point under ``scripts/`` answers ``--help`` with
   exit 0 and writes nothing.** This was not free: ``compile_results.py`` and
   three report scripts used to *regenerate committed artifacts* on a bare
   invocation, which bit phases 5 and 6 before the phase-6 gate gave them
   ``--out``/``--force``. The discovery is by glob, so a new script cannot
   quietly skip the check.
2. **End-to-end smoke, pretrain -> checkpoint -> evaluate -> ``results.json``.**
   Two epochs and five episodes on a synthetic mini-scene, for both the CoFFE
   and MFT routes, each through its own entry point. It proves the pipeline is
   wired, not that any number is right — the equivalence harness owns the
   numbers. The HyperSIGMA route has no e2e here: it needs the released
   ViT-Base checkpoints (PAPER_CANON §7.5), and its wrapper plumbing is
   covered by ``tests/unit/test_hypersigma_contracts.py``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.equivalence._harness import SCENES, pretrain_config, write_scene

REPO = Path(__file__).resolve().parents[2]

#: Every ``scripts/`` module that builds an ``argparse`` parser, found by glob
#: so the list cannot go stale.
ENTRY_POINTS = sorted(
    path.relative_to(REPO).as_posix()
    for path in [
        *(REPO / "scripts").glob("*.py"),
        *(REPO / "scripts" / "reports").glob("*.py"),
        *(REPO / "scripts" / "reproduce").glob("*.py"),
    ]
    if "ArgumentParser" in path.read_text()
)

#: Committed artifact trees a ``--help`` must never touch.
PROTECTED_TREES = ("docs/presentation", "results", "experiments")


def _run(*args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _tree_digest(*relative: str) -> str:
    """A digest over every committed file under ``relative``.

    Used to prove a ``--help`` invocation wrote nothing, rather than trusting
    it not to.
    """
    digest = hashlib.sha256()
    for rel in relative:
        base = REPO / rel
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(REPO).as_posix().encode())
                digest.update(str(path.stat().st_size).encode())
                digest.update(str(path.stat().st_mtime_ns).encode())
    return digest.hexdigest()


def test_the_entry_point_list_is_not_empty() -> None:
    """Guard: a broken glob would make the parametrized test vacuous."""
    assert len(ENTRY_POINTS) >= 22, ENTRY_POINTS
    for expected in ("scripts/pretrain.py", "scripts/evaluate.py", "scripts/compile_results.py"):
        assert expected in ENTRY_POINTS


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_help_exits_zero(script: str) -> None:
    proc = _run(script, "--help")

    assert proc.returncode == 0, f"{script} --help failed:\n{proc.stderr[-2000:]}"
    assert "usage:" in proc.stdout


def test_help_writes_nothing_into_the_committed_artifact_trees() -> None:
    """``--help`` on all of them must leave every frozen artifact untouched.

    Run as one test over all entry points so the digest is taken once before
    and once after — the property is about the whole sweep.
    """
    before = _tree_digest(*PROTECTED_TREES)

    for script in ENTRY_POINTS:
        proc = _run(script, "--help")
        assert proc.returncode == 0, script

    assert _tree_digest(*PROTECTED_TREES) == before


# ----------------------------------------------------------------------
# End-to-end: pretrain -> evaluate
# ----------------------------------------------------------------------

E2E_EPOCHS = 2
E2E_EPISODES = 5
E2E_K_SHOT = 5  # PAPER_CANON §4
E2E_K_QUERY = 5


@pytest.fixture(scope="module")
def trento_mini_root(tmp_path_factory) -> Path:
    """The smallest mini-scene (6 classes, 63 bands), written to its own root."""
    root = tmp_path_factory.mktemp("e2e") / "raw"
    write_scene(SCENES["trento_mini"], root)
    return root


@pytest.fixture(scope="module")
def houston_mini_root(tmp_path_factory) -> Path:
    """The Houston-shaped mini-scene (15 classes, 144 bands + 1 aux)."""
    root = tmp_path_factory.mktemp("e2e_houston") / "raw"
    write_scene(SCENES["houston_mini"], root)
    return root


def _write_config(tmp_path: Path, data_root: Path, model_name: str) -> tuple[Path, Path]:
    """A 2-epoch pretrain config, built from the harness's paper-shaped recipe.

    Only the schedule is scaled down; the arch, mask rates, loss and optimiser
    come from ``tests/equivalence/_harness.pretrain_config``, i.e. from the
    frozen paper run configs.
    """
    config = pretrain_config(
        SCENES["trento_mini"],
        "simmim_token",
        data_root=data_root,
        model_name=model_name,
        epochs=E2E_EPOCHS,
        batch_size=16,
    )
    config["pretrain"]["save_interval"] = 1  # so checkpoint_epoch_2.pth exists
    checkpoint_dir = tmp_path / "checkpoints"
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    config["paths"]["log_dir"] = str(tmp_path / "logs")

    config_path = tmp_path / f"{model_name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, checkpoint_dir


def _assert_results_json(results: dict, model_type: str) -> None:
    """The protocol fields a written ``results.json`` must carry.

    Note the written key is ``OA`` — a ``{mean, std, ci_95}`` block — not a
    bare ``oa`` scalar. Accuracy is checked only against chance: this is a
    smoke test on a synthetic scene, and the equivalence harness owns numbers.
    """
    assert results["model_type"] == model_type
    assert results["distance_metric"] == "euclidean"
    assert results["dataset"] == "trento"
    assert results["n_way"] == 6  # Trento's full class count (PAPER_CANON §5)
    assert results["k_shot"] == E2E_K_SHOT
    assert results["num_episodes"] == E2E_EPISODES

    oa = results["OA"]["mean"]
    assert 0.0 <= oa <= 100.0
    assert oa > 100.0 / 6, f"OA {oa} does not beat chance on the mini-scene"
    # OA == AA by construction: the queries are class-balanced (PAPER_CANON §4).
    assert results["AA"]["mean"] == pytest.approx(oa, abs=1e-9)


def _pretrain(tmp_path: Path, data_root: Path, model_name: str) -> Path:
    """Run ``scripts/pretrain.py`` for two epochs; return the epoch-2 checkpoint."""
    config_path, checkpoint_dir = _write_config(tmp_path, data_root, model_name)

    pretrain = _run("scripts/pretrain.py", "--config", str(config_path), timeout=900)
    assert pretrain.returncode == 0, pretrain.stderr[-3000:]

    checkpoint = checkpoint_dir / f"checkpoint_epoch_{E2E_EPOCHS}.pth"
    assert checkpoint.exists(), sorted(p.name for p in checkpoint_dir.glob("*"))
    assert (checkpoint_dir / "final.pth").exists()
    assert (checkpoint_dir / "encoder_final.pth").exists()
    return checkpoint


@pytest.mark.slow
def test_coffe_pretrain_then_evaluate_end_to_end(tmp_path: Path, trento_mini_root: Path) -> None:
    """The headline route through both CLIs: two epochs, then a 5-episode eval."""
    checkpoint = _pretrain(tmp_path, trento_mini_root, "coffe")
    results_path = tmp_path / "results.json"

    evaluate = _run(
        "scripts/evaluate.py",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        "trento",
        "--data-root",
        str(trento_mini_root),
        "--split",
        "all",
        "--embed-dim",
        "128",
        "--num-heads",
        "2",
        "--num-layers",
        "2",
        "--lambda-factor",
        "0.5",
        "--k-shot",
        str(E2E_K_SHOT),
        "--k-query",
        str(E2E_K_QUERY),
        "--num-episodes",
        str(E2E_EPISODES),
        "--distance-metric",
        "euclidean",
        "--no-projection",
        "--device",
        "cpu",
        "--no-plots",
        "--output",
        str(results_path),
        timeout=900,
    )
    assert evaluate.returncode == 0, evaluate.stderr[-3000:]

    assert results_path.exists()
    _assert_results_json(json.loads(results_path.read_text()), "CoFFE")


@pytest.mark.slow
def test_mft_control_pretrain_then_evaluate_end_to_end(
    tmp_path: Path, trento_mini_root: Path
) -> None:
    """The control route through its own pipeline, ``scripts/evaluate_mft.py``.

    Until the phase-7 gate the control was not reachable from any CLI:
    ``scripts/evaluate.py`` exposed no ``--name``, so ``mft_original`` could
    only be selected programmatically. The gate's answer was one entry point
    per route rather than a route flag, so the control now has its own CLI
    carrying the faithful MFT defaults (embed 64 / 8 heads / 2 layers / mlp
    512), and needs no architecture flags here to reproduce them.
    """
    checkpoint = _pretrain(tmp_path, trento_mini_root, "mft_original")
    results_path = tmp_path / "results.json"

    evaluate = _run(
        "scripts/evaluate_mft.py",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        "trento",
        "--data-root",
        str(trento_mini_root),
        "--split",
        "all",
        "--k-shot",
        str(E2E_K_SHOT),
        "--k-query",
        str(E2E_K_QUERY),
        "--num-episodes",
        str(E2E_EPISODES),
        "--device",
        "cpu",
        "--no-plots",
        "--output",
        str(results_path),
        timeout=900,
    )
    assert evaluate.returncode == 0, evaluate.stderr[-3000:]

    assert results_path.exists()
    _assert_results_json(json.loads(results_path.read_text()), "MFTOriginal")


def test_each_route_has_its_own_entry_point() -> None:
    """One CLI per route, each defaulting to that route's published arch.

    The three routes the paper compares must not share a single CLI whose
    defaults can only be right for one of them (phase-7 gate decision). The
    architecture defaults are read off each parser rather than off a literal,
    so a drifted default fails here.
    """
    routes = {
        "scripts/evaluate.py": ("CoFFE", {"--embed-dim": "128", "--num-heads": "2"}),
        "scripts/evaluate_mft.py": ("MFT", {"--embed-dim": "64", "--num-heads": "8"}),
        "scripts/evaluate_hypersigma.py": ("HyperSIGMA", {}),
    }
    for script in routes:
        assert (REPO / script).exists(), script
        assert script in ENTRY_POINTS

    # The control's CLI selects the route itself instead of exposing a flag.
    mft = (REPO / "scripts" / "evaluate_mft.py").read_text()
    assert 'args.name = "mft_original"' in mft
    assert "--name" not in _run("scripts/evaluate_mft.py", "--help").stdout

    # And CoFFE's no longer claims to evaluate the control.
    coffe_help = _run("scripts/evaluate.py", "--help").stdout
    assert "MFT" not in coffe_help


@pytest.mark.slow
def test_evaluate_rejects_a_checkpoint_from_the_wrong_scene(
    tmp_path: Path, trento_mini_root: Path
) -> None:
    """A band-count mismatch must fail loudly, not silently random-init.

    A 63-band Trento checkpoint pointed at 144-band Houston: the input
    projection cannot load, and the evaluator refuses rather than reporting a
    number produced by a partly random encoder.
    """
    config_path, checkpoint_dir = _write_config(tmp_path, trento_mini_root, "coffe")
    assert _run("scripts/pretrain.py", "--config", str(config_path), timeout=900).returncode == 0

    houston_root = tmp_path / "houston_raw"
    write_scene(SCENES["houston_mini"], houston_root)

    proc = _run(
        "scripts/evaluate.py",
        "--checkpoint",
        str(checkpoint_dir / f"checkpoint_epoch_{E2E_EPOCHS}.pth"),
        "--dataset",
        "houston",
        "--data-root",
        str(houston_root),
        "--split",
        "all",
        "--embed-dim",
        "128",
        "--num-heads",
        "2",
        "--num-layers",
        "2",
        "--no-projection",
        "--device",
        "cpu",
        "--no-plots",
        timeout=900,
    )

    assert proc.returncode != 0
    assert "Band-count mismatch" in (proc.stderr + proc.stdout)


@pytest.mark.slow
def test_an_hsi_only_checkpoint_is_told_which_flag_to_pass(
    tmp_path: Path, houston_mini_root: Path
) -> None:
    """The HSI-only cells need ``--no-aux``, and the error has to say so.

    Phase-8 gate: ``configs/coffe/*_hsi.yaml`` pretrain with ``use_aux: false``
    while the CLI defaults to ``--use-aux``, so the generic recipe fails on the
    12 HSI-only Table 2 cells with a band-count mismatch. The message used to
    send the reader to ``--dataset``, which is usually already right.
    """
    config = pretrain_config(
        SCENES["houston_mini"],
        "simmim_token",
        data_root=houston_mini_root,
        model_name="coffe",
        epochs=E2E_EPOCHS,
        batch_size=16,
    )
    config["model"]["use_aux"] = False
    config["pretrain"]["save_interval"] = 1
    checkpoint_dir = tmp_path / "checkpoints"
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    config["paths"]["log_dir"] = str(tmp_path / "logs")
    config_path = tmp_path / "coffe_hsi.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert _run("scripts/pretrain.py", "--config", str(config_path), timeout=900).returncode == 0
    checkpoint = checkpoint_dir / f"checkpoint_epoch_{E2E_EPOCHS}.pth"

    common = [
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        "houston",
        "--data-root",
        str(houston_mini_root),
        "--split",
        "all",
        "--k-shot",
        "2",
        "--k-query",
        "2",
        "--num-episodes",
        "2",
        "--device",
        "cpu",
        "--no-plots",
    ]

    # Default (--use-aux) builds a 145-band model for a 144-band checkpoint.
    failed = _run("scripts/evaluate.py", *common, timeout=900)
    output = failed.stderr + failed.stdout
    assert failed.returncode != 0
    assert "Band-count mismatch" in output
    assert "--no-aux" in output, output[-2000:]

    # And with the flag the message names, it evaluates.
    ok = _run("scripts/evaluate.py", *common, "--no-aux", timeout=900)
    assert ok.returncode == 0, ok.stderr[-3000:]

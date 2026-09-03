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
   and MFT routes. It proves the pipeline is wired, not that any number is
   right — the equivalence harness owns the numbers. The HyperSIGMA route has
   no e2e here: it needs the released ViT-Base checkpoints (CLAUDE.md hard
   rule 5), and its wrapper plumbing is covered by
   ``tests/unit/test_hypersigma_contracts.py``.
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
    assert len(ENTRY_POINTS) >= 20, ENTRY_POINTS
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
    """The MFT control, pretrained by CLI and evaluated programmatically.

    ``scripts/evaluate.py`` exposes **no ``--name`` and no ``--mlp-dim``
    flag**, so the control is not selectable from that CLI even though the
    script's docstring says it evaluates "CoFFE and the MFT architectural
    control" — ``coffe.eval.episodic.main`` reads both keys off the namespace,
    but the parser never adds them. Every paper MFT eval went through
    ``coffe.runners.eval_runner``, which reads ``model.name`` from the run's
    frozen ``pretrain_config.yaml``, so no published number is affected.

    Reported at the phase-7 gate rather than fixed: adding a flag changes the
    CLI surface. This test therefore drives the same programmatic entry point
    the reproduce pipeline uses. If the flags are added, fold this back into
    the CLI test above.
    """
    from coffe.eval.episodic import run_evaluation

    checkpoint = _pretrain(tmp_path, trento_mini_root, "mft_original")
    results_path = tmp_path / "results.json"

    run_evaluation(
        checkpoint=str(checkpoint),
        dataset="trento",
        name="mft_original",
        data_root=str(trento_mini_root),
        split="all",
        embed_dim=64,
        num_heads=8,
        num_layers=2,
        mlp_dim=512,
        attention_type="mcross",
        k_shot=E2E_K_SHOT,
        k_query=E2E_K_QUERY,
        num_episodes=E2E_EPISODES,
        distance_metric="euclidean",
        use_projection=False,
        device="cpu",
        no_plots=True,
        output=str(results_path),
    )

    assert results_path.exists()
    _assert_results_json(json.loads(results_path.read_text()), "MFTOriginal")


def test_evaluate_cli_cannot_select_the_mft_control() -> None:
    """Pins the gap above, so it is visible rather than folklore.

    Asserted on the parser's own help text: neither flag exists today. If
    ``--name`` is added, this test should be deleted in the same change.
    """
    proc = _run("scripts/evaluate.py", "--help")

    assert proc.returncode == 0
    assert "--name" not in proc.stdout
    assert "--mlp-dim" not in proc.stdout
    # The docstring that the missing flags contradict.
    assert "MFT" in Path(REPO / "scripts" / "evaluate.py").read_text()


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

"""Config parity for the post-paper ablation configs (``configs/ablation/``).

``tests/unit/test_config_parity.py`` holds the paper's config set to
PAPER_CANON: exactly 30 files under ``configs/coffe/``, every one of them
``model.name: "coffe"``. The ablation's recipes must therefore live in their
own directory — and get their own parity test, because an unpinned config is
how a "controlled" comparison quietly stops being controlled.

What is pinned here:

* the encoder geometry is still PAPER_CANON §2 (the ablation changes fusion,
  nothing else);
* the mask rates are the canonical regimes, and the auxiliary-token mask
  probability is the documented 0.5;
* the schedules match the 700-epoch significance protocol the controls were
  run under (PAPER_CANON §8 D17 / §9), so an arm is comparable to its control;
* the paper's own config set is untouched by all of the above.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.reproduce import sig_significance_config as sig

REPO = Path(__file__).resolve().parents[2]
ABLATION_CONFIGS = sorted((REPO / "configs" / "ablation").glob("*.yaml"))

#: PAPER_CANON §2 — kept by the ablation.
PATCH_SIZE = 11
EMBED_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 2

#: The documented auxiliary-token mask probability
#: (docs/ablations/AUX_TOKEN_ABLATION.md D-B).
AUX_MASK_PROB = 0.5

#: The significance protocol the controls ran under.
EPOCHS = 700
SAVE_INTERVAL = 100

#: The two ablation groups and the cell each one trains.
ABLATION_GROUPS = {
    "aux_token_simmim": ("simmim_token", "houston_aux_token_simmim_token"),
    "aux_token_mae": ("mae", "houston_aux_token_mae"),
}


def _load(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _stem(path: Path) -> str:
    return path.stem


def test_the_config_set_is_the_one_this_file_thinks_it_is() -> None:
    """Guard against the parametrized tests below silently covering nothing."""
    assert [p.stem for p in ABLATION_CONFIGS] == [
        "houston_aux_token_mae",
        "houston_aux_token_simmim_token",
    ]


@pytest.mark.parametrize("path", ABLATION_CONFIGS, ids=_stem)
def test_the_ablation_keeps_the_paper_encoder(path: Path) -> None:
    """Only the fusion mechanism differs: D = 128, 2 heads, 2 layers, 11x11."""
    cfg = _load(path)

    assert cfg["data"]["patch_size"] == PATCH_SIZE
    model = cfg["model"]
    assert model["name"] == "coffe_aux_token"
    assert model["embed_dim"] == EMBED_DIM
    assert model["num_heads"] == NUM_HEADS
    assert model["num_layers"] == NUM_LAYERS
    assert model["lambda_factor"] == 0.5
    # The auxiliary token is the route's whole point.
    assert model["use_aux"] is True


@pytest.mark.parametrize("path", ABLATION_CONFIGS, ids=_stem)
def test_the_ablation_runs_the_significance_schedule(path: Path) -> None:
    """700 epochs with an epoch-700 checkpoint — the controls' protocol.

    The 5-seed runner overrides both values anyway; stating them in the config
    means a single ``python scripts/pretrain.py --config ...`` reproduces one
    seed of the same cell.
    """
    pretrain = _load(path)["pretrain"]

    assert pretrain["epochs"] == EPOCHS
    assert pretrain["save_interval"] == SAVE_INTERVAL
    assert EPOCHS % pretrain["save_interval"] == 0
    assert pretrain["datasets"] == ["houston"]
    assert pretrain["val_split"] == 0.0


@pytest.mark.parametrize("path", ABLATION_CONFIGS, ids=_stem)
def test_the_auxiliary_token_is_masked_and_reconstructed(path: Path) -> None:
    """D-B: the aux token is masked per sample at 0.5, weight 1.0.

    ``aux_loss_weight: 1.0`` is what makes the loss the plain union masked mean
    of PAPER_CANON §3 Eq. 1 generalised over the two heads; a config that
    changed it would change the objective without changing its name.
    """
    pretrain = _load(path)["pretrain"]

    assert pretrain["aux_mask_prob"] == AUX_MASK_PROB
    assert pretrain["aux_loss_weight"] == 1.0


def test_the_simmim_cell_carries_the_canonical_token_regime() -> None:
    """SimMIM token = ``(band, spatial) == (0.0, 0.75)`` (PAPER_CANON §1).

    Band masking is not merely unused here — the route rejects it — so 0.0 is
    the only admissible value.
    """
    pretrain = _load(REPO / "configs" / "ablation" / "houston_aux_token_simmim_token.yaml")[
        "pretrain"
    ]

    assert pretrain["objective"] == "simmim"
    assert (pretrain["band_mask_ratio"], pretrain["spatial_mask_ratio"]) == (0.0, 0.75)
    assert pretrain["recon_center_sigma"] == 1.0
    # The control's schedule (experiments/houston_enhanced_spatial_seed42).
    assert pretrain["batch_size"] == 128
    assert pretrain["lr"] == 1.5e-5
    assert pretrain["warmup_epochs"] == 100


def test_the_mae_cell_carries_the_canonical_token_drop_rate() -> None:
    """MAE = 0.75 token drop, plain loss, no projection head (PAPER_CANON §1/§3)."""
    cfg = _load(REPO / "configs" / "ablation" / "houston_aux_token_mae.yaml")
    pretrain = cfg["pretrain"]

    assert pretrain["objective"] == "mae"
    assert pretrain["mask_ratio"] == 0.75
    assert pretrain["band_mask_ratio"] == 0.0
    assert pretrain["recon_center_sigma"] is None
    assert pretrain["norm_pix_loss"] is True
    assert cfg["model"]["use_projection"] is False
    # The control's schedule (experiments/houston_enhanced_mae_lidar_seed42).
    assert pretrain["batch_size"] == 64
    assert pretrain["lr"] == 1.5e-4
    assert (pretrain["decoder_dim"], pretrain["decoder_depth"], pretrain["decoder_heads"]) == (
        64,
        4,
        4,
    )


@pytest.mark.parametrize("group", sorted(ABLATION_GROUPS))
def test_each_ablation_group_points_at_an_existing_houston_config(group: str) -> None:
    """The 5-seed runner's cells resolve, and cover Houston only."""
    variant, stem = ABLATION_GROUPS[group]
    g = sig.GROUPS[group]

    assert g.datasets == ["houston"]
    assert g.variants == [variant]
    assert g.clone_mask is False  # self-contained recipe, nothing cloned
    base = g.base_config("houston", variant)
    assert base == f"configs/ablation/{stem}.yaml"
    assert (REPO / base).exists()
    assert [name for *_, name in sig.runs([group])] == [f"{stem}_seed{seed}" for seed in sig.SEEDS]


def test_the_ablation_groups_are_opt_in() -> None:
    """A bare significance run must still be exactly the paper's five groups.

    ``GROUP_ORDER`` is what ``cells()`` and the aggregator default to, so
    leaving the ablation out of it is what keeps ``--stage all`` with no
    ``--groups`` reproducing the paper and nothing else.
    """
    assert sig.GROUP_ORDER == ["enhanced", "hsi_only", "enhanced_mae", "mft_mae", "mft_spatial"]
    assert set(ABLATION_GROUPS) <= set(sig.GROUPS)
    assert not set(ABLATION_GROUPS) & set(sig.GROUP_ORDER)
    # 5 paper groups x 3 scenes x their variants, unchanged by the additions.
    assert len(list(sig.cells())) == 30


def test_the_papers_config_set_is_untouched() -> None:
    """The ablation lives outside ``configs/coffe/`` — 30 files, all "coffe".

    Duplicated from ``test_config_parity.py`` on purpose: this file is the one
    that would have been tempted to add a 31st.
    """
    coffe_configs = sorted((REPO / "configs" / "coffe").glob("*.yaml"))

    assert len(coffe_configs) == 30
    for path in coffe_configs:
        assert _load(path)["model"]["name"] == "coffe"

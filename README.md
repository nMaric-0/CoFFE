# CoFFE — A Compact In-Domain Fusion Encoder vs. a Hyperspectral Foundation Model

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11+](https://img.shields.io/badge/pytorch-2.11+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Public code release for *"A Compact In-Domain Fusion Encoder versus a
Hyperspectral Foundation Model for Few-Shot HSI-LiDAR Land-Cover
Classification"* (Marić & Kocev). Two pipelines:

1. **Per-scene masked pretraining** — SimMIM-style in-place masking (band
   and/or spatial-token) or token-drop MAE over concatenated HSI + auxiliary
   (LiDAR) bands, with optional centre-weighted reconstruction loss. Produces a
   frozen `CoFFE` encoder; the same script pretrains the `MFTOriginal`
   architectural control.
2. **Few-shot evaluation** — parameter-free **Euclidean nearest-class-mean**
   over the frozen encoder: N-way (all classes in the scene), 5-shot, 1000
   episodes.

A third route, label-free adaptation of the HyperSIGMA foundation model, lives
in `scripts/adapt_hypersigma.py` + `scripts/evaluate_hypersigma.py` and is
evaluated by the same protocol.

Everything you run is logged into a structured `experiments/<name>/` tree
with frozen configs, checkpoints, per-evaluation results, and plots —
designed for tuning, comparing, and revisiting runs from notebooks.

## Quick start

```bash
git clone <repo-url> coffe
cd coffe
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e .
```

That installs the `coffe` package (and the vendored HyperSIGMA sources under
`third_party/`), so `import coffe` works from any working directory.
`requirements.txt` is a one-line mirror of `pyproject.toml` (`-e .[dev]`), so
`pip install -r requirements.txt` does the same thing plus the test and lint
tools. Every entry point under `scripts/` adds the repo root to `sys.path`
itself, so `python scripts/…` runs with or without the install; only
`import coffe` from an unrelated directory needs it.

Optional extras: `.[tensorboard]` for pretraining scalars, `.[notebooks]` for
the Jupyter workflow below, `.[dev]` for pytest + ruff + mypy.

Then either:

- **From notebooks (recommended for research):** open
  `notebooks/pretrain.ipynb`, set the Parameters cell, run all. Then
  `notebooks/evaluate.ipynb` to evaluate the resulting checkpoint, and
  `notebooks/compare.ipynb` to compare across runs.
- **From the CLI (one-off runs):** `python scripts/pretrain.py
  --config configs/coffe/houston_simmim.yaml`, then
  `./scripts/reproduce/run_eval.sh <checkpoint> houston 15 5`.

## Experiment layout

Every notebook-driven run writes to `experiments/<experiment_name>/`:

```
experiments/<experiment_name>/
├── README.md                       name + short description (you write at start)
├── pretrain_config.yaml            frozen training config
├── pretrain_metadata.json          timestamp, git SHA, final/best loss, wallclock
├── pretrain.log
├── checkpoints/checkpoint_epoch_*.pth
└── evaluations/<eval_name>/
    ├── eval_config.json            every parameter passed to the evaluator
    ├── eval_metadata.json          timestamp, git SHA, OA/AA/Kappa summary
    ├── results.json                full results including per-class accuracy
    ├── eval.log
    └── plots/                      confusion matrix, per-class accuracy, t-SNE
```

See `experiments/_example/` for the schema (checked-in placeholder, not a real run).

`experiments/<name>/` directories are gitignored by default; commit the ones
you want to share by force-adding the relevant files.

## Programmatic API (used by the notebooks)

```python
from coffe.runners.pretrain_runner import run_pretrain
from coffe.runners.eval_runner import run_evaluation
from coffe.runners.experiments import ExperimentLogger, load_all_evaluations

# 1. Pretrain
exp = run_pretrain(
    name="houston_coffe_simmim_token_hsi_lidar_seed42",
    description="Houston CoFFE pretrain, SimMIM token regime, 2-layer 2-head encoder.",
    config="configs/coffe/houston_simmim.yaml",
    overrides={"pretrain": {"lr": 3e-4}},  # optional deep-merge
)

# 2. Evaluate (loads the latest checkpoint from the experiment)
ev = run_evaluation(
    experiment_name="houston_coffe_simmim_token_hsi_lidar_seed42",
    eval_name="houston_15way_5shot_euclidean",
    eval_params={
        # n_way defaults to the scene's full class count, which is the paper
        # protocol; use_projection is off at eval.
        "dataset": "houston",
        "k_shot": 5,
        "k_query": 100,
        "num_episodes": 1000,
        "distance_metric": "euclidean",
        "use_projection": False,
    },
)

# 3. Compare
import pandas as pd  # ships with the `.[notebooks]` extra

df = pd.DataFrame(load_all_evaluations())
```

## Modifying the model / pretrain / eval pipelines

- **Model**: `coffe/models/coffe.py` (encoder used by both pipelines).
  Encoder hyperparameters (depth, heads, projection head, pooling) are read
  from the YAML config under `model:`.
- **Pretraining**: `coffe/pretrain/simmim.py` (SimMIM masking + loss) or
  `coffe/pretrain/mae_pretrain.py` (MAE). Mask ratios, decoder hidden dim, and recon
  weighting live under `pretrain:` in the config.
- **Evaluation**: `coffe/eval/episodic.py` holds the live episode loop and the
  class-mean maths; `CoFFE.eval_patch_embeddings` produces the per-token
  features it pools. Distance metric, temperature, and class-mean mode are
  passed as eval params at runtime.

## Project structure

```
CoFFE/
├── coffe/                   # the installable package (`pip install -e .`)
│   ├── models/              # CoFFE + MFTOriginal + components/ + hypersigma/
│   ├── pretrain/            # SimMIM / MAE masked modelling, trainer, HyperSIGMA adaptation
│   ├── data/                # dataset loaders (datasets/) + episode samplers (samplers/)
│   ├── eval/                # the frozen-encoder episodic evaluators
│   ├── runners/             # notebook-friendly pretrain / eval / adapt entry points
│   ├── utils/               # seeding, IO, metrics, spatial weights, plots
│   └── compat.py            # legacy-name aliases for pre-rename artifacts
├── scripts/                 # thin CLIs: pretrain, evaluate, adapt, fit-PCA, compile-results
│   ├── reproduce/           # the experiment drivers behind Tables 2 and 3
│   └── reports/             # provenance / aggregation builders
├── configs/{coffe,mft,hypersigma}/   # per-cell pretraining, adaptation and eval configs
├── results/                 # the JSONs behind the paper tables (see results/README.md)
├── experiments/             # per-run output trees (gitignored except _example/)
├── notebooks/               # pretrain.ipynb, evaluate.ipynb, compare.ipynb, …
├── tests/                   # unit tests + tests/equivalence/ (behaviour goldens)
├── third_party/HyperSIGMA/  # vendored upstream (see its LICENSE / NOTICE)
└── docs/                    # pipeline documentation
```

## Supported Datasets

| Dataset | HSI Bands | Auxiliary | Classes |
|---------|-----------|-----------|---------|
| Houston | 144       | LiDAR (1) | 15      |
| Trento  | 63        | LiDAR (1) | 6       |
| MUUFL   | 64        | LiDAR (2) | 11      |

## Further reading

- [`docs/PRETRAINING.md`](docs/PRETRAINING.md) — pretraining objectives,
  masking regimes, and configuration.
- [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) — the frozen-encoder
  Euclidean nearest-class-mean protocol.
- [`PAPER_CANON.md`](PAPER_CANON.md) — names, protocol constants, results
  tables, and the known paper↔code discrepancies.
- [`CHANGES.md`](CHANGES.md) — the old→new name table and what still reads the
  old names.

## License

MIT — see [LICENSE](LICENSE).

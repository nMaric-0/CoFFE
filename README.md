# CoFFE — A Compact In-Domain Fusion Encoder vs. a Hyperspectral Foundation Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)
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
conda create -n coffe python=3.9 -y && conda activate coffe
pip install -r requirements.txt
```

Everything runs from the repository root (`python scripts/...`, notebooks from
`notebooks/`); there is no editable install step — real packaging metadata lands
in `pyproject.toml` in phase 6.

Then either:

- **From notebooks (recommended for research):** open
  `notebooks/pretrain.ipynb`, set the Parameters cell, run all. Then
  `notebooks/evaluate.ipynb` to evaluate the resulting checkpoint, and
  `notebooks/compare.ipynb` to compare across runs.
- **From the CLI (one-off runs):** `python scripts/pretrain.py
  --config configs/coffe/houston_simmim.yaml`, then
  `./scripts/run_eval.sh <checkpoint> houston 15 5`.

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
from lib.pretrain_runner import run_pretrain
from lib.eval_runner import run_evaluation
from lib.experiments import ExperimentLogger, load_all_evaluations

# 1. Pretrain
exp = run_pretrain(
    name="houston_coffe_simmim_token_hsi_lidar_seed42",
    description="Houston CoFFE pretrain, SimMIM token regime, 2-layer 2-head encoder.",
    config="configs/coffe/houston_simmim.yaml",
    overrides={"pretrain": {"lr": 3e-4}},   # optional deep-merge
)

# 2. Evaluate (loads the latest checkpoint from the experiment)
ev = run_evaluation(
    experiment_name="houston_coffe_simmim_token_hsi_lidar_seed42",
    eval_name="houston_15way_5shot_euclidean",
    eval_params={
        # n_way defaults to the scene's full class count, which is the paper
        # protocol; use_projection is off at eval.
        "dataset": "houston", "k_shot": 5, "k_query": 100,
        "num_episodes": 1000, "distance_metric": "euclidean",
        "use_projection": False,
    },
)

# 3. Compare
import pandas as pd
df = pd.DataFrame(load_all_evaluations())
```

## Modifying the model / pretrain / eval pipelines

- **Model**: `models/coffe.py` (encoder used by both pipelines).
  Encoder hyperparameters (depth, heads, projection head, pooling) are read
  from the YAML config under `model:`.
- **Pretraining**: `pretrain/simmim.py` (SimMIM masking + loss) or
  `pretrain/mae_pretrain.py` (MAE). Mask ratios, decoder hidden dim, and recon
  weighting live under `pretrain:` in the config.
- **Evaluation**: `scripts/evaluate.py` holds the live episode loop and the
  class-mean maths; `CoFFE.eval_patch_embeddings` produces the per-token
  features it pools. Distance metric, temperature, and class-mean mode are
  passed as eval params at runtime.

## Project structure

```
coffe/
├── configs/coffe/           # CoFFE pretraining configs (per scene / regime)
├── configs/mft/             # the MFT architectural control
├── configs/hypersigma/      # HyperSIGMA label-free adaptation
├── data/                    # Dataset loaders + episode sampler
├── docs/                    # Pipeline documentation
├── experiments/             # Per-run output trees (gitignored except _example/)
├── lib/                     # Research utilities
│   ├── experiments.py       # ExperimentLogger / PretrainExperiment / EvalRun
│   ├── pretrain_runner.py   # Notebook-friendly pretrain entry point
│   └── eval_runner.py       # Notebook-friendly eval entry point
├── models/                  # CoFFE + MFTOriginal + components
├── notebooks/               # pretrain.ipynb, evaluate.ipynb, compare.ipynb
├── pretrain/                # SimMIM / MAE masked-modelling pretraining
├── scripts/                 # CLI entry points + shell wrappers
├── tests/                   # unit tests + tests/equivalence/ (behaviour goldens)
├── third_party/HyperSIGMA/  # vendored upstream (see its LICENSE / NOTICE)
├── trainers/pretrain_trainer.py
├── coffe_compat.py          # legacy-name aliases for pre-rename artifacts
└── utils/
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

# MFT-CPEA: Enhanced Pretraining + Cosine Few-Shot Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Research-oriented fork of MFT-CPEA. Two pipelines:

1. **Enhanced pretraining** — unified masked autoencoder over concatenated
   HSI + auxiliary (LiDAR / SAR) bands with optional center-weighted spatial
   reconstruction loss. Produces an `MFTCPEACosine` encoder.
2. **Cosine few-shot evaluation** — parameter-free prototypical-network eval
   with cosine or Euclidean similarity over a pretrained encoder.

Everything you run is logged into a structured `experiments/<name>/` tree
with frozen configs, checkpoints, per-evaluation results, and plots —
designed for tuning, comparing, and revisiting runs from notebooks.

## Quick start

```bash
git clone <repo-url> mft-cpea
cd mft-cpea
conda create -n mft-cpea python=3.9 -y && conda activate mft-cpea
pip install -r requirements.txt
pip install -e .
```

Then either:

- **From notebooks (recommended for research):** open
  `notebooks/pretrain.ipynb`, set the Parameters cell, run all. Then
  `notebooks/evaluate.ipynb` to evaluate the resulting checkpoint, and
  `notebooks/compare.ipynb` to compare across runs.
- **From the CLI (one-off runs):** `python scripts/pretrain_enhanced.py
  --config configs/pretrain/houston_pretrain_enhanced.yaml`, then
  `./scripts/run_cosine_eval.sh <checkpoint> houston 5 5`.

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
    name="houston_enhanced_v1",
    description="Baseline Houston pretrain, 2-layer 2-head encoder.",
    config="configs/pretrain/houston_pretrain_enhanced.yaml",
    overrides={"pretrain": {"lr": 3e-4}},   # optional deep-merge
)

# 2. Evaluate (loads the latest checkpoint from the experiment)
ev = run_evaluation(
    experiment_name="houston_enhanced_v1",
    eval_name="houston_5way_5shot_cosine_t10",
    eval_params={
        "dataset": "houston", "n_way": 5, "k_shot": 5,
        "distance_metric": "cosine", "temperature": 10.0,
    },
)

# 3. Compare
import pandas as pd
df = pd.DataFrame(load_all_evaluations())
```

## Modifying the model / pretrain / eval pipelines

- **Model**: `models/mft_cpea_cosine.py` (encoder used by both pipelines).
  Encoder hyperparameters (depth, heads, projection head, pooling) are read
  from the YAML config under `model:`.
- **Pretraining**: `pretrain/masked_modeling_enhanced.py` (loss + masking
  strategy). Mask ratios, decoder hidden dim, and recon weighting live under
  `pretrain:` in the config.
- **Evaluation**: `models/mft_cpea_cosine.py:MFTCPEACosine.forward_episode`
  (prototype matching). Distance metric, temperature, and prototype mode are
  passed as eval params at runtime.

## Project structure

```
mft-cpea/
├── configs/pretrain/        # Enhanced pretraining configs (Houston, Trento)
├── data/                    # Dataset loaders + episode sampler
├── docs/                    # Pipeline documentation
├── experiments/             # Per-run output trees (gitignored except _example/)
├── lib/                     # Research utilities
│   ├── experiments.py       # ExperimentLogger / PretrainExperiment / EvalRun
│   ├── pretrain_runner.py   # Notebook-friendly pretrain entry point
│   └── eval_runner.py       # Notebook-friendly eval entry point
├── models/                  # MFTCPEACosine + backbones + components
├── notebooks/               # pretrain.ipynb, evaluate.ipynb, compare.ipynb
├── pretrain/                # Unified masked-modeling pretraining
├── scripts/                 # CLI entry points + shell wrappers
├── tests/
├── trainers/pretrain_trainer.py
└── utils/
```

## Supported Datasets

| Dataset | HSI Bands | Auxiliary | Classes |
|---------|-----------|-----------|---------|
| Houston | 144       | LiDAR (1) | 15      |
| Trento  | 63        | LiDAR (1) | 6       |
| MUUFL   | 64        | LiDAR (2) | 11      |

## Further reading

- [`docs/ENHANCED_PRETRAINING.md`](docs/ENHANCED_PRETRAINING.md) — pretraining
  objectives, masking strategy, and configuration.
- [`docs/COSINE_VARIANT.md`](docs/COSINE_VARIANT.md) — cosine evaluation
  architecture and rationale.
- [`SPLIT.md`](SPLIT.md) — how this tree was extracted from the parent repo.

## License

MIT — see [LICENSE](LICENSE).

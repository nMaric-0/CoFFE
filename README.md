# A Compact In-Domain Fusion Encoder versus a Hyperspectral Foundation Model for Few-Shot HSI-LiDAR Land-Cover Classification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11+](https://img.shields.io/badge/pytorch-2.11+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Nikola Marić, Dragi Kocev — Jožef Stefan Institute / IPS Ljubljana.
Paper: `TODO(release)` · code release for **TODO(release): venue**.

A 579K-parameter transformer encoder, pretrained per scene with no labels on the
scene it will be tested on, is compared against a 180M-parameter hyperspectral
foundation model (HyperSIGMA) under one frozen-encoder few-shot protocol. The
compact encoder — **CoFFE**, which fuses HSI and LiDAR at the input rather than
through an external token — wins on every scene, and the decisive margins come
from the LiDAR fusion: **+7.8 OA on Houston** and **+13.0 on MUUFL** over the
best foundation-model configuration (+3.1 on Trento, within spread), at **more
than two orders of magnitude fewer parameters**.

## What's here

Three routes, all measured by the same instrument:

| Route | What it is | Entry points |
|---|---|---|
| **CoFFE** | the paper's compact encoder: 11×11 patch, HSI + LiDAR concatenated **at the input** and jointly embedded, 121 pixel tokens + 1 class-agnostic token, pre-LN transformer `D=128`, 2 heads, 2 layers. Pretrained per scene, no labels. | [`scripts/pretrain.py`](scripts/pretrain.py), [`scripts/evaluate.py`](scripts/evaluate.py) |
| **MFT control** | the architectural control: original MFT (Roy et al.) with its **external** fusion token, pretrained under the same masked objectives and evaluated under the same protocol. | [`scripts/pretrain.py`](scripts/pretrain.py), [`scripts/evaluate_mft.py`](scripts/evaluate_mft.py) |
| **HyperSIGMA** | the foundation-model route: the released SpatViT-B / SpecViT-B checkpoints, used frozen or after **label-free** continued masked reconstruction on the target scene. | [`scripts/adapt_hypersigma.py`](scripts/adapt_hypersigma.py), [`scripts/evaluate_hypersigma.py`](scripts/evaluate_hypersigma.py) |

**The evaluation instrument** is one protocol, applied identically to all three:
the encoder is **frozen** (projection head discarded), episodes are **N-way**
with N = the scene's full class count (Houston 15, Trento 6, MUUFL 11),
**K = 5** support samples per class, **100 queries per class** (class-balanced),
and a query is assigned by **Euclidean nearest-class-mean** — the class mean is
the average of its 5 support features. Nothing is finetuned and there are no
learnable similarity parameters: this is a nearest-class-mean classifier on
fixed features (Mensink et al.), not a prototypical network (Snell et al.).

For CoFFE the feature being classified is **not** just the pooled patch tokens.
The code folds in the weighted class-agnostic token, and every paper run used
`lambda_factor = 0.5`:

```
z = mean_j(patch_emb_j) + 0.5 · cls_emb
```

The paper's §3 wording ("patch tokens pooled") is the incomplete one; this is
the behaviour, it is frozen, and removing the λ term moves OA by −1.70 pp
([`PAPER_CANON.md`](PAPER_CANON.md) §8 D3). The MFT control and HyperSIGMA
override that method with a pass-through, so it is CoFFE-only.

Read the episode counts per table: Table 2 ran **1000** episodes per evaluation,
the Table 3 HyperSIGMA sweep ran **2000**, and one Table 3 cell (Houston 11×11
spectral) used `k_query=30` instead of 100 (§8 D18). Because the query sets are
class-balanced, OA = AA, which is why the paper reports OA only. Details:
[`docs/evaluation.md`](docs/evaluation.md).

**The pretraining regimes** are a pair of mask rates over the same token space,
not separate code paths — band masking of `(pixel, band)` entries at rate `r_b`,
spatial masking of whole pixel tokens at rate `r_s`:

| Regime | `(r_b, r_s)` | Objective |
|---|---|---|
| SimMIM band | `(0.85, 0)` | in-place masked reconstruction, masked-MSE over the union of both masks |
| **SimMIM token** *(headline)* | `(0, 0.75)` | as above |
| SimMIM band+token | `(0.85, 0.75)` | as above — but Table 2's Houston cell used `(0.75, 0.75)`; see [`PAPER_CANON.md`](PAPER_CANON.md) §8 D19 |
| MAE | 75 % of tokens **dropped** | He et al. token-drop recipe with a transformer decoder |

Details: [`docs/pretraining.md`](docs/pretraining.md).

## Results

**Table 2** — OA %, CoFFE and the MFT control, ± across-seed std over 5 seeds;
HyperSIGMA rows are single runs.

| Encoder / config | Modality | Houston | Trento | MUUFL |
|---|---|---|---|---|
| CoFFE SimMIM band | HSI | 58.91 ± 0.9 | 87.04 ± 0.4 | 55.02 ± 0.7 |
| CoFFE SimMIM band | HSI+LiDAR | 60.21 ± 0.4 | 90.32 ± 0.6 | 56.99 ± 1.2 |
| CoFFE SimMIM token | HSI | 65.50 ± 2.1 | 87.60 ± 1.0 | 57.75 ± 1.3 |
| **CoFFE SimMIM token** | **HSI+LiDAR** | **75.30 ± 1.6** | **94.19 ± 0.8** | **67.83 ± 2.2** |
| CoFFE SimMIM band+token | HSI | 58.24 ± 1.1 | 86.10 ± 0.4 | 50.76 ± 1.4 |
| CoFFE SimMIM band+token | HSI+LiDAR | 64.63 ± 1.5 | 92.50 ± 1.2 | 55.57 ± 3.2 |
| CoFFE MAE | HSI | 69.04 ± 1.1 | 88.43 ± 4.4 | 58.65 ± 0.4 |
| CoFFE MAE | HSI+LiDAR | 72.15 ± 0.7 | 91.74 ± 0.7 | 65.33 ± 1.0 |
| MFT SimMIM token | HSI+LiDAR | 67.49 ± 0.3 | 88.22 ± 0.9 | 59.05 ± 0.5 |
| MFT MAE | HSI+LiDAR | 51.97 ± 3.7 | 81.05 ± 4.6 | 45.77 ± 1.8 |
| HyperSIGMA frozen (64×64) | HSI | 61.14ᵘ | 91.07ᵘ | 54.80ᵖ |
| HyperSIGMA 11×11 (patch-native) | HSI | 67.48ʲ | 87.19ʲ | 50.25ˢ |

(ᵘ upscale, ᵖ pad, ʲ joint+SEM, ˢ spatial)

**Table 3** — the HyperSIGMA sweep, OA % ± 95 % CI over episodes. Features:
sp. = spatial branch (768-d), sc. = spectral branch (768-d), fu. = fused SEM
(512-d).

| Input | Adaptation | Feat. | Houston | Trento | MUUFL |
|---|---|---|---|---|---|
| 64×64 pad | frozen | sp. | 59.33 ± 0.10 | 85.03 ± 0.12 | 54.80 ± 0.12 |
| 64×64 upscale | frozen | sp. | 61.14 ± 0.11 | 91.07 ± 0.09 | 53.17 ± 0.13 |
| 64×64 pad | frozen | sc. | 30.09 ± 0.17 | 81.12 ± 0.19 | 34.25 ± 0.12 |
| 64×64 upscale | frozen | sc. | 30.71 ± 0.11 | 67.42 ± 0.17 | 30.77 ± 0.10 |
| 64×64 pad | SEM-only | fu. | 45.32 ± 0.12 | 73.76 ± 0.22 | 44.34 ± 0.13 |
| 11×11 | spatial | sp. | 66.37 ± 0.12 | 86.02 ± 0.11 | 50.25 ± 0.14 |
| 11×11 | spectral | sc. | 21.85 ± 0.10 | 48.59 ± 0.16 | 20.70 ± 0.09 |
| 11×11 | joint+SEM | fu. | 67.48 ± 0.11 | 87.19 ± 0.12 | 44.70 ± 0.12 |

**The two ± are not the same quantity.** Table 3's is a **95 % confidence
interval over the episodes of one run**. Table 2's is the **standard deviation
across 5 seeds** (`[42, 123, 456, 789, 1011]`) of a *separate* experiment — the
5-seed runs were pretrained fresh to 700 epochs, while the means come from the
single canonical runs described under [Reproduce](#reproduce). Two Trento cells
have a further wrinkle recorded in [`PAPER_CANON.md`](PAPER_CANON.md) §8 D20:
their ± was measured at a different mask rate than their mean.

Parameter counts, for the comparison the paper makes: the CoFFE eval encoder is
**579,328** parameters on Houston (marginally fewer on Trento and MUUFL);
HyperSIGMA is ~180M with both ViT-Base bodies plus the SEM, of which its
label-free adaptation trains 1.13M–8.45M — a range that does not cover every
cell of the sweep, since the 11×11 spectral adaptation trains 333,689
(`docs/refactor/LOG.md`, phase-7 S5).

Two provenance notes the tables do not show. The headline Houston cell (75.30)
comes from a run whose directory name says `spatial_mask_test` and `seed52`, and
both are misnomers — it ran seed 42 and it is the headline result;
`scripts/compile_results.py` still filters that directory out as a scratch run,
which is why 75.30 is absent from the compiled `RESULTS.json`
([`PAPER_CANON.md`](PAPER_CANON.md) §8 D14). And the ± of two Trento cells was
measured at a different mask rate than their mean (§8 D20).

The JSONs behind the tables are in [`results/`](results/README.md) — complete
for Table 2, and 20 of Table 3's 24 cells — with the cell-by-cell provenance
trace in [`docs/refactor/AUDIT.md`](docs/refactor/AUDIT.md) §3.

## Installation

Python **3.11+** (CI runs 3.11 and 3.12; this release was developed and verified
on 3.12 with torch 2.11.0+cu128 — [`docs/refactor/ENV.md`](docs/refactor/ENV.md)
records the full environment).

```bash
git clone <repo-url> coffe
cd coffe
python -m venv .venv && source .venv/bin/activate
pip install -e .          # runtime
pip install -e ".[dev]"   # + pytest, ruff, mypy
```

Extras: `.[tensorboard]` for pretraining scalars, `.[notebooks]` for the Jupyter
workflow. `requirements.txt` is a one-line mirror of `-e .[dev]`.

The install also makes the vendored HyperSIGMA sources under `third_party/`
importable, which the HyperSIGMA route needs. Every entry point under
`scripts/` puts the repo root on `sys.path` itself, so `python scripts/…` works
in an uninstalled checkout as well; only `import coffe` from an unrelated
directory needs the install.

## Data

The three scenes are **not redistributed here** — download them from their
owners and place them under `data/raw/` (which is gitignored). This repository
consumes the **pre-patched MFT-format** `.mat` files, i.e. 11×11 patches already
cut around each labelled pixel; the patch extraction and its border padding
happened upstream, in the MFT data preparation.

| Scene | Bands | Aux | Classes | Source |
|---|---|---|---|---|
| Houston 2013 | 144 HSI | 1 LiDAR (elevation) | 15 | IEEE GRSS Data Fusion Contest 2013 |
| Trento | 63 HSI | 1 LiDAR | 6 | University of Trento |
| MUUFL Gulfport | 64 HSI | 2 LiDAR rasters | 11 | University of Florida MUUFL Gulfport dataset |

`scripts/download_data.sh <scene>` prints an acquisition pointer for each scene
and the directory to unpack it into. Note those pointers are **third-party
mirrors** of the pre-patched copies, not the owners' own distribution portals —
if you need the scenes from the source, go to the providers named above:

```bash
bash scripts/download_data.sh houston   # then trento, muufl
```

Expected layout, exactly as the loaders in
[`coffe/data/datasets/patched.py`](coffe/data/datasets/patched.py) require it:

```
data/raw/
├── Houston11x11/   HSI_Tr.mat  HSI_Te.mat  LIDAR_Tr.mat  LIDAR_Te.mat  TrLabel.mat  TeLabel.mat
├── Trento11x11/    (same six files)
└── MUUFL11x11/     (same six files)
```

Every HSI band and LiDAR raster is min-max normalised to [0, 1] independently at
load time. `split="all"` — train and test patches concatenated — is what every
paper evaluation used: the protocol's own support/query split is drawn per
episode.

## Pretrained checkpoints

**HyperSIGMA** (required for that route only) — the released SpatViT-B and
SpecViT-B MAE checkpoints, ~1.4 GB each, from
[WHU-Sigma/HyperSIGMA](https://huggingface.co/WHU-Sigma/HyperSIGMA):

```bash
bash scripts/download_hypersigma_checkpoints.sh    # -> checkpoints/hypersigma/
```

For every published cell the spatial branch is fed a **100-channel** input: PCA
144→100 on Houston (fit it with
`python scripts/fit_pca_hypersigma.py --dataset houston --n-components 100`),
linear band resampling 63/64→100 on Trento and MUUFL. The 3-band PCA front-end
that `scripts/fit_pca_hypersigma.py` produces by default is also supported and
was used in no Table 3 cell.

**CoFFE / MFT checkpoints** — `TODO(release)`: the paper promises public
checkpoints; the hosting link goes here. Until then every CoFFE and MFT number
is reproducible from the recipes below, and nothing else in this repository
needs a checkpoint.

## Reproduce

Hardware: the wall times below are the paper's, for **one NVIDIA RTX 4090**.
PAPER_CANON §3 gives CoFFE pretraining as 0.7–4.3 h per scene **for 1500
epochs**, so the rows whose Houston schedule is 3000 epochs cost roughly twice
that on Houston; HyperSIGMA adaptation is ≈36 h for 2000 epochs. CPU is enough
for the test suite and the no-data checks below, not for a pretraining
schedule.

Each Table 2 cell has a config carrying **the exact recipe of the run that
produced its published mean**, including that run's mask rates and the
**checkpoint epoch that was evaluated** — which is *not* the final one: epoch
950 on Houston, 975 on Trento/MUUFL for CoFFE, 950 for all six MFT cells, with
one cell at 800. That was not a model-selection decision — those epochs are what
a checkpoint sort that ordered filenames as strings happened to return
(`"...950" > "...1500"`), which the phase-7 gate made numeric
([`PAPER_CANON.md`](PAPER_CANON.md) §8 D17). The numbers are sound either way,
and the generic recipe follows from it: train the full schedule, then evaluate
the epoch the config names.

```bash
# one Table 2 cell, end to end (Houston, CoFFE SimMIM token, HSI+LiDAR = 75.30)
python scripts/pretrain.py --config configs/coffe/houston_simmim_token.yaml
# writes to that config's paths.checkpoint_dir, one file per save_interval
python scripts/evaluate.py \
    --checkpoint ./checkpoints/coffe/houston_simmim_token/checkpoint_epoch_950.pth \
    --dataset houston --data-root ./data/raw
```

`scripts/evaluate.py`'s *protocol* defaults **are** the paper's (N-way, K=5,
`k_query` 100, 1000 episodes, `split all`, Euclidean NCM, projection head off),
so the protocol needs nothing on the command line. Two exceptions to that
convenience:

- **HSI-only cells need `--no-aux`.** The `_hsi.yaml` configs pretrain with
  `use_aux: false`, but the CLI defaults to `--use-aux` and does not read the
  cell config, so evaluating an HSI-only checkpoint without `--no-aux` fails on
  a band-count mismatch — one that now names `--no-aux` in its message:

  ```bash
  python scripts/evaluate.py --no-aux       --checkpoint ./checkpoints/coffe/houston_simmim_token_hsi/checkpoint_epoch_950.pth       --dataset houston --data-root ./data/raw
  ```

  The driver and notebook routes do not need it: they read `use_aux` back out
  of the run's frozen `pretrain_config.yaml`.
- **`scripts/evaluate_hypersigma.py` defaults to Table 3's protocol**
  (`k_query` 100, 2000 episodes, `split all`) rather than Table 2's, because
  that is the table it serves. The Table 3 rows below still spell the flags out.

| Paper table (cell group) | Command | Epochs | Wall time |
|---|---|---|---|
| **T2** CoFFE SimMIM ×3 regimes (the 18 means) | `scripts/pretrain.py --config configs/coffe/<scene>_simmim_{band,token,band_token}[_hsi].yaml`, then `scripts/evaluate.py` at the epoch that config names | 1500 (one cell 2000) | 0.7–4.3 h per scene |
| **T2** CoFFE MAE rows, both modalities | `python scripts/reproduce/run_mae_experiments.py` (pretrain + evaluate, all 6 MAE cells) | 1500; Houston 3000 | as above |
| **T2** CoFFE SimMIM HSI-only cells, as one driver | `python scripts/reproduce/run_hsi_only_experiments.py` (the same nine cells as the `_hsi` configs above, run as a matrix) | 1500 | as above |
| **T2** MFT control, SimMIM token | `python scripts/reproduce/run_mft_original_spatial_experiments.py` | 1500; Houston 3000 | as above |
| **T2** MFT control, MAE | `python scripts/reproduce/run_mft_original_mae_experiments.py` | 1500; Houston 3000 | as above |
| **T2** the ± column (5 seeds × 30 cells) | `python scripts/reproduce/run_significance_experiment.py --stage all` | **700** (a separate experiment) | multi-GPU, days |
| **T3** 64×64 frozen, spatial / spectral (12 cells) | `python scripts/evaluate_hypersigma.py --dataset <scene> --native-geometry --input-fit {upscale,pad} --mode {spat_pool,spec_pool} --adapted-checkpoint none --split all --k-query 100 --num-episodes 2000` | no training | minutes per cell |
| **T3** 64×64 pad, SEM-only (3 cells) | `bash scripts/reproduce/run_native_sem_pad_experiments.sh` + `configs/hypersigma/<scene>_backbonenative_pad_sem_only.yaml` | 2000 | ≈36 h per scene |
| **T3** 11×11 spatial (3 cells) | `python scripts/reproduce/run_hypersigma_spatial_pca100.py --dataset <scene>` | 2000 | ≈36 h per scene |
| **T3** 11×11 joint+SEM (3 cells) | Houston: `python scripts/adapt_hypersigma.py --config configs/hypersigma/houston_patchnative_pca100_joint_sem.yaml` — **but** the run overrode that file's schedule (2000 epochs / batch 128 / lr 1e-5 vs the file's 3000 / 64 / 1.5e-4). Trento and MUUFL have only the 3-band `<scene>_patchnative_joint_sem.yaml`, which reproduces no cell as written; their runs added `spat_resample_to: 100` and their own schedule at launch. In all three cases that run's `pretrain_overrides.yaml` is the authority. Then `scripts/evaluate_hypersigma.py --mode fused --split all --k-query 100 --num-episodes 2000` | 2000 | ≈36 h per scene |
| **T3** 11×11 spectral (3 cells) | no committed driver — see below | 2000 | ≈36 h per scene |

From a notebook instead of the shell: `notebooks/pretrain.ipynb`,
`notebooks/evaluate.ipynb` and `notebooks/compare.ipynb` wrap the same
`coffe/runners/` entry points, each with a Parameters cell at the top; the
HyperSIGMA notebooks cover that route. They write the same
`experiments/<name>/` trees as the CLIs.

Of Table 3's eight rows, **two** have a committed driver end to end (64×64 pad
SEM-only, 11×11 spatial), **one** is partial (11×11 joint+SEM, above), and
**five** have none: the four 64×64 frozen rows and 11×11 spectral, all of which
were driven from `notebooks/evaluate_hypersigma_native.ipynb` and
`notebooks/evaluate_hypersigma.ipynb`. The frozen rows involve no training, so
the command in the table above reproduces them from flags alone; the spectral
cells' adaptation settings were never reconstructed into a config. That gap is
deliberate — inventing settings for a frozen paper run would be worse than
naming the notebook that produced it.
[`scripts/reproduce/README.md`](scripts/reproduce/README.md) is the row-by-row
map, and [`docs/hypersigma.md`](docs/hypersigma.md) covers that route's
mechanics.

## Quick sanity run

Everything here runs on this repository as shipped. The first two need no data,
no checkpoints and no GPU:

```bash
pytest -q -m "not gpu and not data"          # the suite
# The behaviour goldens. On any environment but the one they were generated on
# these skip themselves, with the difference in the message — see Tests, below.
pytest -q -rs tests/equivalence
python scripts/evaluate.py --help
python -c "
from coffe.models import CoFFE
m = CoFFE(hsi_channels=144, aux_channels=1, embed_dim=128, num_heads=2,
          num_layers=2, patch_size=11, use_projection=False)
print(sum(p.numel() for p in m.parameters()))   # 579328, the paper's Houston encoder
"
```

With the datasets in place and a GPU, a **20-epoch** CoFFE pretrain plus a
50-episode evaluation exercises the whole pipeline in about 45 seconds:

```bash
python - <<'PY'
from coffe.runners.eval_runner import run_evaluation
from coffe.runners.pretrain_runner import run_pretrain

ROOT = "/tmp/coffe-smoke"  # nothing is written inside the repository

run_pretrain(
    name="houston_simmim_token_smoke",
    description="20-epoch smoke run - NOT the paper's 1500-epoch schedule.",
    config="configs/coffe/houston_simmim_token.yaml",
    overrides={"pretrain": {"epochs": 20, "save_interval": 20}},
    experiments_root=ROOT,
    overwrite=True,
)
ev = run_evaluation(
    experiment_name="houston_simmim_token_smoke",
    eval_name="houston_15way_5shot_smoke",
    epoch=20,
    # The projection head needs no mention: it is a pretraining-only part, and
    # since the phase-8 gate the runner no longer inherits it from the
    # pretraining config, so eval discards it as PAPER_CANON §4 requires.
    eval_params={"dataset": "houston", "num_episodes": 50, "no_plots": True},
    experiments_root=ROOT,
    overwrite=True,
)
print("OA:", ev.metadata["summary"]["OA"])
PY
```

and the frozen HyperSIGMA route, which needs its two ViT-B checkpoints, in about
15 seconds:

```bash
python scripts/evaluate_hypersigma.py \
    --dataset houston --native-geometry --input-fit upscale --mode spat_pool \
    --adapted-checkpoint none --split all --num-episodes 20 --k-query 10 \
    --no-plots --output-dir /tmp/coffe-smoke/hypersigma_native_upscale
```

**Neither GPU leg runs the paper's protocol**, so read them as wiring checks:

| Leg | Protocol used | Paper's protocol | Measured | Paper |
|---|---|---|---|---|
| CoFFE Houston SimMIM token | **20** epochs, 50 episodes | 1500 epochs, 1000 episodes | OA 71.15 ± 0.67 | 75.30 ± 1.6 |
| HyperSIGMA 64×64 upscale / frozen / spatial, Houston | 20 episodes, `k_query` **10** | 2000 episodes, `k_query` 100 | OA 61.03 ± 1.74 | 61.14 ± 0.11 |

The HyperSIGMA leg targets a published Table 3 cell with a frozen encoder and no
training, so given enough episodes it should reproduce that cell — and −0.11 sits
well inside the smoke's own ±1.74. That makes it a weak reproduction check, not
merely a wiring check; weak because 20 episodes at `k_query` 10 is a ~100×
smaller query budget than the published cell's.

The CoFFE leg is a wiring check only, and has no published value to match: a
20-epoch encoder is not the 1500-epoch one, and with `warmup_epochs: 80` in that
config a 20-epoch run never even leaves learning-rate warmup. Treat its OA as a
"the pipeline runs and the number is not absurd" signal, nothing more — an
earlier 20-epoch smoke recorded in `docs/refactor/LOG.md` (phase-7 gate) reported
**74.43 ± 0.66** under the same *stated* settings, and because that run's exact
invocation was not recorded, the 3.3 pp difference cannot be attributed. (One
candidate was checked and ruled out: pretraining from the 5-seed runner's base
config, `configs/coffe/houston_simmim.yaml` at lr 1.5e-5 / batch 64, gives
**56.65 ± 0.96** at 20 epochs — not 74.4 either.) Two 20-epoch measurements
3 pp apart is itself the point: nothing about a truncated schedule is stable
enough to read.

The HyperSIGMA leg moved between phases as well — phase 7 recorded
60.83 ± 1.79 where this run gives 61.03 ± 1.74 — so neither leg is reproducing
bit-for-bit across sessions at these episode counts, which is another reason to
read both as wiring checks.

All numbers here were measured on one RTX 4090 while writing this section; the ±
are 95 % CIs over the smoke's own episodes, not across seeds.

## Project structure

```
CoFFE/
├── coffe/                    the installable package
│   ├── models/               CoFFE, MFTOriginal, components/, hypersigma/ (dual encoder + SEM + wrappers)
│   ├── pretrain/             SimMIM / MAE masked modelling, the training loop, HyperSIGMA adaptation
│   ├── data/                 dataset loaders (datasets/) and episode samplers (samplers/)
│   ├── eval/                 the frozen-encoder episodic evaluators (episodic.py, hypersigma.py)
│   ├── runners/              notebook-friendly pretrain / eval / adapt entry points + experiment logging
│   ├── utils/                seeding, IO, metrics, spatial weights, plots
│   └── compat.py             legacy-name aliases, so pre-rename artifacts still read
├── scripts/                  thin CLIs (pretrain, evaluate, evaluate_mft, evaluate_hypersigma, adapt, fit-PCA)
│   ├── reproduce/            the experiment drivers behind Tables 2 and 3
│   └── reports/              provenance and aggregation builders for results/
├── configs/                  per-cell recipes: coffe/, mft/, hypersigma/
├── constraints/              the pinned environment the behaviour goldens need
├── results/                  the JSONs behind the paper's tables (frozen; see results/README.md)
├── experiments/              per-run output trees (gitignored except _example/)
├── notebooks/                pretrain, evaluate, compare, and the HyperSIGMA notebooks
├── tests/                    unit/, integration/, equivalence/ (behaviour goldens)
├── third_party/HyperSIGMA/   vendored upstream + the local patches its NOTICE lists (own LICENSE)
├── docs/                     pipeline docs + refactor/ (the cleanup's provenance)
├── tools/refactor/           one-shot tooling from the release cleanup, kept for provenance
├── PAPER_CANON.md            names, protocol constants, published tables, known paper↔code gaps
└── CHANGES.md                the old→new name table and the compatibility guarantees
```

Every run — from a notebook or from `coffe/runners/` — writes one directory
under `experiments/<name>/`: the frozen `pretrain_config.yaml`, metadata with
the git SHA and wallclock, checkpoints, and one subdirectory per evaluation
holding `eval_config.json`, `results.json` and plots. See
[`experiments/_example/`](experiments/_example/) for the schema.

## Tests

```bash
pip install -e ".[dev]"
pytest -q -m "not gpu and not data"
```

No dataset and no HyperSIGMA checkpoint is needed: the suite builds synthetic
scenes shaped like the real ones. Tests that do need them are marked `data` /
`gpu` and deselected above; `slow` marks the ones that build ViT-Base bodies or
shell out to the CLIs.

[`tests/equivalence/`](tests/equivalence/README.md) is a behaviour-freeze
harness rather than a unit-test suite: it pins pretraining loss trajectories,
encoder forwards, masking and the full episodic evaluation against committed
goldens, plus a pre-refactor checkpoint that must still load. It is what makes
"this refactor changed no computed number" a checkable claim.

**Those goldens are bit-exact on one environment** — the versions
`tests/equivalence/golden/meta.json` records, *and* the reference machine's
thread count, since BLAS reduction order follows it. Reconstruct it with:

```bash
pip install -e ".[dev]" -c constraints/verification.txt
```

Anywhere else the package **skips itself** and names the difference, because
float drift from a newer torch is not evidence about this code. Everything
outside `tests/equivalence/` is unaffected either way: measured in a fresh clone
on today's resolved versions, 486 passed, 0 failed. Details, including the
measurements behind the pin, are in
[`tests/equivalence/README.md`](tests/equivalence/README.md).

## Citation

```bibtex
@inproceedings{maric2026coffe,
  title     = {A Compact In-Domain Fusion Encoder versus a Hyperspectral
               Foundation Model for Few-Shot HSI-LiDAR Land-Cover Classification},
  author    = {Mari{\'c}, Nikola and Kocev, Dragi},
  year      = {2026},
  note      = {TODO(release): venue, pages, DOI}
}
```

[`CITATION.cff`](CITATION.cff) carries the same metadata in machine-readable
form.

## License & acknowledgements

This repository is released under the [MIT License](LICENSE).

[`third_party/HyperSIGMA/`](third_party/HyperSIGMA) is vendored upstream code
(Apache-2.0), covered by **its own LICENSE and NOTICE** in that directory — not
by this repository's MIT license. It is not pristine: its `NOTICE` records the local
patches applied to it — an `mmengine` `get_dist_info` import shim,
relative-import fixes so the tree loads in place, and a `patch_size == 3` FPN
branch that mirrors upstream's `patch_size == 11` handling. That last one is a
live code path, not a cosmetic edit: every patch-native (11×11) Table 3 cell
runs through it. The MFT architecture and the
pre-patched dataset format come from Roy et al.'s MFT release; the Houston 2013,
Trento and MUUFL Gulfport scenes remain the property of their respective
providers.

Funding acknowledgement: `TODO(release)`.

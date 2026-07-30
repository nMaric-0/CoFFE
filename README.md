# CoFFE: Compact In-Domain Fusion Encoder for Few-Shot HSI-LiDAR Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of

> Nikola Marić and Dragi Kocev.
> **A Compact In-Domain Fusion Encoder versus a Hyperspectral Foundation Model
> for Few-Shot HSI-LiDAR Land-Cover Classification.**
> MACLEAN workshop, ECML PKDD 2026.

The paper compares two label-free routes to a frozen encoder for scarce-label
land-cover classification on co-registered HSI-LiDAR scenes:

1. **CoFFE**: a compact spectral-spatial transformer (579,328 parameters on
   Houston) with input-level HSI-LiDAR fusion, pretrained from scratch on the
   target scene by masked modelling (SimMIM-style band / token masking, plus a
   standard MAE baseline).
2. **HyperSIGMA**: the hyperspectral foundation model (~180M parameters, two
   ViT-Base branches plus a fusion module), taken frozen off the shelf and
   adapted label-free by continued masked reconstruction. HSI-only, since it
   has no native path for the auxiliary modality.

Every encoder is frozen and probed with the same parameter-free protocol:
N-way 5-shot Euclidean nearest-class-mean (NCM), 100 queries per class,
1000 episodes per run, N = all classes of the scene. With input-level fusion,
CoFFE leads the best foundation-model configuration by 7.8 OA on Houston and
13.0 on MUUFL.

## Paper-to-code terminology

The codebase predates the paper's naming; this table is the bridge.

| Paper                                        | Code                                                                                     |
|----------------------------------------------|------------------------------------------------------------------------------------------|
| CoFFE encoder                                | `MFTCPEACosine` in `models/mft_cpea_cosine.py`; `model.name: "mft_cpea"` in configs       |
| Input-level fusion                           | `use_aux: true` (LiDAR concatenated with HSI before tokenization, one token per pixel)    |
| *SimMIM token*, (r_b, r_s) = (0, 0.75)       | variant `spatial`; `spatial_mask_ratio: 0.75`, `band_mask_ratio: 0`                       |
| *SimMIM band*, (0.85, 0)                     | variant `spectral`; `band_mask_ratio: 0.85`, `spatial_mask_ratio: 0`                      |
| *SimMIM band+token*, (0.85, 0.75)            | variant `both`; both ratios enabled                                                       |
| MAE baseline                                 | `pretrain.objective: "mae"` (`configs/pretrain/*_pretrain_mae*.yaml`)                     |
| MFT control (*SimMIM token* / MAE)           | `configs/pretrain/mft_original_<scene>_spatial.yaml` / `..._mae.yaml`                     |
| HyperSIGMA *frozen*                          | released checkpoints evaluated directly, no adaptation                                    |
| *SEM-only* / *spatial* / *spectral* / *joint + SEM* | `model.adapt_mode: "sem_only"` / `"spatial_only"` / `"spectral_only"` / `"joint_sem"` |
| backbone-native (64x64, pad / upscale)       | `native_geometry: true` with `input_fit: "pad"` or `"upscale"`                            |
| patch-native (11x11)                         | `native_geometry: false` (3x3 stride-3 patch embedding, 4x4 token grid)                   |
| Euclidean NCM probe                          | `forward_episode` with `distance_metric="euclidean"`, `prototype_mode="mean_features"`    |

## Installation

```bash
git clone https://github.com/nMaric-0/CoFFE.git
cd CoFFE
conda create -n coffe python=3.9 -y && conda activate coffe
pip install -r requirements.txt
pip install -e .
```

## Data

All three benchmarks are consumed as 11x11 patches centred on labelled pixels
(border-padded at the edges); every HSI band and LiDAR raster is min-max
normalised to [0, 1] independently by the loaders.

| Dataset      | HSI bands | LiDAR rasters | Classes | Labelled px |
|--------------|-----------|----------------|---------|-------------|
| Houston 2013 | 144       | 1              | 15      | 15,029      |
| Trento       | 63        | 1              | 6       | 30,414      |
| MUUFL        | 64        | 2              | 11      | 53,687      |

`bash scripts/download_data.sh <houston|trento|muufl>` prints the download
location of the pre-extracted patch archives; place them under
`./data/raw/{Houston11x11,Trento11x11,MUUFL11x11}`. Original data credits:
Houston 2013 (2013 IEEE GRSS Data Fusion Contest, Debes et al. 2014), Trento
(courtesy of L. Bruzzone, University of Trento), MUUFL Gulfport (University of
Florida GatorSense, Gader et al. 2013; Du and Zare 2017).

## Route 1: CoFFE per-scene pretraining

The shipped scene configs are the *SimMIM token* regime; the other regimes are
one-line overrides. The paper budget is **1500 epochs** (shipped configs
default to a longer schedule; a `final.pth` / `encoder_final.pth` checkpoint is
written at the end of training either way).

CLI:

```bash
python scripts/pretrain_enhanced.py \
    --config configs/pretrain/houston_pretrain_enhanced.yaml
```

Programmatic (notebook) form, with the regime and budget made explicit:

```python
from lib.pretrain_runner import run_pretrain

run_pretrain(
    name="houston_simmim_band",
    description="SimMIM band (0.85, 0), HSI+LiDAR, paper budget.",
    config="configs/pretrain/houston_pretrain_enhanced.yaml",
    overrides={"pretrain": {
        "band_mask_ratio": 0.85, "spatial_mask_ratio": 0.0, "epochs": 1500,
    }},
)
```

Regimes: `(0, 0.75)` token, `(0.85, 0)` band, `(0.85, 0.75)` band+token.
HSI-only counterparts: `configs/pretrain/<scene>_pretrain_hsi_only.yaml`.
MAE baseline: `configs/pretrain/<scene>_pretrain_mae.yaml` (HSI+LiDAR) and
`..._mae_hsi_only.yaml`. The decoder and projection head exist only for
pretraining and are not part of the evaluated encoder.

## Route 2: HyperSIGMA label-free adaptation

```bash
# 1. Fetch the released SpatViT-B / SpecViT-B checkpoints (WHU-Sigma, Hugging Face)
bash scripts/download_hypersigma_checkpoints.sh

# 2. Houston only: fit the 144 -> 100 PCA for the spatial branch
python scripts/fit_pca_hypersigma.py  # writes checkpoints/hypersigma/pca_houston_100band.pkl

# 3. Adapt (75% token masking, per-token z-scored targets; paper budget 2000 epochs)
python scripts/adapt_hypersigma.py \
    --config configs/pretrain/hypersigma_houston_adapt.yaml
```

Trento and MUUFL map the spectral dimension by linear band resampling instead
of PCA; the spectral branch always receives the raw bands. The *frozen*
configurations evaluate the released checkpoints directly, with the 11x11
patch fitted to the 64x64 input by centred zero-padding or bicubic upsampling
(`input_fit`). Vendored upstream code lives in `third_party/HyperSIGMA/` with
its original license and notice.

## MFT architectural control

The original external-token MFT, pretrained under the same two objectives and
evaluated identically (token masking acts on the 121 pixel tokens):

```bash
bash scripts/run_mft_original_spatial_experiments.sh   # SimMIM token
bash scripts/run_mft_original_mae_experiments.sh       # MAE
```

## Few-shot evaluation (the paper protocol)

Frozen encoder, Euclidean NCM over the class means of K=5 support features,
N = all scene classes, 100 queries per class, 1000 episodes, features used as
produced (no centering or L2 normalization). Note that the CLI and model
**default to cosine similarity; pass Euclidean explicitly** to reproduce the
paper.

```bash
python scripts/evaluate_cosine.py \
    --checkpoint <path/to/encoder_checkpoint.pth> \
    --dataset houston --split all \
    --k-shot 5 --k-query 100 --num-episodes 1000 \
    --distance-metric euclidean
```

`--n-way` defaults to all classes of the scene; add `--no-aux` for HSI-only
checkpoints. The programmatic equivalent (architecture and pooling are
auto-loaded from the experiment's frozen pretrain config):

```python
from lib.eval_runner import run_evaluation

run_evaluation(
    experiment_name="houston_simmim_band",
    eval_name="paper_protocol",
    eval_params={
        "dataset": "houston", "split": "all",
        "k_shot": 5, "k_query": 100, "num_episodes": 1000,
        "distance_metric": "euclidean",
    },
)
```

Each run writes a frozen config, metadata (git SHA, wallclock), logs,
checkpoints, and per-evaluation `results.json` under `experiments/<name>/`;
see `experiments/_example/` for the schema.

## Multi-seed significance study

Every CoFFE and MFT configuration is retrained from five seeds
(42, 123, 456, 789, 1011) under an identical shortened schedule (700 epochs)
and its final checkpoint is evaluated under the paper protocol above. The
across-seed OA standard deviations from this study are the +/- values in
Table 2 of the paper. Orchestration:

```bash
python scripts/run_significance_experiment.py
```

Shipped per-seed logs:

- `experiments/significance_report_enhanced.json`: HSI+LiDAR SimMIM variants.
- `experiments/significance_report.json`: HSI-only SimMIM, MAE, and both MFT
  controls.

## Results

Headline numbers (OA %, 5-shot N-way; +/- is the across-seed standard
deviation over five retrainings; HyperSIGMA is single-run):

| Encoder      | Configuration  | Modality   | Houston      | Trento       | MUUFL        |
|--------------|----------------|------------|--------------|--------------|--------------|
| CoFFE        | SimMIM token   | HSI+LiDAR  | 75.30 +/- 1.6 | 94.19 +/- 0.8 | 67.83 +/- 2.2 |
| CoFFE        | MAE            | HSI+LiDAR  | 72.15 +/- 0.7 | 91.74 +/- 0.7 | 65.33 +/- 1.0 |
| MFT control  | SimMIM token   | HSI+LiDAR  | 67.49 +/- 0.3 | 88.22 +/- 0.9 | 59.05 +/- 0.5 |
| HyperSIGMA   | best per scene | HSI        | 67.48        | 91.07        | 54.80        |

The full configuration grid, the HyperSIGMA input-regime x adaptation sweep,
and the protocol definition are in the paper. Raw result JSONs backing the
tables ship under `experiments/`: `aggregated_results.json` (CoFFE cells),
`gathered_results.json` and `hypersigma_native_sem_pca100_report.json`
(HyperSIGMA sweep), `mft_faithful_results.json` (MFT control), and the two
significance reports above (seed spread).

## Pretrained checkpoints

Per-scene pretrained CoFFE encoders and the adapted HyperSIGMA front-ends:
**[link to release / archive: TODO]**.

## Citation

```bibtex
@inproceedings{maric2026coffe,
  author    = {Mari{\'c}, Nikola and Kocev, Dragi},
  title     = {A Compact In-Domain Fusion Encoder versus a Hyperspectral
               Foundation Model for Few-Shot {HSI}-{LiDAR} Land-Cover
               Classification},
  booktitle = {ECML PKDD 2026 Workshops (MACLEAN)},
  year      = {2026},
  note      = {To appear}
}
```

## License and acknowledgements

MIT, see [LICENSE](LICENSE). HyperSIGMA code and checkpoints by Wang et al.
(IEEE TPAMI 2025), vendored under `third_party/HyperSIGMA/` with the original
license and notice; the architectural control follows the multimodal fusion
transformer of Roy et al. (IEEE TGRS 2023). Datasets courtesy of the 2013
IEEE GRSS Data Fusion Contest (Houston), L. Bruzzone / University of Trento
(Trento), and the University of Florida GatorSense group (MUUFL).

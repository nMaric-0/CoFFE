# `scripts/reproduce/` — the experiment drivers

These are the drivers that produced the paper's runs. They assume the datasets
under `data/raw/` and (for the HyperSIGMA route) the released checkpoints; none
of them is needed to use the encoders. Every path below is relative to the repo
root. The Python drivers and the `run_*_experiments.sh` wrappers resolve the
repo root themselves; `run_eval.sh` and `run_eval_trento.sh` are thin
`scripts/evaluate.py` invocations that expect to be run **from the repo root**
(they pass `--data-root ./data/raw`).

## Table 2 — CoFFE and the MFT control

| Driver | Produces |
|---|---|
| `run_mae_experiments.{py,sh}` | all six CoFFE MAE cells — **both** modalities (`*_mae.yaml` for HSI+LiDAR, `*_mae_hsi.yaml` for HSI-only) |
| `run_hsi_only_experiments.{py,sh}` | the nine CoFFE **SimMIM** HSI-only cells (3 scenes × 3 regimes); the MAE HSI-only cells come from the driver above |
| `run_mft_original_spatial_experiments.{py,sh}` | MFT control, SimMIM token |
| `run_mft_original_mae_experiments.{py,sh}` | MFT control, MAE |
| `run_eval.sh`, `run_eval_trento.sh` | shell wrappers around `scripts/evaluate.py` for a single checkpoint |

The **means** in Table 2 come from single canonical runs evaluated at a
mid-schedule checkpoint (epoch 950 Houston / 975 Trento+MUUFL for CoFFE, with
Houston SimMIM band HSI+LiDAR at 800; epoch 950 for all six MFT cells). Each
cell's exact recipe, including its evaluated epoch, is in
`configs/{coffe,mft}/<scene>_<regime>[_hsi].yaml`; `docs/refactor/AUDIT.md` §3
D1 is the per-cell provenance table.

## Table 2 ± — the 5-seed significance experiment

`run_significance_experiment.py` orchestrates `sig_pretrain_worker.py` and
`sig_eval_worker.py` over the seeds and groups declared in
`sig_significance_config.py` (`EPOCHS = 700`, seeds `[42, 123, 456, 789, 1011]`),
then `scripts/reports/aggregate_significance.py` writes
`results/significance_report.json`. This is a *separate* experiment from the
means — see `results/README.md`.

## Table 3 — HyperSIGMA

| Driver | Produces |
|---|---|
| `run_hypersigma_spatial_pca100.{py,sh}` | the 11×11 patch-native PCA-100 family |
| `run_native_sem_pad_experiments.sh` | the 64×64 backbone-native SEM-only (pad) family |

`run_hypersigma_spatial_pca100.py` uses `configs/hypersigma/<scene>_patchnative_joint_sem.yaml`
as its base and overrides `adapt_mode: spatial_only`, the 100-band spatial
front-end (`spat_resample_to: 100` + the `pca_<scene>_100band` paths) and the
schedule (lr 1e-5, batch 128) in code, so the 11×11 spatial row has a script
path rather than its own config.

### Table 3 coverage, row by row

| Table 3 row | Committed reproduction path |
|---|---|
| 64×64 pad / upscale, frozen, spatial + spectral (12 cells) | **none committed** — the source run `hypersigma_native_ablation_run1` was driven from `notebooks/evaluate_hypersigma_native.ipynb`. No adaptation is involved (frozen backbone); the cells are eval-flag variations. |
| 64×64 pad, SEM-only (3 cells) | `run_native_sem_pad_experiments.sh` + `configs/hypersigma/<scene>_backbonenative_pad_sem_only.yaml` |
| 11×11 spatial (3 cells) | `run_hypersigma_spatial_pca100.{py,sh}` (config base + in-code `spatial_only` override) |
| 11×11 spectral (3 cells) | **none committed** — the source runs (`hypersigma_baseline_spectral_only_run1`, `hypersigma_{trento,muufl}_spectral_only_run1`) were driven from `notebooks/evaluate_hypersigma.ipynb`. |
| 11×11 joint+SEM (3 cells) | **partially committed** (corrected in phase 6). All three cells ran with the **100-band** spatial front-end. Houston has the matching config, `configs/hypersigma/houston_patchnative_pca100_joint_sem.yaml`, but the run overrode its schedule (2000 epochs / batch 128 / lr 1e-5 / min_lr 5e-7 vs the file's 3000 / 64 / 1.5e-4 / 1e-6). Trento and MUUFL have only the 3-band `<scene>_patchnative_joint_sem.yaml`; their runs added `spat_resample_to: 100` and their own schedule at launch. In all three cases `experiments/hypersigma_adapt_<scene>_pca100_joint_sem_run1/pretrain_overrides.yaml` is the authority. |

The two "none committed" rows are a **known gap**, recorded at the phase-5 gate:
their settings were not reconstructed into configs, because inventing settings
for a frozen paper run is worse than naming the notebook that produced it.
`docs/refactor/AUDIT.md` §3 D2 is the cell-by-cell trace: which run produced
each cell, the protocol drift (Table 3 used 2000 episodes against Table 2's
1000, and the Houston 11×11 spectral cell was evaluated at `k_query=30` — see
also `PAPER_CANON.md` §8 D18), and the coverage gaps (no MUUFL/Trento
`backbone_native` adaptation was ever completed).

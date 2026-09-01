# CoFFE — Project Overview

*Few-shot, multimodal hyperspectral land-cover classification.*

This document captures the current state of the project for presentation: the
setup, the main model, the self-supervised pretraining and its ablations, the
HyperSIGMA foundation-model baseline, the adaptation that was done, and the
finetuning currently running. Compiled numbers live in
[RESULTS.md](RESULTS.md) / [RESULTS.json](RESULTS.json).

---

## 1. Problem & motivation

We classify land cover from **hyperspectral imagery (HSI)** plus a co-registered
**auxiliary modality** (LiDAR elevation), under a **few-shot** regime: at test
time a class is defined by only *K = 5* labelled examples. Models are evaluated
by **nearest-class-mean on frozen features** over many random *N*-way *K*-shot
episodes — there is no per-task gradient finetuning at eval time; a class mean is
the average of its support features and queries go to the nearest class mean.

The research question: **what self-supervised pretraining yields HSI features
that transfer best to few-shot tasks**, and how does a purpose-built encoder
(CoFFE) compare to adapting a large pretrained HSI **foundation model**
(HyperSIGMA).

## 2. Datasets

| Dataset | HSI bands | Aux (LiDAR) | Classes | Notes |
|---|---:|---:|---:|---|
| **Houston** | 144 | 1 (elevation) | 15 | 2013 GRSS DF; urban scene |
| **Trento** | 63 | 1 (elevation) | 6 | rural/agricultural |
| **MUUFL** | 64 | 2 (elevation rasters) | 11 | Gulfport campus |

All data are pre-cut into **11×11 spatial patches** centred on each labelled
pixel; a sample is `{"hsi": [C_hsi,11,11], "aux": [C_aux,11,11]}`. Loaders live in
[coffe/data/datasets/](../../coffe/data/datasets/) (`HoustonPatchedDataset`,
`TrentoPatchedDataset`, `MUUFLPatchedDataset`), with a `CombinedPatchedDataset`
that channel-pads for multi-dataset pretraining.

## 3. Pipeline & experiment layout

The workflow is **pretrain → evaluate**, one directory per experiment under
`experiments/<name>/`:

```
experiments/<name>/
  pretrain_config.yaml      # fully-resolved config (eval auto-reads arch from here)
  pretrain_metadata.json    # timestamp, git SHA, device, wall-clock, loss history, status
  checkpoints/              # checkpoint.pth (best val), checkpoint_epoch_*.pth
  evaluations/<eval_name>/
    results.json            # OA / AA / Kappa (+ per-class), with 95% CI
    eval_config.json, eval.log
```

Evaluation is orchestrated by [coffe/runners/eval_runner.py](../../coffe/runners/eval_runner.py),
which auto-loads the encoder architecture from `pretrain_config.yaml` so eval
params can't drift from how the model was trained.

**Eval protocol.** 5-shot, *N*-way (N = #classes), 1000 episodes for CoFFE
(`k_query=100`); Euclidean distance to the class means in every paper run
(cosine with temperature 10 is a non-paper option).
**The projection head is discarded at eval time** (it exists only to shape the
pretraining objective) — `use_projection` defaults to `False` for evaluation.

## 4. Main model — CoFFE

[coffe/models/coffe.py](../../coffe/models/coffe.py). A compact
spectral-spatial transformer encoder built for few-shot transfer.

- **Unified tokenization.** HSI and aux bands are concatenated on the channel
  axis (when `use_aux=True`), passed through a `ChannelTokenizer`
  (`Conv2d(C, D, 1) + BN + GELU`) then a `SpatialTokenizer`, giving one token per
  pixel: `[B, H·W, D]`. Components in
  [coffe/models/components/](../../coffe/models/components/).
- **Class-agnostic CLS token** + learnable positional embeddings, then a small
  **transformer encoder** (`embed_dim=128`, `num_heads=2`, `num_layers=2` in
  every paper run; PAPER_CANON §2).
- **Class-token folding.** Each patch token is nudged toward the class-agnostic
  embedding: `z̄ⱼ = zⱼ + w · z_cls` (`w` = the `lambda_factor` config key,
  0.5 in every paper run) — `CoFFE.eval_patch_embeddings`. This is live on the
  paper's eval path, so the pooled feature is
  `z = mean_j(patch_emb_j) + 0.5 · cls_emb` (PAPER_CANON §8 D3).
- **Projection head** (MLP, optional L2-norm) — **train-time only**, removed for
  evaluation.
- **Few-shot head.** Pooling over patch tokens (uniform, or Gaussian
  centre-weighted when `pool_sigma` is set) → class means over the support →
  Euclidean (or cosine) distance → logits. The live implementation is the
  episode loop in [scripts/evaluate.py](../../scripts/evaluate.py);
  `CoFFE.forward_episode` is an unexecuted duplicate (audit R10).

Key knobs: `embed_dim`, `lambda_factor`, `temperature=10`, `distance_metric`,
`pool_sigma`, `use_aux`, `use_projection`. See
[docs/EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md) for the protocol write-up.

## 5. Self-supervised pretraining & ablations

Two objectives wrap the CoFFE encoder; both are driven by
[scripts/pretrain.py](../../scripts/pretrain.py) (AdamW, cosine schedule +
warmup, grad-clip 1.0; `lr` is per-config — 1.5e-4 in most, 1.5e-5 in
`configs/coffe/houston_simmim.yaml`). Schedule lengths are per-run: 1500 epochs
for the canonical runs, 3000 for Houston MAE and Houston MFT, 2000 for
`houston_enhanced_spectral_run2` (PAPER_CANON §8 D17).

**A. SimMIM masked modelling**
([coffe/pretrain/simmim.py](../../coffe/pretrain/simmim.py),
`SimMIMPretrainModel`). Two composable masks with an MLP decoder
that reconstructs the full cube (MSE over masked entries, optional center-weighted):
  - **Band masking** — per-(pixel, band) Bernoulli on the raw input, masked
    values replaced by learnable per-channel fills (`band_mask_ratio`).
  - **Spatial-token masking** — in-place whole-pixel-token masking after
    tokenization (`spatial_mask_ratio`).
  - **Regimes (the masking ablation):** SimMIM band (0.85 / 0.0), SimMIM token
    (0.0 / 0.75), SimMIM band+token (0.85 / 0.75) — legacy ids `spectral`,
    `spatial`, `both`. One Table 2 cell used 0.75 band (PAPER_CANON §8 D19).
    See [docs/PRETRAINING.md](../PRETRAINING.md).

**B. MAE** ([coffe/pretrain/mae_pretrain.py](../../coffe/pretrain/mae_pretrain.py),
`MAEPretrainModel`). Classic He-et-al. recipe in the unified token space: remove
`mask_ratio=0.75` of tokens, asymmetric transformer decoder
(`decoder_dim=64, depth=4, heads=4`), optional per-token `norm_pix_loss`. No
projection head.

**Ablation axes** (driven by
[scripts/reproduce/run_hsi_only_experiments.py](../../scripts/reproduce/run_hsi_only_experiments.py)
and [scripts/reproduce/run_mae_experiments.py](../../scripts/reproduce/run_mae_experiments.py)):

1. **Dataset** — Houston / Trento / MUUFL.
2. **Modality** — `HSI+LiDAR` vs `HSI-only` (the `*_no_lidar` configs).
3. **Masking regime** — SimMIM band / token / band+token, or MAE.

Result: adding LiDAR helps on every dataset, and the SimMIM token / band+token
regimes beat band-only (see [RESULTS.md](RESULTS.md)).

## 6. HyperSIGMA implementation (foundation-model baseline)

[coffe/models/hypersigma/](../../coffe/models/hypersigma/). HyperSIGMA is a large pretrained
HSI foundation model with a **dual-branch ViT** design, integrated here as a
**frozen few-shot baseline**.

- **SpatViT branch** ([spat_vit_branch.py](../../coffe/models/hypersigma/spat_vit_branch.py))
  — spatial pathway. HSI is PCA-reduced to **3 components**
  ([preprocessing.py](../../coffe/models/hypersigma/preprocessing.py), `PCAPreprocessor`,
  optionally `PCAStandardize`); a ViT-Base body (frozen) produces FPN spatial
  features. `spat_patch_k=3`.
- **SpecViT branch** ([spec_vit_branch.py](../../coffe/models/hypersigma/spec_vit_branch.py))
  — spectral pathway over raw bands, with an `AdaptiveAvgPool1d` to **100 spectral
  tokens** (so any band count fits). ViT-Base body frozen.
- **SEM fusion** ([sem.py](../../coffe/models/hypersigma/sem.py)) — gated spatial-spectral
  enhancement across 4 stages → a **512-d** fused feature.
- **Eval wrapper** ([few_shot.py](../../coffe/models/hypersigma/few_shot.py),
  `HyperSIGMAFewShot`) — selects the feature used for prototypes:
  `fused` (512-d SEM), `spat_pool` (768-d spatial), or `spec_pool` (768-d
  spectral), L2-normalised. No learnable parameters at eval.

Only the small **random-init** pieces (patch-embed, positional embeddings,
`spat_map`, `l1`, deformable offsets) and the SEM/decoders are ever trained; the
transformer bodies stay frozen.

## 7. Adaptation (done)

"Adaptation" = a light, **Level-2 MAE finetuning** of HyperSIGMA's random-init
components to each target dataset's *unlabelled* pixels. Two steps:

1. **Fit PCA** ([scripts/fit_pca_hypersigma.py](../../scripts/fit_pca_hypersigma.py))
   — 3-component PCA per dataset → `checkpoints/hypersigma/pca_<ds>_3band.pkl`
   (+ per-channel `_stats.pkl` for input standardization).
2. **MAE adaptation**
   ([coffe/pretrain/hypersigma_mae.py](../../coffe/pretrain/hypersigma_mae.py),
   `HyperSIGMAMaskedAdaptation`; runner
   [scripts/adapt_hypersigma.py](../../scripts/adapt_hypersigma.py) /
   notebook entry [coffe/runners/adapt_runner.py](../../coffe/runners/adapt_runner.py)) — upstream-style
   **token-level masking** (`mask_ratio=0.75`, HyperGlobal-450K default) with a
   learnable `mask_token` at masked positions and per-token z-scored
   reconstruction targets. A single `adapt_mode` knob controls both adaptation and
   eval feature:
   - `spatial_only` — SpatViT pieces + spatial decoder (16 tokens → 27-px targets).
   - `spectral_only` — SpecViT pieces + spectral decoder (100 tokens → 121-px targets).
   - `joint_sem` — both branches **+ SEM** + 3 decoders, loss `L_spat+L_spec+L_fused`.

Configs: `configs/hypersigma/{houston,trento,muufl}_patchnative_joint_sem.yaml`. The
config recipe is 3000 epochs / `lr=1.5e-4` / batch 64; **the launched runs
overrode** to **2000 epochs, `lr=1e-5`, batch 128, warmup 100** (see each
experiment's `pretrain_metadata.json`). Trained on RTX 4090s; the Trento
`joint_sem` run took ~25.8 h wall-clock for 2000 epochs.

## 8. Finetuning being run (in progress)

The HyperSIGMA MAE adaptation jobs are the active finetuning. Status from each
experiment's `pretrain_metadata.json` (as of 2026-06-04):

| Dataset | `spatial_only` | `spectral_only` | `joint_sem` |
|---|---|---|---|
| Houston | ✅ complete (run2; run1 interrupted) | ✅ complete | ✅ adapted (evaluated) |
| Trento | ✅ complete | ⏸ interrupted | ✅ complete |
| MUUFL | ✅ complete | ▶ **running** | — not yet started |

- **Evaluated so far:** Houston (`joint_sem`/`spatial_only`/`spectral_only`) and
  Trento (all three) — numbers in [RESULTS.md](RESULTS.md).
- **Pending:** MUUFL HyperSIGMA evaluations — the MUUFL `spectral_only`
  adaptation is still training; MUUFL few-shot results will follow once adaptation
  finishes and the eval is run.
- **Next:** fill the MUUFL HyperSIGMA row, settle the Houston `spatial_only`
  run-to-run variance (44–53% OA across runs), and fix euclidean as
  the headline metric (euclidean wins consistently for HyperSIGMA features).

## 9. Evaluation & metrics

Each `results.json` reports **OA**, **AA**, **Kappa** as mean / std / 95% CI over
episodes, plus a **per-class** breakdown (`accuracy`, `total/correct_samples`) and
`class_names`. Query sets are class-balanced, so OA == AA. HyperSIGMA results
carry **both** `cosine` and `euclidean` blocks; CoFFE carries the single
configured metric. Compilation: [scripts/compile_results.py](../../scripts/compile_results.py).

## 10. Key files & pointers

| Area | Path |
|---|---|
| Main model | [coffe/models/coffe.py](../../coffe/models/coffe.py) |
| SimMIM pretraining | [coffe/pretrain/simmim.py](../../coffe/pretrain/simmim.py) |
| MAE pretraining | [coffe/pretrain/mae_pretrain.py](../../coffe/pretrain/mae_pretrain.py) |
| HyperSIGMA model | [coffe/models/hypersigma/](../../coffe/models/hypersigma/) |
| HyperSIGMA adaptation | [coffe/pretrain/hypersigma_mae.py](../../coffe/pretrain/hypersigma_mae.py), [scripts/adapt_hypersigma.py](../../scripts/adapt_hypersigma.py) |
| Evaluation | [coffe/runners/eval_runner.py](../../coffe/runners/eval_runner.py), [scripts/evaluate.py](../../scripts/evaluate.py), [scripts/evaluate_hypersigma.py](../../scripts/evaluate_hypersigma.py) |
| Ablation drivers | [scripts/reproduce/run_hsi_only_experiments.py](../../scripts/reproduce/run_hsi_only_experiments.py), [scripts/reproduce/run_mae_experiments.py](../../scripts/reproduce/run_mae_experiments.py) |
| Existing docs | [docs/EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md), [docs/PRETRAINING.md](../PRETRAINING.md) |
| Compiled results | [RESULTS.md](RESULTS.md), [RESULTS.json](RESULTS.json) |

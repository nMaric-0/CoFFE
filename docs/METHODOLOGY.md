# CoFFE — Methodology

*Source document for the methodology section of the MACLEAN'26 workshop paper.*
This write-up is the authoritative prose description of the method, derived
**from the code**, not from older design notes. Two existing documents are
partially stale and must not be used verbatim:

- `docs/ENHANCED_PRETRAINING.md` describes a superseded **four-task** objective
  (separate spatial / spectral / LiDAR / denoise losses). That design was
  **replaced** by a single *unified band+spatial masking* objective; see the
  module header of `pretrain/masked_modeling_enhanced.py` (lines 17–19).
- The single-run config `configs/pretrain/houston_pretrain_enhanced.yaml` uses
  `band_mask_ratio=0.9`, whereas the **masking-regime ablation** that produced
  the reported results uses `0.85` (`scripts/run_hsi_only_experiments.py:51`).
  Quote the ablation values in the paper.

Throughout, "CoFFE encoder" denotes the compact spectral–spatial fusion
transformer in `models/mft_cpea_cosine.py`. The class-aware patch-embedding
adaptation (CPEA; the `z̄ᵢ = zᵢ + λ·z_class` step at
`models/mft_cpea_cosine.py:270`) is **deliberately excluded** from the method:
pooled patch tokens are fed directly to the prototype classifier.

---

## 1. Problem formulation and notation

We address land-cover classification from co-registered **hyperspectral imagery
(HSI)** and an **auxiliary elevation modality (LiDAR)** under a **few-shot**
regime. A labelled example is a spatial patch centred on a pixel,

$$x = \{\, x^{\text{hsi}} \in \mathbb{R}^{C_h \times P \times P},\;
            x^{\text{aux}} \in \mathbb{R}^{C_a \times P \times P} \,\},$$

with patch size $P = 11$, $C_h$ HSI bands and $C_a$ auxiliary channels.

Evaluation is **episodic**. Each episode draws an $N$-way $K$-shot task: a
support set $S = \{(x_i, y_i)\}_{i=1}^{NK}$ with $K = 5$ labelled patches per
class and a query set $Q$ of unlabelled patches over the same $N$ classes.
A frozen encoder $f_\theta$ maps each patch to a feature vector; a class
**prototype** is the mean of its support features,

$$c_n = \frac{1}{K}\sum_{(x_i,y_i)\in S,\, y_i=n} f_\theta(x_i),$$

and a query $x_q$ is assigned to the nearest prototype. **There is no
gradient-based finetuning at evaluation time** and the classifier has **no
learnable parameters** — the only knowledge transferred is the pretrained
encoder. This isolates representation quality from task-specific adaptation,
which is the variable we study.

**Metrics.** Overall Accuracy (OA), Average per-class Accuracy (AA) and Cohen's
$\kappa$, each reported as the mean over episodes with a 95% confidence interval
(Student-$t$). Because query sets are **class-balanced**, OA $\equiv$ AA in every
run; we report OA and $\kappa$. (`scripts/evaluate_cosine.py`.)

**Research question.** Does a *compact, purpose-built* encoder pretrained with
self-supervised masked modeling on the target scene transfer better to these
few-shot tasks than *adapting a large pretrained HSI foundation model*
(HyperSIGMA)? The methodology therefore describes two model families evaluated
under the identical episodic protocol above.

---

## 2. Datasets and preprocessing

| Dataset | HSI bands $C_h$ | Aux $C_a$ | Classes $N$ | Scene |
|---|---:|---:|---:|---|
| Houston (2013 GRSS DF) | 144 | 1 (elevation) | 15 | urban |
| Trento | 63 | 1 (elevation) | 6 | rural / agricultural |
| MUUFL (Gulfport) | 64 | 2 (elevation rasters) | 11 | mixed campus |

(`data/datasets/{houston,trento,muufl}.py`.)

**Patch extraction.** All scenes are cut into $11\times11$ patches centred on
each pixel, yielding $\{x^{\text{hsi}}, x^{\text{aux}}\}$ tensors
(`data/datasets/base.py:93`). For self-supervised pretraining we use **all**
pixels (labelled and unlabelled), `PretrainDataset` (`pretrain/masked_modeling.py:151`).

**Normalization.** Per-channel min–max scaling to $[0,1]$, applied independently
to every HSI band and auxiliary channel (`data/datasets/base.py:63`).

**Multi-dataset pretraining (optional).** `CombinedPretrainDataset`
(`pretrain/masked_modeling.py:195`) zero-pads or truncates each scene to a common
channel count so several datasets can be pretrained jointly. The reported
results pretrain per-dataset.

---

## 3. The CoFFE encoder (unified multimodal fusion)

The encoder (`models/mft_cpea_cosine.py`) is a compact spectral–spatial
transformer. Its defining design choice is **early, token-level fusion**: HSI and
auxiliary bands are concatenated and embedded jointly, rather than encoded by
separate per-modality branches and fused late.

**Unified tokenization.** HSI and auxiliary bands are concatenated along the
channel axis into a single $[B, C_h{+}C_a, P, P]$ tensor (`tokenize`,
`mft_cpea_cosine.py:215`). A **channel tokenizer** (1×1 Conv → BatchNorm → GELU,
`components/tokenizers.py:6`) projects the $C_h{+}C_a$ bands to the embedding
dimension $D$; a **spatial tokenizer** (3×3 Conv → BatchNorm → GELU, then
flatten, `components/tokenizers.py:30`) mixes local spatial context and emits one
token per pixel:

$$[B, C_h{+}C_a, P, P]\;\xrightarrow{\text{1×1 conv}}\;[B, D, P, P]
   \;\xrightarrow{\text{3×3 conv, flatten}}\;[B,\, N{=}P^2,\, D].$$

Each of the $N = 121$ tokens therefore carries information from **both
modalities at the pixel it represents** — fusion is intrinsic to the token, not a
downstream operation.

**Sequence assembly.** A learnable class-agnostic token is prepended and a
learnable positional embedding of length $1{+}N$ is added
(`mft_cpea_cosine.py:134`, `:140`). The class token is *class-agnostic*: it is a
single shared parameter, not a per-class embedding.

**Transformer encoder.** A stack of pre-LayerNorm transformer blocks
(`components/transformer.py:7`), each LN → multi-head self-attention → residual,
LN → MLP (4× expansion, GELU) → residual, followed by a final LayerNorm. The
reported configuration is intentionally small — $D{=}128$, 2 heads, 2 layers —
because the labelled regime is tiny; the repository's base configuration (8
heads, 4 layers) over-parameterises the smaller scenes.

**Projection head (train-time only).** During pretraining a small MLP projection
head with optional $\ell_2$-normalisation (`components/projection.py:8`;
reported: 1 layer, hidden 512, no BatchNorm, $\ell_2$-normalised) shapes the
self-supervised objective. **It is discarded for evaluation**
(`PROJECT_OVERVIEW.md:62`): features for the prototype classifier are taken from
the encoder output, so the projection only influences *how* the encoder is
trained, not the few-shot feature itself.

> **Note (CPEA omitted).** The implementation also contains a class-aware
> patch-embedding adaptation step. We do not use it; this write-up and the paper
> describe the encoder without it.

---

## 4. Self-supervised pretraining

The encoder is pretrained with masked reconstruction over the **unified token
space** of Section 3, exploiting all (mostly unlabelled) pixels. We study one
primary objective (unified masked modeling) and one baseline (MAE).

### 4.1 Unified masked modeling

(`pretrain/masked_modeling_enhanced.py`, `EnhancedMaskedSpectralSpatialModel`.)
Two composable masking operators act on the concatenated HSI+aux input:

1. **Band masking** — `UnifiedBandMasking` (`masked_modeling.py:28`). Each
   *(pixel, band)* entry of the raw $[B, C_h{+}C_a, P, P]$ tensor is masked
   independently with probability $r_b$; masked entries are replaced by a
   **learnable per-channel fill value** so the 1×1 channel tokenizer receives a
   well-defined input. This is a spectral-reconstruction task: visible bands at a
   pixel must predict the hidden ones.

2. **Spatial-token masking** — `SpatialTokenMasking` (`masked_modeling.py:68`).
   After tokenization, a fraction $r_s$ of whole pixel tokens are replaced by a
   learnable mask token. *Implementation detail (disclosed):* unlike the
   token-dropping MAE, all $N$ tokens still pass through the encoder — masked
   positions carry the mask token rather than being removed — so this variant
   yields **no encoder-side compute saving**; it is a spatial-context
   reconstruction signal, not an efficiency mechanism.

When both are active, band masking is applied first, then tokenization, then
spatial-token masking; the loss is taken over the **union** of the two masks.

**Decoder and loss.** A lightweight 2-layer MLP decoder (`masked_modeling.py:110`)
maps each encoded token back to the full $C_h{+}C_a$-dimensional pixel vector.
With binary per-entry mask $m$, prediction $\hat{v}$ and target $v$ (the
unmasked input), the loss is masked mean-squared error,

$$\mathcal{L} = \frac{\sum_{j} w_j\, m_j\, (\hat{v}_j - v_j)^2}{\sum_j m_j},$$

computed **only over masked entries** (`masked_modeling_enhanced.py:204`).
$w_j$ is an optional Gaussian **center-weighting** over the patch
(`recon_sigma`, `masked_modeling_enhanced.py:102`) that emphasises reconstruction
near the patch centre — the pixel that is actually classified — with mean weight
normalised to 1 so loss scale is preserved.

**Masking regimes (the ablation).** Three regimes are defined by the ratio pair
$(r_b, r_s)$ (`scripts/run_hsi_only_experiments.py:51`):

| Regime | $r_b$ (band) | $r_s$ (spatial) | Signal emphasised |
|---|---:|---:|---|
| `spectral` | 0.85 | 0.0 | spectral correlations |
| `spatial`  | 0.0  | 0.75 | spatial context |
| `combined` | 0.85 | 0.75 | both |

### 4.2 MAE baseline

(`pretrain/mae_pretrain.py`, `MAEPretrainModel`.) The classic He et al. recipe in
the same unified token space: a fraction (0.75) of pixel tokens are **removed**
before encoding, an **asymmetric transformer decoder** reconstructs the masked
tokens, with optional per-token normalised-pixel loss. No projection head. This
is the strongest single-objective point of comparison for the unified masked
model.

### 4.3 Optimisation

AdamW, learning rate $1.5\times10^{-4}$ with cosine decay to $10^{-6}$ and a
100-epoch linear warm-up, weight decay 0.05, gradient clipping 1.0, batch size
64, fixed seed 42 with deterministic kernels. Houston is pretrained for 3000
epochs, Trento and MUUFL for 1500 (`configs/pretrain/*_enhanced.yaml`;
`trainers/pretrain_trainer.py:92`).

---

## 5. Parameter-free few-shot evaluation

At evaluation the pretrained encoder is frozen and the projection head removed.
For each episode (`forward_episode`, `mft_cpea_cosine.py:326`):

1. **Token features.** Support and query patches are encoded to patch-token
   embeddings $[\,\cdot\,, N, D]$.
2. **Spatial pooling.** Tokens are pooled to one vector per patch — either a
   uniform mean or a **Gaussian center-weighted** pool (`pool_sigma`,
   `utils/spatial_weights.py`) that up-weights the patch centre.
3. **Prototypes.** Class prototypes are the mean of support features
   (Section 1).
4. **Classification.** Logits are a distance/similarity to prototypes. Two
   options are implemented:
   - **Euclidean:** $\text{logit}_n = -\lVert z_q - c_n \rVert_2^2$.
   - **Cosine:** features are $\ell_2$-normalised and
     $\text{logit}_n = \tau\, \langle z_q, c_n\rangle$ with temperature
     $\tau = 10$.

> **Naming vs. reported metric (disclosed).** The encoder class is named
> `MFTCPEACosine`, but the **headline numbers use euclidean** prototypes
> (`RESULTS.md:23`), which consistently outperform cosine on these features
> (markedly so for HyperSIGMA). The paper should state this and report euclidean
> as the primary metric, with cosine as a secondary comparison.

**Episode protocol.** 5-shot, $N$-way with $N$ equal to the number of classes
in the scene; CoFFE evaluations use $k_{\text{query}}=100$ over 1000 episodes.
Class filtering requires $\ge K{+}Q$ samples per class
(`data/samplers/patched_episode_sampler.py`).

---

## 6. Foundation-model comparison: adapted HyperSIGMA

HyperSIGMA is a large pretrained HSI foundation model, integrated as a **frozen
few-shot baseline** to test the purpose-built-vs-foundation question
(`models/hypersigma/`, `PROJECT_OVERVIEW.md:127`).

**Architecture.** A dual-branch ViT:
- **Spatial branch (SpatViT).** HSI is PCA-reduced to 3 components
  (`PCAPreprocessor`, fit per dataset); a frozen ViT-Base body produces FPN
  spatial features.
- **Spectral branch (SpecViT).** Raw bands are pooled to a fixed 100 spectral
  tokens via `AdaptiveAvgPool1d` (so any band count fits); a frozen ViT-Base body
  encodes them.
- **SEM fusion.** A gated spatial–spectral enhancement module fuses the two
  branches across stages into a single 512-d feature (`sem.py`).

Only the small randomly-initialised pieces (patch-embed, positional embeddings,
projection maps, deformable offsets) and the SEM/decoders are trainable; **the
ViT bodies stay frozen.**

**Adaptation.** A light, Level-2 **MAE finetuning** of the trainable pieces on
each target scene's *unlabelled* pixels (token-level masking 0.75 with a
learnable mask token and per-token z-scored reconstruction targets;
`pretrain/hypersigma_mae.py`). A single `adapt_mode` selects which branch is
adapted and which feature is used at evaluation:
- `spatial_only` — SpatViT pieces + spatial decoder.
- `spectral_only` — SpecViT pieces + spectral decoder.
- `joint_sem` — both branches **plus SEM**, with loss
  $\mathcal{L}_{\text{spat}}+\mathcal{L}_{\text{spec}}+\mathcal{L}_{\text{fused}}$.

**Evaluation.** Identical episodic protocol (Section 5). The prototype feature is
selected from {`fused` 512-d, `spat_pool`, `spec_pool`}, $\ell_2$-normalised; no
learnable parameters at eval (`hypersigma_cosine.py`). Both cosine and euclidean
metrics are reported for HyperSIGMA.

---

## 7. Design rationale

- **Compact encoder.** With only $K=5$ labelled examples per class and a few
  hundred labelled patches per scene, a large encoder over-fits; the 2-layer /
  2-head configuration is a deliberate match to the data budget.
- **Token-level fusion.** Concatenating HSI and LiDAR before embedding lets the
  transformer reason over a single joint token per pixel, capturing
  spectral–elevation correlations from the first layer rather than reconciling
  two separately-encoded streams late.
- **Masked SSL on the target scene.** Masked reconstruction turns the abundant
  *unlabelled* pixels into supervision, which is the only way to learn a strong
  encoder when labels are scarce.
- **Parameter-free evaluation.** A prototype classifier with no learnable parts
  measures the encoder's representation directly and prevents an evaluation-time
  head from masking a weak backbone.

---

## 8. Limitations and threats to validity

These are stated plainly to pre-empt reviewer objections.

- **The unified masked model is not uniformly best.** On Houston the MAE baseline
  beats the unified masked model at matched modality (HSI-only: 69.04 vs. the best
  enhanced regime 65.50 OA, euclidean), and MAE with LiDAR is the overall Houston
  best (72.15; `RESULTS.md`). The contribution is *regime- and scene-dependent*: on
  Trento and MUUFL the enhanced `spatial` regime is the best configuration, and
  across scenes spatial/combined masking and the addition of LiDAR help
  consistently — but unified masking does not dominate MAE everywhere. The paper
  must frame the finding this way rather than as a blanket win.
- **Run-to-run variance is material.** Repeated runs of the same configuration
  differ by several OA points (e.g. Trento enhanced-`spatial` HSI+LiDAR 89.4 vs.
  94.2; Houston HyperSIGMA `spatial_only` 44 vs. 53). Report ranges / multiple
  seeds, not single best runs (`RESULTS.md` "other_runs").
- **Spatial-token masking is not the efficient MAE.** It keeps all $N$ tokens in
  the encoder (Section 4.1); we make no efficiency claim for it.
- **Metric/name mismatch.** See Section 5 — euclidean is the reported metric
  despite the "Cosine" class name.
- **HyperSIGMA coverage is incomplete.** MUUFL HyperSIGMA evaluations were not
  available at the time of writing; comparison claims are scoped to Houston and
  Trento.

---

## 9. Key source files

| Component | Path |
|---|---|
| CoFFE encoder | `models/mft_cpea_cosine.py` |
| Tokenizers | `models/components/tokenizers.py` |
| Transformer | `models/components/transformer.py` |
| Projection head | `models/components/projection.py` |
| Unified masked modeling | `pretrain/masked_modeling_enhanced.py`, `pretrain/masked_modeling.py` |
| MAE baseline | `pretrain/mae_pretrain.py` |
| HyperSIGMA | `models/hypersigma/` |
| HyperSIGMA adaptation | `pretrain/hypersigma_mae.py`, `scripts/adapt_hypersigma.py` |
| Evaluation | `scripts/evaluate_cosine.py`, `lib/eval_runner.py` |
| Ablation drivers | `scripts/run_hsi_only_experiments.py`, `scripts/run_mae_experiments.py` |
| Compiled results | `docs/presentation/RESULTS.md`, `docs/presentation/RESULTS.json` |

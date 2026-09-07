# Feature-width ablation — HyperSIGMA at CoFFE's 128 dimensions

## The question

[`PAPER_CANON.md`](../PAPER_CANON.md) §6 Table 3 evaluates HyperSIGMA on 768-d
features (spatial, spectral) or a 512-d fused SEM feature; CoFFE's eval feature
is 128-d (§2). The paper's headline margins are therefore measured across a
4–6× difference in feature width, under a classifier — nearest class mean from
K=5 supports (§4) — where width is not neutral: five samples estimate a
prototype badly in 768 dimensions, while a wider feature also offers more
directions along which to separate 15 classes. So the comparison invites a
question it cannot answer on its own: is the margin a property of the
encoders, or of the width?

This ablation compresses HyperSIGMA's frozen features to a target width
*before* the prototypes are formed and re-runs the identical protocol, so the
margin can be read at matched capacity.

**No published number changes.** Table 3 is what it is; this is a separate
measurement, and every artifact it writes is marked so the aggregators cannot
mistake one for a published cell.

## What was run

The best published cell of each scene, taken from the frozen runs rather than
from the paper text:

| Scene | Cell | Feature | Published OA | Adaptation |
|---|---|---|---|---|
| Houston | 11×11 patch-native, PCA-100, joint+SEM | fused SEM, **512-d** | 67.48 ± 0.11 | label-free joint+SEM |
| Trento | 64×64 backbone-native, upscale | `spat_pool`, **768-d** | 91.07 ± 0.09 | none (frozen) |
| MUUFL | 64×64 backbone-native, pad | `spat_pool`, **768-d** | 54.80 ± 0.12 | none (frozen) |

Protocol held at Table 3's values throughout: 5-shot, N-way (all classes),
100 queries per class, 2000 episodes, `split="all"`, seed 42, Euclidean NCM
(§4, §8 D18).

## Method

**Encode once.** The encoder is frozen, so a labelled patch has exactly one
feature vector, and the live loop's habit of re-encoding inside every episode
buys nothing: a Houston run pushes 3.15M patches through a ViT-Base pair to
encode 15,029 distinct ones. [`coffe/eval/feature_cache.py`](../coffe/eval/feature_cache.py)
walks the scene once — 13 s where the published run's own log shows 46 min —
and [`coffe/eval/reduced_ncm.py`](../coffe/eval/reduced_ncm.py) indexes that
cache per episode. That is what makes a full sweep affordable: after the cache,
each variant is a projection plus a `cdist`.

**Compress, label-free.** [`coffe/eval/dim_reduction.py`](../coffe/eval/dim_reduction.py)
supplies the maps, none of which can see a label (asserted structurally in
`tests/unit/test_dim_reduction.py`):

- **PCA** to the target width, fit unsupervised on the cached features — the
  best linear map into that width in the least-squares sense, so it is
  deliberately generous to the foundation model. Two fit corpora: all labelled
  features (transductive but label-free) and the MFT train split alone
  (inductive).
- **Gaussian random projection** (Johnson–Lindenstrauss, `R/√dim`), five seeds,
  fitting nothing — the cost of the dimension count on its own.
- **SEM stage blocks**, Houston only: the fused feature is `4 × 128` by
  construction ([`coffe/models/hypersigma/sem.py`](../coffe/models/hypersigma/sem.py)),
  so one stage's block or the mean of the four is a 512→128 compression the
  architecture already implies.

Widths swept: 16, 32, 64, 128, 256, 512 (below each cell's native width), with
**128 the headline** because that is CoFFE's `D`.

**Two metrics per variant.** Native features are L2-normalised by the
evaluator, and projection destroys that. `euclidean_l2` (the primary) restores
unit norm after the projection; `euclidean` is the paper's metric on the raw
projected coordinates. Cosine is recorded too and is by construction identical
for both, being scale-invariant.

**Same episodes, so the deltas are paired.** The plan is drawn from the same
sampler at the same seed as the published run, via a new
`PatchedEpisodeSampler.sample_episode_indices()` that consumes the RNG in the
identical order and returns indices instead of patches. Every variant therefore
sees the control's exact 2000 episodes, and the tables report a *paired* mean
difference with a Wilcoxon signed-rank test — far tighter than comparing two
confidence intervals.

**What makes it trustworthy.** Two gates, both of which held:

1. `tests/integration/test_reduced_ncm_equivalence.py` runs the cached loop and
   the live loop over the same fake, batch-invariant encoder and requires
   *exact* agreement — per-episode OA and κ, per-class blocks, the global
   confusion matrix. So the cached path is the same protocol, not a lookalike.
2. Every cell's control is the published configuration at native width, gated
   against the frozen run's own full-precision mean (not Table 3's 2-dp print,
   which would fold rounding into the drift). Miss it by more than 0.05 pp and
   the run is marked failed, because then the cache or the loop is wrong. All
   three passed — see the first table below.

Three readers walk `experiments/*/evaluations/*/results.json` and would
otherwise absorb these runs as the cells they re-run, since an ablation shares
the published cell's `model_type`, `mode`, geometry and checkpoint:
`scripts/compile_results.py`, `scripts/reports/aggregate_experiment_results.py`
and `scripts/reports/build_experiment_metadata.py`. All three now consult
`describe_feature_reduction` and exclude anything carrying the marker, naming
what they dropped rather than dropping it silently.

## Results

<!-- generated by scripts/reports/render_dim_sweep.py; do not hand-edit -->
<!-- BEGIN generated tables -->
### Control: the cache reproduces the published cell

| Scene | Published cell | Feature | Paper (Table 3) | Frozen run | Re-run | Drift |
|---|---|---|---|---|---|---|
| Houston | 11x11 patch-native, PCA-100, joint+SEM adaptation, fused SEM feature | 512-d | 67.48 ± 0.11 | 67.4789 | 67.4789 | +0.00000 pp |
| Trento | 64x64 backbone-native (upscale), frozen encoder, pooled SpatViT feature | 768-d | 91.07 ± 0.09 | 91.0723 | 91.0473 | -0.02508 pp |
| MUUFL | 64x64 backbone-native (pad), frozen encoder, pooled SpatViT feature | 768-d | 54.80 ± 0.12 | 54.7967 | 54.7967 | +0.00005 pp |

`Paper` is Table 3's printed value; the gate compares against the frozen run's own full-precision mean, so `Drift` is the cache's, not the paper's rounding.

### At CoFFE's width (128-d)

| Scene | Native OA | PCA-128 | Random-proj-128 | CoFFE 128-d (Table 2) | CoFFE minus best HyperSIGMA |
|---|---|---|---|---|---|
| Houston | 67.48 | 68.64 (+1.16) | 66.74 ± 0.48 (-0.74) | 75.30 | +6.66 pp |
| Trento | 91.05 | 91.67 (+0.63) | 90.82 ± 0.22 (-0.23) | 94.19 | +2.52 pp |
| MUUFL | 54.80 | 54.40 (-0.40) | 53.63 ± 0.43 (-1.17) | 67.83 | +13.03 pp |

Δ in parentheses is paired against that cell's own control on the identical 2000 episodes. The last column is CoFFE's published Table 2 cell minus the best of its native width and its PCA-128 value, whichever is higher, so the margin is never flattered by the compression. (The random projection is below native in every scene, so it never enters.)

### Houston — OA by feature width

| Width | Variance kept (pool fit) | PCA (fit: all labelled features) | PCA (fit: MFT train split) | Random projection (mean ± std, 5 seeds) |
|---|---|---|---|---|
| 16 | 85.29% | 66.32 (-1.16) | 66.10 (-1.37) | 58.06 ± 1.77 (-9.42) |
| 32 | 94.92% | 68.48 (+1.00) | 68.48 (+1.00) | 61.92 ± 0.91 (-5.56) |
| 64 | 98.19% | 68.61 (+1.13) | 68.67 (+1.19) | 65.17 ± 1.17 (-2.31) |
| 128 | 99.47% | 68.64 (+1.16) | 68.72 (+1.24) | 66.74 ± 0.48 (-0.74) |
| 256 | 99.89% | 68.65 (+1.17) | 68.74 (+1.26) | 67.15 ± 0.61 (-0.32) |
| **512 (native)** | 100% | **67.48** | | |

SEM stage blocks (the fused feature is 4 x 128 by construction):

| Map | OA | Δ vs 512-d | Paired test |
|---|---|---|---|
| `stage0` | 66.21 | -1.27 pp | p<0.001 |
| `stage1` | 67.02 | -0.46 pp | p<0.001 |
| `stage2` | 67.13 | -0.35 pp | p<0.001 |
| `stage3` | 64.57 | -2.91 pp | p<0.001 |
| `stage_mean` | 67.03 | -0.45 pp | p<0.001 |

### Trento — OA by feature width

| Width | Variance kept (pool fit) | PCA (fit: all labelled features) | PCA (fit: MFT train split) | Random projection (mean ± std, 5 seeds) |
|---|---|---|---|---|
| 16 | 97.61% | 91.52 (+0.47) | 90.57 (-0.47) | 89.74 ± 1.29 (-1.30) |
| 32 | 98.48% | 91.64 (+0.59) | 90.61 (-0.43) | 89.91 ± 1.02 (-1.13) |
| 64 | 98.98% | 91.67 (+0.62) | 90.65 (-0.40) | 90.63 ± 0.63 (-0.42) |
| 128 | 99.35% | 91.67 (+0.63) | 90.68 (-0.37) | 90.82 ± 0.22 (-0.23) |
| 256 | 99.65% | 91.68 (+0.63) | 90.71 (-0.34) | 91.01 ± 0.21 (-0.04) |
| 512 | 99.89% | 91.68 (+0.63) | 90.73 (-0.31) | 91.09 ± 0.16 (+0.04) |
| **768 (native)** | 100% | **91.05** | | |

### MUUFL — OA by feature width

| Width | Variance kept (pool fit) | PCA (fit: all labelled features) | PCA (fit: MFT train split) | Random projection (mean ± std, 5 seeds) |
|---|---|---|---|---|
| 16 | 87.10% | 51.97 (-2.82) | 51.79 (-3.00) | 51.16 ± 2.07 (-3.64) |
| 32 | 91.95% | 53.46 (-1.34) | 53.44 (-1.35) | 50.95 ± 1.05 (-3.85) |
| 64 | 95.42% | 54.16 (-0.64) | 54.11 (-0.69) | 53.79 ± 0.50 (-1.01) |
| 128 | 97.70% | 54.40 (-0.40) | 54.39 (-0.41) | 53.63 ± 0.43 (-1.17) |
| 256 | 99.10% | 54.48 (-0.32) | 54.48 (-0.31) | 54.55 ± 0.25 (-0.25) |
| 512 | 99.89% | 54.49 (-0.31) | 54.50 (-0.29) | 54.48 ± 0.26 (-0.32) |
| **768 (native)** | 100% | **54.80** | | |

### Without re-normalising after projection

The `euclidean` block: the paper's metric on the raw projected coordinates, whose norms shrink and vary. Reported at 128-d only.

| Scene | PCA-128 | Random-proj-128 |
|---|---|---|
| Houston | 67.47 (-0.01) | 66.77 ± 0.53 (-0.71) |
| Trento | 91.05 (-0.00) | 90.82 ± 0.23 (-0.23) |
| MUUFL | 54.75 (-0.05) | 53.66 ± 0.47 (-1.14) |
<!-- END generated tables -->

### Reading it

**Feature width is not the confound.** On the paper's own metric, with no
re-normalisation, PCA to 128 dimensions moves OA by **−0.005 pp** (Houston),
**−0.001 pp** (Trento; p = 0.61, not significant) and **−0.052 pp** (MUUFL).
The compression is free — 128 principal components already carry 99.47 %,
99.35 % and 97.70 % of each feature's variance. For this task HyperSIGMA's
512-d and 768-d eval features contain no more than ~128 dimensions of usable
structure.

**Matched at 128-d, the paper's margins stand.** Taking whichever is higher for
HyperSIGMA, its native width or its compression:

| Scene | Published margin | Margin at 128-d |
|---|---|---|
| Houston | +7.82 | **+6.66** |
| Trento | +3.12 | **+2.52** |
| MUUFL | +13.03 | **+13.03** |

Houston's margin narrows by 1.2 pp and Trento's by 0.6; MUUFL's does not move.
All three keep their sign and their order of magnitude, and CoFFE remains ahead
at equal feature width with ~2 orders of magnitude fewer parameters (§6).

**What the small gains actually are.** Where compression *helps* — +1.16 pp on
Houston, +0.63 on Trento — the cause is the re-normalisation, not the width:
the same projection without it is flat to within 0.05 pp. Projecting onto the
top-128 subspace and restoring unit norm is a marginally better geometry for a
5-shot class mean than the full-width unit sphere. It is worth knowing that this
is available to the foundation-model route for free, and it is not enough to
change the comparison.

**How far below 128 you can go is scene-dependent.** A random projection —
which fits nothing — costs 0.2–1.2 pp at 128-d and 1.3–9.4 pp at 16-d (Trento
−1.30, MUUFL −3.64, Houston −9.42). PCA is far more tolerant, and its tolerance
tracks how concentrated the features are: Trento keeps 97.6 % of its variance
in 16 components and its PCA-16 is *above* its control (+0.47), while Houston
(85.3 % at 16) loses 1.16 and MUUFL (87.1 %) loses 2.82. So 128-d is
comfortable everywhere and PCA-32 already is for all three, but only Trento
tolerates 16. The seed spread (±0.2–0.5 pp at 128-d, ±1–2 pp at 16-d) is why
this was run over five projections rather than one.

**SEM stages are near-interchangeable, except the deepest.** Houston's stage 0,
1, 2 blocks and the 4-stage mean all land within 0.5 pp of the full 512-d fused
feature; stage 3 loses 2.91 pp. Whatever the fusion contributes to this cell, it
is not concentrated in the final stage's block.

## Caveats

**The PCA fit is transductive, on purpose.** The `pool` corpus is the scene's
own labelled features — no labels, but the same pixels the queries come from.
That is generous to HyperSIGMA by design: it upper-bounds what any 128-d linear
compression of these features can do. The `train` corpus (the MFT train split
alone) is the inductive counterpart, and the two agree closely enough that
nothing in the conclusion rests on the transduction.

**Trento's train split is thin.** 819 patches, so a `pca_train` basis of 512
components is estimated from barely more samples than dimensions. Read the
Trento `pca_train` row at 256 and 512 with that in mind; the 128-d headline is
comfortable.

**The cache is not bit-exact with the live loop.** A cached feature is computed
in a batch of 256 rather than the live loop's per-episode batches, so float
reduction order differs. Neither branch has batch-coupled state, and the three
controls land within 0.03 pp of their frozen runs, but that is why the control
gate carries a tolerance rather than demanding equality. The same applies
run-to-run on one machine: re-running the whole sweep over the *same* caches
reproduced every OA in these tables to two decimals, moving only one
five-seed standard deviation (Houston, 16-d: 1.76 → 1.77) — cuBLAS kernel
selection, not a protocol difference.

**This is an ablation, not a correction.** Table 3 keeps its numbers, and its
runs keep their provenance. `results/hypersigma_dim_sweep.json` is regenerable
output, not a frozen artifact (see [`results/README.md`](../results/README.md)).

## Re-running

```bash
# all three cells, full sweep (~10 min on one RTX 4090, most of it the caches)
python scripts/experiments/run_hypersigma_dim_sweep.py --scenes all --device cuda:0

# just gate the cache against the published cells, no variants
python scripts/experiments/run_hypersigma_dim_sweep.py --scenes all --control-only

# rebuild the aggregate and the tables from artifacts already on disk
python scripts/experiments/run_hypersigma_dim_sweep.py --aggregate-only
python scripts/reports/render_dim_sweep.py
```

Caches land in `experiments/hypersigma_dimreduce_run1/features/` and are reused
when their sidecar matches the cell's encoder config exactly; `--refresh-cache`
forces a re-encode.

## Surprise found on the way — proposed as PAPER_CANON §8 D21

**Trento's labelled-pixel count in Table 1 does not match the data on disk.**
PAPER_CANON §5 (paper Table 1) gives Trento 30,414 labelled pixels. The
pre-patched MFT files hold 819 (`TrLabel.mat`) + 29,395 (`TeLabel.mat`) =
**30,214**, and every Trento eval log in `experiments/` records
`Loaded trento (all): 30214 samples` — including the runs behind the published
Table 3 Trento cells. Houston (2,832 + 12,197 = 15,029) and MUUFL
(2,683 + 51,004 = 53,687) match §5 exactly.

So the discrepancy is 200 pixels in one scene's dataset description, not in any
evaluated number: the episodes were always drawn from the 30,214 that exist.
Nothing was changed. Recorded here for Nikola to fold into §8 as D21 with
whatever wording the paper's own bookkeeping calls for.

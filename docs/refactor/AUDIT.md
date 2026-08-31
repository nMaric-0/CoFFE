# AUDIT.md — Phase 1: what every file in this repo is for

Branch `refactor/cleanup`, on top of `f0025c9` (phase-0 baseline).
Nothing outside `docs/refactor/` and `tools/refactor/` was modified.

Companion machine-readable output: **`docs/refactor/manifest.json`**
(213 records, one per tracked file). Raw closure data:
**`docs/refactor/closure.json`**.

Tools added this phase:
- `tools/refactor/closure.py` — static import-closure tracer over named root
  sets; parses `.ipynb` code cells; follows ancestor package `__init__.py`
  edges (an import of `a.b.c` executes `a/__init__.py` and `a/b/__init__.py`).
- `tools/refactor/build_manifest.py` — encodes the verdicts below into
  `manifest.json` so it can be regenerated and diffed.

---

## 1. Entry-point map

### 1.1 Runnable entry points

| Kind | Files |
|---|---|
| CLI, argparse + `__main__` | `scripts/{pretrain_enhanced,evaluate_cosine,adapt_hypersigma,evaluate_hypersigma_cosine,fit_pca_hypersigma}.py` |
| Group runners (argparse) | `scripts/run_{mae,hsi_only}_experiments.py`, `scripts/run_mft_original_{mae,spatial}_experiments.py`, `scripts/run_hypersigma_spatial_pca100.py` |
| Significance pipeline | `scripts/run_significance_experiment.py` → `sig_{pretrain,eval}_worker.py` (config: `sig_significance_config.py`) → `aggregate_significance.py` |
| Exploratory pipelines | `scripts/run_{ablation,combo,bestcfg}.py` → `*_{pretrain,eval}_worker.py` → `aggregate_*.py` |
| Aggregators / compilers | `scripts/{compile_results,compile_mft_faithful_results,gather_requested_results,gather_native_pca100_raw,build_native_pca100_report,aggregate_experiment_results,build_experiment_metadata}.py` |
| Shell drivers | `scripts/run_{cosine_eval,trento_cosine_eval,mae_experiments,hsi_only_experiments,mft_original_mae_experiments,mft_original_spatial_experiments,hypersigma_spatial_pca100,native_sem_pad_experiments}.sh`, `rerun_mft_faithful_eval_ep1500.sh`, `download_{data,hypersigma_checkpoints}.sh` |
| Notebook API (`lib/`) | `run_pretrain` (`lib/pretrain_runner.py`), `run_evaluation` + `run_hypersigma_evaluation` (`lib/eval_runner.py`), `run_adapt_hypersigma` (`lib/adapt_runner.py`), `ExperimentLogger` (`lib/experiments.py`) |
| Notebooks | 16 `.ipynb` (4 base + 9 duplicates + 3 variants) |

`lib/*` is a thin wrapper layer: `lib/eval_runner.py:209` imports
`scripts.evaluate_cosine.run_evaluation`; `lib/adapt_runner.py:86` imports
`scripts.adapt_hypersigma.run_adapt`. So the notebook API and the CLI share one
implementation — good news for the phase-2 equivalence harness, which can pin
either surface.

### 1.2 Dynamic dispatch — checked, and it is *not* dynamic

`scripts/pretrain_enhanced.py:297` (`model.name`, default `"mft_cpea"`) and
`:302` (`objective`, default `"enhanced"`) dispatch on config strings, and
`scripts/evaluate_cosine.py:149` does the same. In every case the dispatch is
an `if/elif` chain over **statically imported** classes
(`pretrain_enhanced.py:40-45`, `evaluate_cosine.py:52`), so the static closure
is complete. A repo-wide grep found **no** `importlib`, `__import__`
(except a cosmetic `__import__('os')` at `run_hsi_only_experiments.py:105`),
`exec`, or config-driven module loading. `EXTRA_EDGES` in `closure.py` is
therefore empty.

### 1.3 Paper artifact → root mapping (verified against the JSONs, not assumed)

All 30 Table 2 cells and all 24 Table 3 cells were located by exact numeric
match on parsed JSON (§3, D1/D2). The mapping is **not** what the skill brief
assumed:

| Paper artifact | Actual provenance |
|---|---|
| Table 2 **means** | single canonical runs, via `experiments/aggregated_results.json` + `experiment_metadata.json` (CoFFE) and `experiments/mft_faithful_results.json` (MFT) |
| Table 2 **± values** | across-seed std from the 5-seed significance experiment: `experiments/significance_report.json` (18 cells) and `experiments/significance_report copy.json` (12 cells) |
| Table 3 (all 24 cells) | `experiments/_report_raw.json` (20 cells) + `experiments/aggregated_results.json` (3) + on-disk `experiments/*/evaluations/*/results.json` (1); aggregated by `experiments/hypersigma_native_sem_pca100_report.json` |
| `docs/presentation/RESULTS.{md,json}` | **superseded** for HyperSIGMA; for CoFFE it carries the right means with a *within-run* CI instead of the paper's across-seed std |

**Table 2 is a two-source composite**: mean from one run, ± from a different
5-seed experiment at a different epoch. Nothing in the repo states this. It is
the single most important thing the reproduction docs must say.

---

## 2. Import closure and classification

### 2.1 Counts

126 traceable files (`.py` + `.ipynb`, excluding `third_party/`):

| Root set | Closure size |
|---|---|
| `paper_significance` | 63 |
| `paper_group_runners` | 62 |
| `paper_hypersigma` | 57 |
| `paper_core_cli` | 44 |
| `paper_compile` | 5 |
| **PAPER union** | **79** |
| `exploratory` | 74 |
| `notebooks_keep` | 65 |
| `tests` | 52 |

Manifest classification over all 213 tracked files:

| Class | Count |
|---|---|
| PAPER | 124 |
| INFRA | 36 |
| EXPLORATORY | 19 |
| VENDORED | 13 |
| DEAD | 11 |
| DUPLICATE | 10 |

Verdicts: 158 keep, 42 delete, 13 rename.

**Key negative result:** `exploratory − PAPER = ∅` for library code. The 15
ablation/combo/bestcfg scripts reach no module the paper routes do not already
reach. Same for every notebook. So the D9 decision is purely about which
*driver scripts* ship — it cannot orphan library code either way.

### 2.2 The `__init__.py` re-export trap (this is how D10 resolves)

Naive closure marks `models/backbones/*` and `models/wrappers/*` as reachable,
because `models/__init__.py:5` and `:12` re-export them and every entry point
does `from models import MFTCPEACosine`. But **import-reachable is not used**.
A symbol-level scan (every public class/function vs every tracked `.py`/`.ipynb`)
shows these form a **closed island**:

```
models/backbones/base.py         <- models/__init__.py, its 3 siblings, models/wrappers/pretrain_wrapper.py
models/backbones/hsi_baseline.py <- models/backbones/__init__.py  ONLY
models/backbones/mft_channel.py  <- models/backbones/__init__.py, mft_pixel.py, pretrain_wrapper.py
models/backbones/mft_pixel.py    <- models/backbones/__init__.py  ONLY
models/wrappers/pretrain_wrapper.py <- models/wrappers/__init__.py ONLY
```

No symbol (`BackboneBase`, `MFTChannelBackbone`, `MFTPixelBackbone`,
`HSIBaselineBackbone`, `PretrainWrapper`, `MAEDecoder`, `ContrastiveHead`)
is referenced anywhere outside the island. Deleting the 7 files together with
the two re-export blocks in `models/__init__.py` is behavior-preserving for
every entry point. → **DEAD**.

### 2.3 Truly unreached (nobody imports them at all)

| File | Evidence |
|---|---|
| `data/transforms/__init__.py`, `data/transforms/augmentations.py` | no in-repo importer of `data.transforms`. The notebook `Compose` hits are a false positive — the only occurrence is the English word in a comment ("Compose the final overrides dict"). 0 % coverage at baseline. Sole `torchvision` consumer. |
| `utils/checkpoints.py` | no importer. Its `fix_state_dict_keys` has a **live twin** at `scripts/evaluate_cosine.py:107`, which is what actually runs (`:202`). The `fix_state_dict_keys` mentions in `pretrain/{mae_pretrain,mft_mae,mft_spatial_mae}.py` and `tests/test_mft_original_shapes.py:138` are docstrings/comments, not calls. |
| `setup.py` | not imported; placeholder metadata. |
| `tests/__init__.py` | package marker — INFRA, keep. |

### 2.4 D10 candidates that are *live* (the seeded list is partly wrong)

| Candidate | Verdict | Evidence |
|---|---|---|
| `pretrain/masked_modeling.py` | **PAPER** | `UnifiedBandMasking` ← `masked_modeling_enhanced.py`, `scripts/adapt_hypersigma.py`; `SpatialTokenMasking`/`MLPDecoder` ← `masked_modeling_enhanced.py`, `mft_spatial_mae.py`; `PretrainDataset`/`CombinedPretrainDataset` ← `trainers/pretrain_trainer.py` |
| `pretrain/mae_pretrain.py` | **PAPER** | `MAEPretrainModel` ← `scripts/pretrain_enhanced.py:42` |
| `pretrain/decoders.py` | **PAPER (partial)** | `TransformerDecoder` ← `mae_pretrain.py`, `mft_mae.py`. But `SimpleMLPDecoder`, `TwoLayerMLPDecoder`, `build_decoder` are referenced only by `pretrain/__init__.py`'s re-export → dead *symbols* in a live file. Proposal only (see gate). |
| `trainers/pretrain_trainer.py` | **PAPER** | `PretrainTrainer` ← `scripts/pretrain_enhanced.py:45` |

### 2.5 Test-only liveness

None found. Every module imported by a test is also imported by paper code.
No test/module pair needs to be retired together.

---

## 3. D1–D12 verdicts

### D1 — Epoch count — **CONFIRMED, and worse than stated**

The paper's "1500 epochs" describes the *schedule*, not the evaluated
checkpoint. Three separate epoch regimes are in play:

1. **Pretraining schedule** — `epochs: 1500` for all SimMIM runs on all three
   scenes (matches the paper). **Exceptions: 3000** for Houston MAE (both
   modalities) and Houston MFT (both objectives), and **2000** for
   `houston_enhanced_spectral_run2`. Verified from each run's own
   `pretrain_config.yaml` and `pretrain_metadata.json` `history.epochs_run`
   (e.g. `experiments/houston_mae_lidar/pretrain_metadata.json` → `epochs_run: 3000`,
   and `checkpoints/` really does hold up to `checkpoint_epoch_3000.pth`).
2. **Evaluated checkpoint** — **epoch 950 on Houston, 975 on Trento and MUUFL,
   800 for one cell**. Never the final checkpoint. From each
   `evaluations/*/eval_config.json` `checkpoint` field.
3. **`EPOCHS = 700`** (`scripts/sig_significance_config.py:46`) — the *separate*
   5-seed experiment that supplies only the ± column.

`scripts/rerun_mft_faithful_eval_ep1500.sh` did produce `*_eval_ep1500` evals,
and their numbers are **not** in Table 2 (Houston MFT SimMIM token: ep1500
gives 67.70, the paper reports 67.49 = ep950). That script documents a recipe
the paper did not use.

Full provenance, every cell exact:

| Table 2 row | Scene | Paper OA | Mean source run | eval subdir | ck epoch | sched epochs | Paper ± | ± source cell (5-seed std) |
|---|---|---|---|---|---|---|---|---|
| CoFFE SimMIM band HSI | Houston | 58.91 | `houston_enhanced_spectral_no_lidar` | `houston_enhanced_spectral_no_lidar_eval` | **950** | 1500 | ±0.9 | `sig:hsi_only/houston/spectral` |
| CoFFE SimMIM band HSI | Trento | 87.04 | `trento_enhanced_spectral_no_lidar` | `trento_enhanced_spectral_no_lidar_eval` | **975** | 1500 | ±0.4 | `sig:hsi_only/trento/spectral` |
| CoFFE SimMIM band HSI | MUUFL | 55.02 | `muufl_enhanced_spectral_no_lidar` | `muufl_enhanced_spectral_no_lidar_eval` | **975** | 1500 | ±0.7 | `sig:hsi_only/muufl/spectral` |
| CoFFE SimMIM band HSI+LiDAR | Houston | 60.21 | `houston_enhanced_spectral_run2` | `houston_enhanced_spectral_run2` | **800** | 2000 | ±0.4 | `sigcopy:houston/spectral` |
| CoFFE SimMIM band HSI+LiDAR | Trento | 90.32 | `trento_enhanced_spectral_run2` | `trento_6way_5shot_spectral_run2` | **975** | 1500 | ±0.6 | `sigcopy:trento/spectral` |
| CoFFE SimMIM band HSI+LiDAR | MUUFL | 56.99 | `muufl_enhanced_spectral_run1` | `muufl_enhanced_spectral_run1` | **975** | 1500 | ±1.2 | `sigcopy:muufl/spectral` |
| CoFFE SimMIM token HSI | Houston | 65.50 | `houston_enhanced_spatial_no_lidar` | `houston_enhanced_spatial_no_lidar_eval` | **950** | 1500 | ±2.1 | `sig:hsi_only/houston/spatial` |
| CoFFE SimMIM token HSI | Trento | 87.60 | `trento_enhanced_spatial_no_lidar` | `trento_enhanced_spatial_no_lidar_eval` | **975** | 1500 | ±1.0 | `sig:hsi_only/trento/spatial` |
| CoFFE SimMIM token HSI | MUUFL | 57.75 | `muufl_enhanced_spatial_no_lidar` | `muufl_enhanced_spatial_no_lidar_eval` | **975** | 1500 | ±1.3 | `sig:hsi_only/muufl/spatial` |
| **CoFFE SimMIM token HSI+LiDAR** | Houston | 75.30 | `houston_enhanced_spatial_mask_test_run1_seed52` | `houston_15way_5shot_spatial_mask_test_run1_seed52` | **950** | 1500 | ±1.6 | `sigcopy:houston/spatial` |
| **CoFFE SimMIM token HSI+LiDAR** | Trento | 94.19 | `trento_enhanced_spatial_run2` | `trento_6way_5shot_spatial_run2` | **975** | 1500 | ±0.8 | `sigcopy:trento/spatial` |
| **CoFFE SimMIM token HSI+LiDAR** | MUUFL | 67.83 | `muufl_enhanced_spatial_run1` | `muufl_enhanced_spatial_run1` | **975** | 1500 | ±2.2 | `sigcopy:muufl/spatial` |
| CoFFE SimMIM band+token HSI | Houston | 58.24 | `houston_enhanced_spectral_spatial_no_lidar` | `houston_enhanced_spectral_spatial_no_lidar_eval` | **950** | 1500 | ±1.1 | `sig:hsi_only/houston/both` |
| CoFFE SimMIM band+token HSI | Trento | 86.10 | `trento_enhanced_spectral_spatial_no_lidar` | `trento_enhanced_spectral_spatial_no_lidar_eval` | **975** | 1500 | ±0.4 | `sig:hsi_only/trento/both` |
| CoFFE SimMIM band+token HSI | MUUFL | 50.76 | `muufl_enhanced_spectral_spatial_no_lidar` | `muufl_enhanced_spectral_spatial_no_lidar_eval` | **975** | 1500 | ±1.4 | `sig:hsi_only/muufl/both` |
| CoFFE SimMIM band+token HSI+LiDAR | Houston | 64.63 | `houston_enhanced_spec_spat_combined` | `houston_enhanced_spec_spat_combined_run1` | **950** | 1500 | ±1.5 | `sigcopy:houston/both` |
| CoFFE SimMIM band+token HSI+LiDAR | Trento | 92.50 | `trento_enhanced_spectral_spatial_run2` | `trento_6way_5shot_spectral_spatial_run2` | **975** | 1500 | ±1.2 | `sigcopy:trento/both` |
| CoFFE SimMIM band+token HSI+LiDAR | MUUFL | 55.57 | `muufl_enhanced_spectral_spatial_run2` | `muufl_enhanced_spectral_spatial_run2` | **975** | 1500 | ±3.2 | `sigcopy:muufl/both` |
| CoFFE MAE HSI | Houston | 69.04 **(read 69.03)** | `houston_mae_no_lidar` | `houston_mae_no_lidar_eval` | **950** | 3000 | ±1.1 | `sig:enhanced_mae/houston/hsi_only` |
| CoFFE MAE HSI | Trento | 88.43 | `trento_mae_no_lidar` | `trento_mae_no_lidar_eval` | **975** | 1500 | ±4.4 | `sig:enhanced_mae/trento/hsi_only` |
| CoFFE MAE HSI | MUUFL | 58.65 | `muufl_mae_no_lidar` | `muufl_mae_no_lidar_eval` | **975** | 1500 | ±0.4 | `sig:enhanced_mae/muufl/hsi_only` |
| CoFFE MAE HSI+LiDAR | Houston | 72.15 | `houston_mae_lidar` | `houston_mae_lidar_eval` | **950** | 3000 | ±0.7 | `sig:enhanced_mae/houston/lidar` |
| CoFFE MAE HSI+LiDAR | Trento | 91.74 | `trento_mae_lidar` | `trento_mae_lidar_eval` | **975** | 1500 | ±0.7 | `sig:enhanced_mae/trento/lidar` |
| CoFFE MAE HSI+LiDAR | MUUFL | 65.33 | `muufl_mae_lidar` | `muufl_mae_lidar_eval` | **975** | 1500 | ±1.0 | `sig:enhanced_mae/muufl/lidar` |
| MFT SimMIM token HSI+LiDAR | Houston | 67.49 | `mft_original_houston_spatial_faithful` | `mft_original_houston_spatial_faithful_eval` | **950** | 3000 | ±0.3 | `sig:mft_spatial/houston/spatial` |
| MFT SimMIM token HSI+LiDAR | Trento | 88.22 | `mft_original_trento_spatial_faithful` | `mft_original_trento_spatial_faithful_eval` | **950** | 1500 | ±0.9 | `sig:mft_spatial/trento/spatial` |
| MFT SimMIM token HSI+LiDAR | MUUFL | 59.05 | `mft_original_muufl_spatial_faithful` | `mft_original_muufl_spatial_faithful_eval` | **950** | 1500 | ±0.5 | `sig:mft_spatial/muufl/spatial` |
| MFT MAE HSI+LiDAR | Houston | 51.97 | `mft_original_houston_mae_faithful` | `mft_original_houston_mae_faithful_eval` | **950** | 3000 | ±3.7 | `sig:mft_mae/houston/mae` |
| MFT MAE HSI+LiDAR | Trento | 81.05 | `mft_original_trento_mae_faithful` | `mft_original_trento_mae_faithful_eval` | **950** | 1500 | ±4.6 | `sig:mft_mae/trento/mae` |
| MFT MAE HSI+LiDAR | MUUFL | 45.77 | `mft_original_muufl_mae_faithful` | `mft_original_muufl_mae_faithful_eval` | **950** | 1500 | ±1.8 | `sig:mft_mae/muufl/mae` |

(`sig:` = `experiments/significance_report.json`; `sigcopy:` =
`experiments/significance_report copy.json`. Every ± matched its cell's std to
2 dp with no exceptions across all 30 cells — this is not coincidence, it is
the composition rule.)

**Reproduction recipe to document:** pretrain per-scene for 1500 epochs
(AdamW, cosine, warmup 80–100), save every 25–50 epochs, then evaluate the
**epoch-950 (Houston) / epoch-975 (Trento, MUUFL)** checkpoint under the §4
protocol. For the ± column, repeat pretraining fresh to 700 epochs under seeds
`[42, 123, 456, 789, 1011]` and take the across-seed std.

### D2 — HyperSIGMA eval protocol drift — **CONFIRMED, and narrowed**

All 24 Table 3 cells located with exact mean **and** CI match. Protocol
actually used:

- **23 of 24 cells**: `k_shot=5`, `k_query=100`, `num_episodes=2000`,
  `distance_metric=euclidean` — read from each `eval_config.json`; for the
  SEM-only pad row (which has no `eval_config.json`) from its driver
  `scripts/run_native_sem_pad_experiments.sh:43-44`
  (`K_QUERY=100`, `NUM_EPISODES=2000`).
- **1 exception — Houston 11×11 spectral (21.85 ± 0.10)**: comes from
  `experiments/hypersigma_baseline_spectral_only_run1/evaluations/houston_15way_5shot_adapted_spectral_only_run1`,
  whose `eval_config.json` records **`k_query=30`** and
  `distance_metric=cosine`. The value the paper quotes is the `euclidean`
  sub-block of that dual-metric `results.json`, so the *metric* is right but
  the *query count* is 30, not 100.

**New sub-finding:** Table 3 used **2000 episodes**; Table 2 used **1000**.
`PAPER_CANON.md` §4 states "1000 per run" as law for both. Since Table 3
reports ±95 % CI over episodes, the episode count directly sets the CI width —
so this is a real protocol difference between the two tables, not a cosmetic
one.

`docs/presentation/RESULTS.md:23-25` is therefore describing an **earlier
generation** of HyperSIGMA runs: its rows report joint_sem at 600 episodes with
both cosine and euclidean (Houston joint_sem 61.59 euclidean / 52.92 cosine),
against Table 3's 67.48. Those runs are **superseded**. RESULTS.md's CoFFE rows
do carry the correct Table 2 means but with a within-run CI column
(`65.50 ± 0.19`) rather than the paper's across-seed std (`65.50 ± 2.1`).

Table 3 source runs: `hypersigma_native_ablation_run1` (12 frozen 64×64 cells),
`hypersigma_native_sem_pad_run1` (3 SEM-only cells),
`hypersigma_{houston,trento,muufl}_pca100_{spatial_only,joint_sem}_run1` (6),
`hypersigma_baseline_spectral_only_run1` + `hypersigma_{trento,muufl}_spectral_only_run1` (3).

### D3 — CPEA remnant (`lambda_factor`) — **LIVE on the paper eval path**

Call path traced end to end:

1. `lib/eval_runner.py:209` → `from scripts.evaluate_cosine import run_evaluation`.
2. `lib/eval_runner.py:33-46` `_ARCH_KEYS_FROM_MODEL` includes
   `"lambda_factor"`, so it is copied from the run's `pretrain_config.yaml`
   into the eval config. Every Table 2 CoFFE run's
   `evaluations/*/eval_config.json` accordingly records
   **`"lambda_factor": 0.5`** (and `"use_projection": false`,
   `"distance_metric": "euclidean"`, `"pool_sigma": null`).
3. `scripts/evaluate_cosine.py:176` passes it into `MFTCPEACosine(...)`.
4. The episode loop does **not** call `forward_episode`; it re-implements it
   (`:337-338` "Uses manual feature extraction (instead of
   model.forward_episode) to capture intermediate embeddings"), and at
   **`scripts/evaluate_cosine.py:392-393`**:

   ```python
   s_adapted = model.adapt_embeddings(s_patch, s_cls)
   q_adapted = model.adapt_embeddings(q_patch, q_cls)
   ```

5. `models/mft_cpea_cosine.py:294`:

   ```python
   adapted = patch_emb + lambda_factor * cls_emb.unsqueeze(1)
   ```

6. Pooling at `scripts/evaluate_cosine.py:399-400` (`pool_sigma is None` on
   every paper run, so **uniform** mean, not Gaussian).

The `renormalize` branch at `models/mft_cpea_cosine.py:296-297` is inert,
because `use_projection=False` makes `self.projection = nn.Identity()`
(`:164`), which has no `l2_normalize` attribute.

**So the paper's eval feature is not "patch tokens pooled". It is**

```
z = mean_j(patch_emb_j) + 0.5 · cls_emb
```

Because `cls_emb` is per-sample, this is not a constant offset — it genuinely
changes Euclidean distances and prototypes. λ is **KEPT** (behavior is frozen),
needs an honest name and docstring in phase 4/6, and is a paper↔code
description gap for Nikola to acknowledge.

**MFT control: λ is dead there**, correctly. `models/mft_original.py:237-245`
overrides `adapt_embeddings` to `return patch_emb` unchanged, and
`:234` packs `patch_emb = cls_emb.unsqueeze(1)` (single token), so the MFT
feature is exactly the fused CLS token. Its eval configs record
`lambda_factor: None`. That matches the paper's description of the control.

### D4 — `configs/pretrain/base.yaml` is stale — **CONFIRMED, and it is unused**

`utils/io.py:10` `load_config` is a flat `OmegaConf.load` + `to_container` with
**no inheritance mechanism**, and no config in `configs/` declares a
`defaults:`/`_base_`/`extends:` key. Repo-wide, `base.yaml` is referenced only
by `PAPER_CANON.md:187` and `SPLIT.md:44` — never by code, script, or notebook.
It is dead *and* stale (8 heads / 4 layers / λ=2.0 / 800 epochs vs the paper's
2/2 and λ=0.5). → **delete**. "Fixing it to paper defaults" would mean wiring
in an inheritance mechanism that does not exist — scope creep.

The real scene configs *are* correct on architecture: `num_layers: 2`,
`num_heads: 2`, `embed_dim: 128`, `lambda_factor: 0.5` throughout.

### D5 — `.gitignore` `lib/` trap — **already neutralised** (phase-0 refinement stands)

Negation rules at `.gitignore:78-84` re-include the five `lib/*.py`; verified
with `git check-ignore` at phase 0. Only the cosmetic removal of the shadowed
bare `lib/` rule remains for phase 5.

### D6 — Requirements kitchen-sink — **CONFIRMED, plus a missing dependency**

True third-party import set over the PAPER closure + tests + kept notebooks:

**Needed:** `torch`, `numpy`, `scipy`, `omegaconf`, `scikit-learn`,
`matplotlib`, `seaborn`, `tqdm`, `PyYAML`, `pytest`; `tensorboard` is a soft
dependency (`trainers/pretrain_trainer.py:133` imports `SummaryWriter` inside a
guarded `try` and warns on failure).

**Prunable — imported by nothing in the closure:** `hydra-core`, `h5py`,
`scikit-image`, `spectral`, `rasterio`, `timm`, `einops`, `wandb`, and
`torchvision` (its only consumer is the DEAD `data/transforms/augmentations.py`).

**Bug: `PyYAML` is imported by 9 files (`lib/*.py`, `scripts/*.py`) and is not
declared in `requirements.txt` at all.** It currently arrives transitively via
`omegaconf`. Add it in phase 6.

`setup.py`'s `install_requires` additionally lists `einops` and `timm`, which
nothing imports.

### D7 — README is wrong — **CONFIRMED**

All five claimed defects present. Additionally `setup.py` carries placeholder
metadata: `name="mft-cpea"`, `author="Nikola"`,
`author_email="nikola@example.com"`, `url=".../yourusername/mft-cpea"`.
Rewrite in phase 8 against `PAPER_CANON.md` §6.

### D8 — Duplicate/junk files — **CONFIRMED with one important correction**

Count is **9 notebooks**, not 7: `pretrain copy{,  2, 3}.ipynb` (3),
`evaluate_hypersigma copy{, 2, 3}.ipynb` (3),
`evaluate_hypersigma_pca100 copy{, 3}.ipynb` (2), plus `pretrain_run2.ipynb`.
None is byte-identical to its base — they are divergent working copies that
differ in their parameter cells. None reaches a module the PAPER closure does
not already reach.

`scripts/adapt_hypersigma_houston.py` is an explicit backward-compatibility
shim (`from scripts.adapt_hypersigma import main, run_adapt`). Its stated
purpose — "notebooks keep working" — no longer holds: the only mentions
repo-wide are markdown prose and captured log output inside notebooks, plus a
**stale docstring at `lib/adapt_runner.py:1`** which says it wraps
`adapt_hypersigma_houston` while `:86` imports `adapt_hypersigma`. → DEAD.

> **`experiments/significance_report copy.json` IS NOT JUNK — DO NOT DELETE.**
> It is the only source of the across-seed std for the 12 `enhanced`
> (HSI+LiDAR) Table 2 cells, including all three headline
> **CoFFE SimMIM token HSI+LiDAR** numbers. `significance_report.json`'s
> `groups` list is `["hsi_only", "enhanced_mae", "mft_mae", "mft_spatial"]` —
> the `enhanced` group is absent from it (its config docstring calls that group
> "v1, done"). The two files also use different cell-key schemas
> (`houston/spatial` vs `hsi_only/houston/spatial`).
> Manifest verdict: **rename** →
> `experiments/significance_report_enhanced_v1.json`.

### D9 — Exploratory pipelines — **CONFIRMED, no library fallout**

15 scripts + 3 report JSONs, reachable only from the ablation/combo/bestcfg
roots, appearing in no paper table. Their closure minus the PAPER closure is
empty, so deleting them orphans nothing. Default recommendation: delete
(history retains them). **Nikola's call at the gate.**

### D10 — Legacy pretrain stack overlap — **PARTLY REFUTED** (see §2.2, §2.4)

| Seeded candidate | Verdict |
|---|---|
| `models/backbones/*` (4 files + `__init__`) | **DEAD** — closed island |
| `models/wrappers/pretrain_wrapper.py` (+ `__init__`) | **DEAD** — closed island |
| `pretrain/masked_modeling.py` | **PAPER** — live, multiple consumers |
| `pretrain/mae_pretrain.py` | **PAPER** — live |
| `pretrain/decoders.py` | **PAPER** — live file, 3 dead symbols |
| `trainers/pretrain_trainer.py` | **PAPER** — live |
| `data/transforms/augmentations.py` | **DEAD** — no importer |

### D11 — `SPLIT.md` — **CONFIRMED internal**

102 lines describing how to split this tree out of the private parent repo
`mft-cpea`, naming the internal branch
`claude/refactor-pretraining-pipelines-CXanw` (`SPLIT.md:3`) and a private
clone URL. → delete for the public release. **Nikola's call.**

### D12 — Metrics surface — **REFUTED as stated; refined**

`utils/metrics.py` does **not** compute AA or κ. It contains exactly two
functions: `accuracy` (:7) and `confusion_matrix` (:12, a thin `sklearn`
wrapper). AA and κ are computed in **`scripts/evaluate_cosine.py`**:
`compute_kappa` (:277) and `compute_metrics_from_confusion_matrix` (:293-322).

The code already documents the paper's own justification, at
`scripts/evaluate_cosine.py:514`:

```
# NOTE: With balanced episodes (same k_query per class), OA and AA are ...
```

So D12's remedy is unchanged (README leads with OA, states why κ is omitted),
but it applies to `scripts/evaluate_cosine.py`, not `utils/metrics.py`.

---

## 4. Surprises found this phase (reported, not fixed)

### D13 (new) — the committed configs do not reproduce the paper runs

The reproducible recipe lives in the frozen `experiments/*/pretrain_config.yaml`,
**not** in `configs/`. Concrete divergences:

| Committed config | Says | Paper run actually used |
|---|---|---|
| `configs/pretrain/trento_pretrain_enhanced.yaml` | `band_mask_ratio: 0.9` | **0.85** (`experiments/trento_enhanced_spectral_run2/pretrain_config.yaml`) |
| `configs/pretrain/houston_pretrain_enhanced.yaml` | `epochs: 3000` | **1500** (`experiments/houston_enhanced_spatial_mask_test_run1_seed52/...`) |
| `configs/pretrain/hypersigma_houston_adapt_pca100.yaml:52-56` | 3000 epochs / batch 64 / lr 1.5e-4 | **2000 / 128 / 1e-5** (`experiments/hypersigma_adapt_houston_pca100_spatial_only_run1/pretrain_config.yaml:24-27`) |

The paper's stated mask rates — band `(0.85, 0)`, token `(0, 0.75)` — are
**exactly** what the runs used, confirmed across all six canonical run configs.
So the paper is right and `configs/` is stale. Phase 5/8 must regenerate
`configs/` from the frozen run configs, not the reverse. (The repo's own
`hypersigma_native_sem_pca100_report.json` `coverage_gaps_and_anomalies` list
independently flags the HyperSIGMA half of this.)

### D14 (new) — the headline Houston number comes from a run the repo classifies as scratch

Table 2's **CoFFE SimMIM token HSI+LiDAR / Houston = 75.30** — the paper's
single most important number — comes from
`experiments/houston_enhanced_spatial_mask_test_run1_seed52`.

`scripts/compile_results.py:35` defines
`_TEST_MARKERS = ("test_run", "spatial_mask_test", "_example")` and `:61-62`
drops any experiment matching them as a `"scratch/test run"`. That filter
excludes exactly this run, which is why `docs/presentation/RESULTS.json` has
**no** Houston "Enhanced: spatial / HSI+LiDAR" entry and why `75.3` does not
appear anywhere under `datasets.houston` in it.

Also: the directory name says `seed52`, but the run used **seed 42** —
`pretrain_metadata.json` records `overrides.hardware.seed: 42` and a
description reading "…on seed 42. First run." The `seed52` suffix is a
misnomer. Nothing is wrong with the *number*; the naming and the compiler
filter simply disagree with the paper's selection.

### D15 (new) — `utils/checkpoints.py` is dead but the canon points at it

`PAPER_CANON.md` §7.2 names `utils/checkpoints.py` "(or its successor)" as
where a state_dict key-mapping shim must live. That file is currently dead;
the live key-fixing logic is `scripts/evaluate_cosine.py:107-...`. Any future
key-map shim must go where the live code is (or `checkpoints.py` must be
revived deliberately). See risk R2.

### D16 (new) — `model_type` is written into results.json and compared by readers

`scripts/evaluate_cosine.py:823` writes
`"model_type": "MFTOriginalCosine" | "MFTCPEACosine"` into eval results.
`scripts/compile_results.py:58,104` and `scripts/build_experiment_metadata.py:112-115`
read it back — but they only compare against `"HyperSIGMADual"` /
`"HyperSIGMA" in m`, both of which `PAPER_CANON` §1 preserves. So renaming the
CoFFE/MFT values is reader-safe, but it *does* change the content of
newly-written `results.json`. Phase 4 should add the alias map to readers
before changing the writer.

---

## 5. Risk register

| # | Risk | Bites in | Mitigation |
|---|---|---|---|
| R1 | `configs/` does not reproduce the paper (D13). Anyone using it gets different numbers. | phase 5, 8 | regenerate configs from `experiments/*/pretrain_config.yaml`; never the reverse |
| R2 | `PAPER_CANON` §7.2 points the key-map shim at the dead `utils/checkpoints.py` (D15). Deleting it before phase 4 removes the canon's named landing spot. | phase 3, 4 | either keep the file as the deliberate shim home, or amend §7.2 to name `evaluate_cosine.py`'s successor. **Gate question.** |
| R3 | `experiments/significance_report copy.json` looks like D8 junk but is load-bearing for 12 Table 2 cells. | phase 3 | manifest verdict is `rename`, not `delete`; called out at the gate |
| R4 | Table 2's ± and means come from different experiments at different epochs (D1). A reproduction script that reruns "the paper" will match neither column exactly. | phase 8 | document the two-source composition explicitly |
| R5 | `pretrain/__init__.py` and `models/__init__.py` re-export names nothing uses. Deleting island files requires editing those `__init__` blocks in the same commit, or imports break. | phase 3 | treat file + re-export line as one atomic change; equivalence harness must import every entry point |
| R6 | 3 dead symbols inside the live `pretrain/decoders.py`; 1 dead symbol in `models/components/projection.py` (`ModalitySpecificProjection`), `transformer.py` (`TransformerLayer`), `tokenizers.py` (`SpatialAuxTokenizer`), `mft_blocks.py` (`StandardSelfAttention`, `MFTBlock`), `data/samplers/patched_episode_sampler.py` (`CrossDatasetEpisodeSampler`), `utils/io.py` (`save_results`), `utils/visualization.py` (`plot_embeddings`), `lib/eval_runner.py` (`load_pretrain_config`, `find_checkpoint`), `trainers/pretrain_trainer.py` (`create_pretrain_dataloaders`). | phase 3/6 | symbol-level prune is a *proposal* only; not in the manifest as file deletions |
| R7 | Table 3 used 2000 episodes, Table 2 used 1000, while `PAPER_CANON` §4 states 1000 as law (D2). | phase 8 | amend §4 to state the per-table episode count |
| R8 | One Table 3 cell used `k_query=30` (D2). Reproduction docs must not claim a uniform protocol. | phase 8 | footnote that cell |
| R9 | `pyproject.toml` `addopts` forces `--cov=models --cov=data` on every pytest run; no `gpu`/`data` markers registered (phase-0 E3). | phase 2 | register markers; make coverage opt-in |

## 6. DO-NOT-RENAME — string literals naming on-disk artifacts

These read or construct names of frozen artifacts on Nikola's machines. Their
*values* are frozen (`PAPER_CANON` §7.3); the variables holding them may be
renamed.

| Location | Literal(s) |
|---|---|
| `scripts/sig_significance_config.py:66-74` | `_ENHANCED_CANONICAL` — the 9 canonical run dir names (`houston_enhanced_spatial_run1`, `houston_enhanced_spectral_run2`, `houston_enhanced_spec_spat_combined`, `trento_enhanced_spat_run1`, `trento_enhanced_spectral_run1`, `trento_enhanced_spectral_spatial_run1`, `muufl_enhanced_spatial_run1`, `muufl_enhanced_spectral_run1`, `muufl_enhanced_spectral_spatial_run2`) |
| `scripts/sig_significance_config.py:78` | `_NO_LIDAR_SUFFIX` = `{"spatial": "spatial", "spectral": "spectral", "both": "spectral_spatial"}` |
| `scripts/sig_significance_config.py:111-154` | group ids used as dir-name components: `"enhanced"`, `"hsi_only"`, `"enhanced_mae"`, `"mft_mae"`, `"mft_spatial"`; variant ids `"spatial"`, `"spectral"`, `"both"` |
| `scripts/sig_significance_config.py:52` | `EVAL_NAME = "sig_eval_epoch700"` — the eval subdir name on disk |
| `scripts/pretrain_enhanced.py:297,302,316,318,376` | config-key **values** read from frozen `pretrain_config.yaml`: `"mft_cpea"`, `"enhanced"`, `"mae"`, `"mft_original"` |
| `scripts/evaluate_cosine.py:149,766,885` | same: `"mft_cpea"` default and config dict value |
| `scripts/build_experiment_metadata.py:107` | `if mname == "mft_cpea"` |
| `scripts/compile_results.py:35` | `_TEST_MARKERS = ("test_run", "spatial_mask_test", "_example")` — substring match on frozen dir names (see D14) |
| `scripts/compile_results.py:72,85-86` | `"spectral" in exp_l`, `"spatial" in exp_l`, `"_spat" in exp_l` — regime inferred from frozen dir names |
| `scripts/compile_results.py:58,104`, `scripts/build_experiment_metadata.py:112-115`, `scripts/evaluate_hypersigma_cosine.py:548` | `model_type` value `"HyperSIGMADual"` (canon §1 preserves this name) |
| `scripts/compile_mft_faithful_results.py:34` | `VARIANTS = ["spatial", "mae"]`; selector `"mft_original_{dataset}_{variant}_faithful"` |
| `scripts/gather_requested_results.py:106-206` | ~30 hard-coded `load_eval("<exp dir>", "<eval subdir>")` pairs |
| `scripts/bestcfg_config.py:36`, `scripts/run_hsi_only_experiments.py:56-57`, `scripts/gather_requested_results.py:164` | variant id literals `"spatial"`, `"spectral"`, `"both"` |
| `configs/**/*.yaml` `paths:` blocks (81 occurrences of `checkpoint_dir`/`log_dir`/`data_root`) | e.g. `configs/pretrain/houston_pretrain_enhanced.yaml:85` `"./checkpoints/pretrained/houston_enhanced_CenterWeightedLoss_2layers"`; `configs/pretrain/hypersigma_houston_adapt_pca100.yaml:48` `"checkpoints/hypersigma_adapted/houston_pca100"` |
| `models/mft_cpea_cosine.py`, `models/mft_original.py`, `models/hypersigma/*`, `pretrain/*` | every `nn.Module` **attribute** name — these are state_dict keys (`PAPER_CANON` §7.2) |

### Rename-eligible (user-facing labels only, canon §1)

| Location | Current | Canonical |
|---|---|---|
| `scripts/evaluate_cosine.py:789` | `"original-MFT"` / `"MFT-CPEA-Cosine"` | `"MFT (original)"` / `"CoFFE"` |
| `scripts/evaluate_cosine.py:823` | `model_type` `"MFTOriginalCosine"` / `"MFTCPEACosine"` | `"MFTOriginal"` / `"CoFFE"` — **writer change, see D16** |
| `README.md` title | "MFT-CPEA: Enhanced Pretraining + Cosine Few-Shot Evaluation" | paper title |

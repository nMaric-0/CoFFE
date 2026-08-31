# PAPER_CANON.md — Source of truth for the CoFFE repository

This file distills the camera-ready paper:

> **"A Compact In-Domain Fusion Encoder versus a Hyperspectral Foundation Model
> for Few-Shot HSI-LiDAR Land-Cover Classification"**, N. Marić and D. Kocev,
> Jožef Stefan Institute / IPS Ljubljana.

The repository is the public code release for this paper. **Where the code and this
file disagree on names, terminology, or protocol description, this file wins.
Where they disagree on *behavior* (numbers, semantics), STOP — that is a
discrepancy to report to Nikola, never something to silently "fix" in either
direction.** The paper is frozen; the code must be brought to match its
vocabulary without changing what it computes.

---

## 1. Canonical naming law (old → new)

| Legacy name in repo | Canonical name | Notes |
|---|---|---|
| `MFT-CPEA`, `MFTCPEA`, `mft_cpea`, "MFT-CPEA-Cosine" | **CoFFE** (Cross-modal Fusion Feature Encoder) | The paper's compact encoder. Python class `CoFFE`, module `coffe/models/coffe.py`, config `model.name: "coffe"`. |
| `MFTCPEACosine` (class) | `CoFFE` | Class rename only. **state_dict attribute names are frozen** (see §7). |
| `models/mft_cpea_cosine.py` | `coffe/models/coffe.py` | |
| "Cosine" in file/class/script/doc names (`evaluate_cosine.py`, `run_cosine_eval.sh`, `hypersigma_cosine.py`, `docs/COSINE_VARIANT.md`, `HyperSIGMACosine`) | Drop. Protocol name is **"Euclidean nearest-class-mean (NCM)"** | The paper protocol is Euclidean NCM (`distance_metric="euclidean"` in every paper run). `cosine` may remain as a **non-default option value** of `distance_metric`, never in a name, title, or default. |
| `MFTOriginalCosine` | `MFTOriginal` | The architectural control: original MFT (Roy et al.), external fusion token, pretrained under the same masked objectives. Config `model.name: "mft_original"` is already correct. |
| objective `"enhanced"` | `"simmim"` | SimMIM-style in-place masked reconstruction (Xie et al.). |
| variant `"spectral"` | **SimMIM band** → id `simmim_band` | Band masking of (pixel, band) entries, rates `(r_b, r_s) = (0.85, 0)`. |
| variant `"spatial"` | **SimMIM token** → id `simmim_token` | Spatial-token masking of whole pixel tokens, `(0, 0.75)`. This is the headline configuration. |
| variant `"both"` | **SimMIM band+token** → id `simmim_band_token` | `(0.85, 0.75)`. |
| `"mae"` / `enhanced_mae` | **MAE** (drop-style baseline, He et al.) → id `mae` | |
| `hsi_only` (group) | Modality axis: `hsi` vs `hsi_lidar` | Modality is an input regime (`use_aux`), orthogonal to the objective. |
| `EnhancedMaskedSpectralSpatialModel` | `SimMIMPretrainModel` (or similar under `coffe/pretrain/simmim.py`) | |
| `HyperSIGMACosine` | `HyperSIGMAFewShot` | Frozen-feature NCM evaluator over `HyperSIGMADual`. `HyperSIGMADual` keeps its name. |
| HyperSIGMA input regimes | `backbone_native` (64×64, `pad` or `upscale`) and `patch_native` (11×11) | Paper §3. |
| HyperSIGMA adaptations | `frozen`, `sem_only`, `spatial`, `spectral`, `joint_sem` | Label-free continued masked reconstruction from released checkpoints (75% token masking, per-patch z-scored targets). |
| README title "MFT-CPEA: Enhanced Pretraining + Cosine Few-Shot Evaluation" | Paper title (or "CoFFE — ..." short form) | |
| User-facing labels like `"MFT-CPEA-Cosine"`, `"original-MFT"` in scripts | `"CoFFE"`, `"MFT (original)"` | e.g. `scripts/evaluate_cosine.py:789,823`. |

**Terminology in prose/docstrings:** "prototypical network" → only for the cited
Snell et al. method; this repo's evaluator is a **nearest-class-mean classifier
on fixed features** (Mensink et al.). "5-way" is wrong everywhere it appears:
episodes are **N-way** where N = the scene's full class count.

## 2. CoFFE architecture constants (paper §3, Fig. 1)

- Input: `P×P = 11×11` patch, `C_h` HSI bands + `C_a` LiDAR channels **concatenated at the input** and jointly embedded (input-level fusion — the paper's key architectural claim vs MFT's external token).
- Tokens: `P² = 121` pixel tokens + 1 prepended **class-agnostic token**; learnable positional embeddings.
- Encoder: pre-LayerNorm transformer, `D=128`, `2 heads`, `2 layers`.
- Eval feature: patch tokens pooled (uniform or **Gaussian center-weighted**) → `z ∈ R^128`.
- Pretraining-only parts (discarded at eval): 2-layer MLP decoder (hidden 256), projection head.
- Parameter count (Houston eval encoder): **579,328** (marginally fewer on Trento/MUUFL). Keep a test asserting this within ±1%.

## 3. Pretraining objectives (paper §3, Eq. 1)

- SimMIM-style **in-place** masked reconstruction over the token space; two composable masks: band `(pixel, band)` entries and spatial whole-token masks.
- Loss: masked MSE over the **union** of the two masks; optional Gaussian center weight `w_j` with **mean one** up-weighting the classified pixel.
- MAE (token-drop style) is the single-objective baseline in the same token space.
- Per-scene pretraining, no labels; AdamW, cosine schedule; one NVIDIA RTX 4090.
- Paper timing: CoFFE pretraining (1500 epochs) 0.7–4.3 h/scene; HyperSIGMA adaptation (2000 epochs) ≈ 36 h.

## 4. Evaluation protocol (paper §3–4) — these constants are law

| Constant | Value |
|---|---|
| Episode type | N-way, **N = scene's class count** (Houston 15 / Trento 6 / MUUFL 11) |
| Support | K = 5 per class |
| Queries | 100 per class, class-balanced |
| Episodes | 1000 per run |
| Classifier | Euclidean nearest-class-mean; prototype = mean of the 5 support features |
| Encoder | Frozen; projection head **off** at eval (`use_projection: False`) |
| Metric | OA only (class-balanced queries ⇒ OA = AA; κ omitted by design) |
| Reporting | OA mean ± 95% CI over episodes (within a training); **across-seed std** for CoFFE/MFT over 5 seeds |
| Seeds | `[42, 123, 456, 789, 1011]` (from `scripts/sig_significance_config.py`) |

## 5. Datasets (paper Table 1)

| Dataset | C_h | C_a | N | Image (px) | GSD (m) | Labelled px |
|---|---|---|---|---|---|---|
| Houston 2013 | 144 | 1 | 15 | 349×1905 | 2.5 | 15,029 |
| Trento | 63 | 1 | 6 | 600×166 | 1.0 | 30,414 |
| MUUFL | 64 | 2 | 11 | 325×220 | 0.54/1.0 | 53,687 |

Preprocessing: 11×11 patches centred on each labelled pixel, border-padded at
edges; every HSI band and LiDAR raster min-max normalised to [0,1]
independently. HyperSIGMA spatial branch input: PCA 144→100 on Houston; linear
band resampling 63/64→100 on Trento/MUUFL; spectral branch receives raw bands.

## 6. Headline results (paper Tables 2–3) — for README + report-mapping only

Never regenerate or "correct" these from code. They anchor the README rewrite
and the audit's mapping of `experiments/*.json` → paper cells.

**Table 2** — OA % ± across-seed std (CoFFE/MFT); HyperSIGMA single-run:

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

(superscripts: u upscale, p pad, j joint+SEM, s spatial)

**Table 3** — HyperSIGMA sweep, OA mean ± 95% CI (features: sp.=spatial 768-d,
sc.=spectral 768-d, fu.=fused SEM 512-d):

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

Other numbers for the README: CoFFE has >2 orders of magnitude fewer parameters
than HyperSIGMA (~579K vs ~180M with both ViT-Base bodies + SEM; label-free
adaptation trains 1.13M–8.45M). Headline margins over the best FM config:
+7.8 OA Houston, +13.0 MUUFL, +3.1 Trento (within spread).

## 7. Hard invariants (apply to every phase)

1. **Numerics are frozen.** No refactor step may change any computed number:
   loss values, features, prototypes, OA. The equivalence harness
   (`tests/equivalence/`) is the arbiter. If a change is behavior-altering and
   seems *desirable*, stop and report — do not apply.
2. **state_dict key contract.** `nn.Module` **attribute names** in `CoFFE`,
   `MFTOriginal`, the HyperSIGMA wrappers, and every pretrain model are frozen,
   because they define checkpoint keys. Renaming classes, files, and configs is
   free; renaming attributes requires a key-mapping shim in
   `utils/checkpoints.py` (or its successor) **plus** a loading test against a
   pre-refactor checkpoint fixture.
3. **Frozen artifacts stay readable.** Existing experiment trees on Nikola's
   machines contain `pretrain_config.yaml` with `model.name: "mft_cpea"`,
   objective `"enhanced"`, variant `"spatial"/"spectral"/"both"`, and directory
   names like `houston_enhanced_spatial_run1`. All readers (eval runner,
   aggregators, compile scripts) must accept legacy vocabulary via an explicit
   alias map (emit `DeprecationWarning`). **Writers emit canonical vocabulary
   only.** String literals that name on-disk artifacts (e.g.
   `_ENHANCED_CANONICAL` in `sig_significance_config.py`, checkpoint dir names
   in configs' `paths:`) are on the DO-NOT-RENAME list.
4. **`third_party/HyperSIGMA/` is vendored.** Never restyle, rename, or "fix"
   it; only import paths referencing it may change. `LICENSE` and `NOTICE`
   there are preserved and must be mentioned in the README.
5. **Data and large checkpoints are not in git.** Every test must pass with no
   datasets and no HyperSIGMA checkpoints present (use synthetic tensors sized
   per §5; gate real-data tests with skip markers).
6. **No silent scope creep.** Deletions come only from the approved audit
   manifest; renames only from §1; anything else is proposed, not done.

## 8. Known discrepancies (pre-seeded; Phase 1 resolves each with evidence)

These were found by an initial paper↔repo comparison. The audit must confirm,
refute, or refine each, citing file:line and/or `experiments/*.json` metadata.
None may be "fixed" by changing behavior — resolution means *documenting* and,
where the paper allows, renaming.

- **D1 — Epoch count.** Paper §4 says CoFFE pretraining is 1500 epochs;
  `sig_significance_config.py` says `EPOCHS = 700` (its docstring: "fresh to
  700 epochs"), while `scripts/rerun_mft_faithful_eval_ep1500.sh` implies
  epoch-1500 evals. Determine which epoch produced each Table 2 cell from run
  metadata and record it. Reproduction docs must state the true recipe.
- **D2 — HyperSIGMA eval protocol drift.** `docs/presentation/RESULTS.md`
  describes HyperSIGMA evals at `k_query=30` (Houston), 600–2000 episodes, with
  both cosine and euclidean — but paper Table 3 states one protocol (5-shot
  N-way, Euclidean, ±95% CI). Identify which report JSON matches Table 3 cells
  exactly and mark superseded ones.
- **D3 — CPEA remnant (`lambda_factor`).** `models/mft_cpea_cosine.py`
  implements class-aware adaptation (`adapted = patch_emb + λ·cls_emb`,
  ~line 294) with per-config λ (Houston yaml: 0.5). The paper describes pooling
  patch tokens only. Trace the exact eval path used by
  `lib/eval_runner.py` for the paper runs: does this adaptation execute and
  affect features? If **live**, it stays, gets an honest name and a docstring,
  and is flagged to Nikola as a paper/code description gap. If **dead** on the
  paper path, it is prune-eligible (still requires harness proof).
- **D4 — `configs/pretrain/base.yaml` is stale**: 4 layers / 8 heads /
  λ=2.0 / 800 epochs vs the paper's 2/2 encoder. Decide whether base.yaml is a
  real base (then fix to paper defaults) or unused (then prune).
- **D5 — `.gitignore` contains `lib/`** (Python packaging boilerplate) while
  `lib/` holds first-party source; new files there are silently untracked.
  Restructure must eliminate this trap.
- **D6 — Requirements kitchen-sink.** `requirements.txt` lists hydra-core,
  wandb, timm, rasterio, spectral, tensorboard, seaborn, etc. Compute the true
  import set; prune in Phase 6.
- **D7 — README is wrong** on: title/branding, "cosine ... prototypical
  network" protocol, the `n_way: 5` example (paper is N-way full-class), OA/AA/
  Kappa metrics, and it omits the MFT control and HyperSIGMA routes entirely.
- **D8 — Duplicate/junk files**: `notebooks/* copy*.ipynb` (7 files),
  `notebooks/pretrain_run2.ipynb`, `experiments/significance_report copy.json`,
  `scripts/adapt_hypersigma_houston.py` vs `scripts/adapt_hypersigma.py`.
- **D9 — Exploratory pipelines** (`ablation_*`, `combo_*`, `bestcfg_*` scripts
  + their report JSONs) appear in no paper table. Default recommendation:
  remove from the release (they remain in git history); Nikola may instead
  choose `archive/`. His call at the Phase 1 gate.
- **D10 — Legacy pretrain stack overlap.** `pretrain/masked_modeling.py` vs
  `pretrain/masked_modeling_enhanced.py`; `pretrain/mae_pretrain.py`;
  `pretrain/decoders.py` (Transformer decoders vs the paper's 2-layer MLP);
  `trainers/pretrain_trainer.py` vs `lib/pretrain_runner.py`;
  `models/backbones/*` (3 legacy backbones); `models/wrappers/pretrain_wrapper.py`
  (`ContrastiveHead` — no contrastive learning in the paper). Trace which are
  in the paper-run import closure.
- **D11 — `SPLIT.md`** documents extraction from a private parent repo —
  internal; default: remove from the public release.
- **D12 — Metrics surface.** `utils/metrics.py` computes AA/κ; paper reports OA
  only (OA=AA by construction). Keeping AA/κ in results JSONs is fine; README
  and docs must lead with OA and state why κ is omitted.

## 9. Target vocabulary for new artifacts

New configs: `configs/{coffe,mft,hypersigma}/<scene>_<regime>[_hsi].yaml`, e.g.
`configs/coffe/houston_simmim_token.yaml`, `configs/coffe/houston_mae_hsi.yaml`,
`configs/mft/trento_simmim_token.yaml`,
`configs/hypersigma/muufl_patchnative_joint_sem.yaml`.
New experiment names: `<scene>_<model>_<regime>_<modality>_seed<k>`.
CLI verbs: `pretrain`, `evaluate`, `adapt-hypersigma`, `reproduce`.

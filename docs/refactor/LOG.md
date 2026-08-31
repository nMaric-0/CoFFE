# Refactor log

One entry per phase. Append only — never rewrite a past entry.

---

## Phase 0 — record pre-refactor baseline

- **Date:** 2026-08-31
- **Baseline commit:** `9541085` "Final experiments" (2026-07-30), on `main`
- **Tag:** `pre-refactor` → `9541085`
- **Branch:** `refactor/cleanup` created from `main`

### What changed

No code, config, or notebook was touched. Additions only:

| Path | What |
|---|---|
| `docs/refactor/ENV.md` | environment record |
| `docs/refactor/packages.txt` | 205 installed distributions |
| `docs/refactor/BASELINE.md` | test summary, inventory summary, known-junk snapshot, phase-0 discrepancies |
| `docs/refactor/baseline_pytest.txt` | verbatim baseline pytest output |
| `docs/refactor/inventory.json` | machine-readable repo inventory |
| `docs/refactor/LOG.md` | this file |
| `tools/refactor/inventory.py` | the inventory tool (stdlib + `ast`) |

Also swept into this commit by Nikola's decision (see BASELINE.md §5): the
pre-existing uncommitted `.gitignore` archive-symlink rules and the three
untracked governance docs `CLAUDE.md`, `PAPER_CANON.md`, `WORKFLOW.md`.

### Verification results

`pytest -q --tb=short` → **62 passed, 0 failed, 0 skipped**, 2 benign
third-party warnings, 86.79 s. Coverage plugins present, so pyproject `addopts`
ran as configured (`-p no:cov` not needed). Baseline coverage of
`models` + `data`: 50 % of 2168 statements.

Environment: repo-local **uv** venv (not conda), Python 3.12.3,
torch 2.11.0+cu128, numpy 2.4.4, scipy 1.17.1, sklearn 1.8.0.
`torch.cuda.is_available()` → True (4 × RTX 4090); refactor verifies on CPU.

Inventory: 108 Python files / 19,889 LOC (excluding `third_party/`, `.venv`,
caches, `experiments/*/`), 260 top-level defs, 82 top-level classes, no parse
errors; 88 non-Python files. `scripts/` alone is 39 files / 8,053 LOC.

### Open questions for phase 1

1. **E1** — `CLAUDE.md` "Environment facts" are wrong (conda `coffe` does not
   exist; the env is a uv `.venv`; there are 4 GPUs, not 1). Correct at a gate.
2. **E2** — the baseline tree was dirty; resolved per BASELINE.md §5. Diffs
   against `pre-refactor` will include the `.gitignore` change and the three
   governance docs.
3. **E3** — no `gpu` / `data` pytest markers exist yet; phase 2 must register
   them in `pyproject.toml`.
4. **E4** — `.claude/` is gitignored, so the refactor's own phase skills are
   unversioned. Nikola's call whether the release carries them.
5. **D5 refinement** — the `.gitignore` `lib/` trap is **already neutralised**
   at baseline by negation rules (`.gitignore:78–84`), verified with
   `git check-ignore`; all five `lib/*.py` files are tracked. Only the cosmetic
   removal of the shadowed bare `lib/` rule remains, for phase 5.
6. **D8 refinement** — the duplicate-notebook count is **8** `* copy*` files
   plus `pretrain_run2.ipynb`, not the 7 stated in PAPER_CANON §8 D8.
7. Low coverage areas to weigh in phase 1's import-closure trace (low coverage
   is a hint, not proof): `data/transforms/augmentations.py` 0 %,
   `data/samplers/patched_episode_sampler.py` 13 %, `models/backbones/*`
   16–17 %, `models/wrappers/pretrain_wrapper.py` 22 %.

---

## Phase 1 — audit (closure, D1–D12 verdicts, manifest)

**Changed:** only `docs/refactor/` and `tools/refactor/`. No behavior touched.

- Added `tools/refactor/closure.py` — static import-closure tracer over named
  root sets; parses `.ipynb` code cells; follows ancestor package `__init__.py`
  edges. Deterministic; re-runnable.
- Added `tools/refactor/build_manifest.py` — encodes the audit verdicts into
  `docs/refactor/manifest.json` so it can be regenerated and diffed.
- Added `docs/refactor/closure.json` (raw closure data),
  `docs/refactor/manifest.json` (213 records), `docs/refactor/AUDIT.md`
  (narrative + D1–D16 verdicts + risk register + DO-NOT-RENAME list).

**Verification:** `pytest -q` → **62 passed** (unchanged from the phase-0
baseline). Both new tools byte-deterministic across reruns. `gpu`/`data`
markers still unregistered (phase-0 E3 stands; phase 2 fixes it).

**Closure result:** 126 traceable files; PAPER union closure = 79.
Classification: PAPER 124, INFRA 36, EXPLORATORY 19, VENDORED 13, DEAD 11,
DUPLICATE 10. Verdicts: 158 keep, 42 delete, 13 rename.

**Load-bearing negative result:** `exploratory − PAPER = ∅` for library code.
The ablation/combo/bestcfg scripts and every notebook reach no module the paper
routes do not already reach, so the D9 decision cannot orphan library code.

**Headline findings**

1. **Table 2 is a two-source composite.** Means come from single canonical runs
   evaluated at **epoch 950 (Houston) / 975 (Trento, MUUFL)**; the ± column is
   the across-seed std from a *separate* 5-seed experiment trained fresh to
   **700** epochs. All 30 cells matched exactly, both columns. Nothing in the
   repo states this. (D1)
2. **D3 λ is LIVE.** The paper eval feature is
   `z = mean_j(patch_emb_j) + 0.5·cls_emb`, not "patch tokens pooled" —
   `scripts/evaluate_cosine.py:392-393` → `models/mft_cpea_cosine.py:294`, with
   `lambda_factor: 0.5` in every Table 2 CoFFE eval config. Behavior stays;
   this is a paper↔code description gap.
3. **D8 correction — `experiments/significance_report copy.json` is NOT junk.**
   It is the only source of the ± for the 12 `enhanced` (HSI+LiDAR) Table 2
   cells, including all three headline numbers. Verdict changed to `rename`.
4. **D10 partly refuted.** `pretrain/masked_modeling.py`, `mae_pretrain.py`,
   `decoders.py` and `trainers/pretrain_trainer.py` are all LIVE.
   `models/backbones/*` + `models/wrappers/*` are DEAD, but only provably so
   via symbol-level analysis — a package `__init__` re-export makes them
   import-reachable.
5. **D12 refuted as stated.** `utils/metrics.py` holds only `accuracy` and
   `confusion_matrix`; AA/κ live in `scripts/evaluate_cosine.py:277-322`.
6. **Paper §2 invariant verified exactly:** Houston eval encoder =
   **579,328** parameters (delta 0).

**New surprises (reported, not fixed)**

- **D13** — the committed `configs/` do **not** reproduce the paper runs
  (Trento band rate 0.9 vs the 0.85 that ran; Houston 3000 vs 1500 epochs;
  HyperSIGMA PCA-100 3000/64/1.5e-4 vs the 2000/128/1e-5 that ran). The paper's
  own stated mask rates are correct — `configs/` is the stale artifact.
- **D14** — Table 2's headline Houston 75.30 comes from
  `houston_enhanced_spatial_mask_test_run1_seed52`, which
  `scripts/compile_results.py:35` deliberately filters out as a scratch run
  (hence its absence from `RESULTS.json`). The `seed52` in the name is a
  misnomer; the run used seed 42.
- **D15** — `PAPER_CANON` §7.2 points the state_dict key-map shim at
  `utils/checkpoints.py`, which is dead; the live logic is in
  `scripts/evaluate_cosine.py:107`.
- **D16** — `model_type` is written into `results.json` and read back by
  compilers; renaming its CoFFE/MFT values is a writer change needing reader
  aliases first.
- **D2 sub-finding** — Table 3 used **2000** episodes, Table 2 used **1000**,
  while `PAPER_CANON` §4 states 1000 as law for both; and one Table 3 cell
  (Houston 11×11 spectral) used `k_query=30`.

**Open questions for the gate:** see the phase-1 gate checklist — D1 recipe
wording, D3 acknowledgement, D9 disposition, D11, notebook keep-list, the
`significance_report copy.json` rescue, and R2 (`utils/checkpoints.py` vs
canon §7.2).

### Phase 1 gate — Nikola's decisions (2026-08-31)

| Item | Decision | Status |
|---|---|---|
| D1 epoch recipe | correct as audited | recorded in canon as **D17** |
| D3 λ removal | "does not contribute — remove and archive" | **ON HOLD — measured to change numbers** (see below) |
| D9 exploratory | archive into an untracked tree | 18 records → `verdict: archive`, `target: archive/exploratory/…`, `approved: true` |
| D11 `SPLIT.md` | delete | `approved: true` |
| D13–D16 | promote to PAPER_CANON | added to §8, **plus D17 (Table 2 composition) and D18 (episode counts)** |

**D3 is the one item I did not execute.** The stated premise was that λ "does
not contribute". I measured it instead of assuming, with
`tools/refactor/lambda_probe.py` (new): the real Houston headline checkpoint
(epoch 950), the exact `evaluate_cosine.py` episode path, CPU, 40 episodes,
λ=0.5 vs λ=0:

- OA **74.85 → 73.15**, i.e. **−1.70 pp**
- **5.4 %** of all query predictions flip (agreement 94.57 %)
- **0 of 40** episodes unchanged
- mechanism: `mean ‖cls − mean(cls)‖ = 4.50` > patch-pool spread `4.09`, so the
  class-agnostic token is not constant and Euclidean translation-invariance
  does not make removal free

So removal would move every CoFFE cell in Table 2, including the headline
75.30. `CLAUDE.md` hard rule 1 and `PAPER_CANON` §7.1 both say a
behavior-altering change that seems desirable must be reported, not applied.
Phase 1 is also scoped to `docs/refactor/` + `tools/refactor/` only, so code
removal could not happen here regardless — it would be a phase-3 action.

**D3 RESOLVED — Nikola chose option 1: keep λ, documented.** `approved: true`.
No computed number moves. Documentation obligations recorded in
`PAPER_CANON` §8 D3 and on the manifest record:

1. Document the real eval feature `z = mean_j(patch_emb_j) + 0.5·cls_emb` on
   `adapt_embeddings` and in the release protocol description — documenting what
   the code computes, not an erratum against the paper.
2. Honest names for `lambda_factor` / `adapt_embeddings` in phase 4. Verified
   rename-safe: `lambda_factor` is a plain float and appears in none of the 42
   state_dict keys.
3. Document that the `renormalize` branch (`:296-297`) is inert at eval.

Two follow-on facts established while scoping the documentation:

- **λ is CoFFE-only.** `MFTOriginal.adapt_embeddings`
  (`models/mft_original.py:237-245`) and `HyperSIGMACosine.adapt_embeddings`
  (`models/hypersigma/hypersigma_cosine.py:109-117`) are explicit no-op
  pass-throughs, so Table 3 and the MFT control are untouched by it.
- **`MFTCPEACosine.forward_episode` (`:326`) is dead** — a non-executed
  duplicate of the hand-rolled loop in `scripts/evaluate_cosine.py:387-400`.
  Added as risk **R10**: the phase-2 equivalence harness must pin the *live*
  path, or it would certify code the paper never ran.

**Still open (not answered at this gate):** notebook keep-list (default applied:
pretrain, evaluate, compare, adapt_hypersigma_native_sem), the
`significance_report copy.json` rescue (kept as `rename` — it is load-bearing),
R2 (`utils/checkpoints.py` vs canon §7.2, now recorded as D15), and a review of
the remaining 24-entry delete list.

**Phase 5 prerequisite created by the D9 decision:** `archive/` must be added to
`.gitignore` so the archived tree is present on disk but untracked
("non-traceable"), as requested.

---

## Phase 2 — equivalence harness (goldens G1–G5 + checkpoint fixtures)

**Changed:** `tests/equivalence/` (new), `pyproject.toml` (pytest `markers`
only), `docs/refactor/LOG.md`. No behavior touched — nothing under `models/`,
`pretrain/`, `scripts/`, `lib/`, `data/`, `trainers/`, `utils/` or
`third_party/` was modified. Goldens were generated on top of `4d42f64`
(`[phase 1]`), i.e. before any prune/rename/restructure commit exists.

### What was built

```
tests/equivalence/
├── README.md               # what is pinned, and how to re-baseline
├── _harness.py             # scene factory, config builders, fingerprints
├── conftest.py             # determinism contract, session scene fixture
├── make_golden.py          # guarded golden generator
├── golden/                 # 5 committed JSONs + meta.json
├── fixtures/               # 2 committed pre-refactor checkpoints
└── test_equivalence.py     # 32 tests
```

The harness drives the **real paper entry points** — `run_pretrain` from
`scripts/pretrain_enhanced.py` and `run_evaluation` /
`load_model_with_checkpoint` from `scripts/evaluate_cosine.py` (the same
implementations `lib/pretrain_runner.py` and `lib/eval_runner.py` wrap). No
preprocessing, masking, training or eval logic is reimplemented in the harness.

**Synthetic scenes go through the repo's own dataset code.** Rather than
building tensors, the harness writes `.mat` files in the exact on-disk layout of
the real pre-patched scenes (verified against `data/raw/*`: HSI `[N,11,11,C]`
float64, LiDAR `[N,11,11,C_aux]` float64, labels `[1,N]` int64 valued 1..N), so
`_load_split` / `_extract_array` / `_ensure_nchw` / `_ensure_1d` / `_normalize`
/ `_build_class_indices` all execute unmodified. Three mini-scenes:
`houston_mini` (144/1/15), `trento_mini` (63/1/6), `muufl_mini` (64/2/11),
30 px/class, 11×11 patches. No datasets and no HyperSIGMA checkpoints are
required (canon §7.5).

**R10 discharged.** The eval feature is captured off the **live** path
(`forward_features` → `adapt_embeddings` → mean, mirroring
`scripts/evaluate_cosine.py:389-400`), not off the dead
`MFTCPEACosine.forward_episode`. A dedicated test
(`test_dead_forward_episode_still_matches_live_path`) records that the two still
agree, so a later phase that prunes the dead one can show it is
behavior-preserving.

**Configs come from the frozen run configs** (`experiments/*/pretrain_config.yaml`),
not from the stale `configs/` (D13). Only the schedule is scaled: 3 epochs
instead of 1500–3000, batch 16, `warmup_epochs` 1 (required — `CosineAnnealingLR`
gets `T_max = epochs - warmup`), and 20 episodes at `k_query=10` instead of 1000
at `k_query=100`. Every semantic knob is the paper's: arch (D=128, 2 heads,
2 layers, λ=0.5, proj 512/1/L2), mask rates, loss, AdamW settings, and
`distance_metric: euclidean` / `use_projection: False` / `pool_sigma: None` /
N-way full-class / K=5.

### Goldens

| Golden | Content |
|---|---|
| **G1** `g1_pretrain_loss.json` | Per-epoch mean pretraining loss to 8 decimals, 7 configs: CoFFE × {`simmim_band`, `simmim_token`, `simmim_band_token`, `simmim_band_token_houston_run`, `mae`} and MFTOriginal × {`simmim_token`, `mae`}. |
| **G2** `g2_encoder_forward.json` | Every state_dict tensor of a freshly-seeded model (`sum`/`abs_sum`/`norm`) plus the pooled eval feature `z`, for CoFFE HSI+LiDAR, CoFFE HSI-only, MFTOriginal, and the HyperSIGMA wrapper in both paper regimes (`patch_native` 11×11, `backbone_native` 64×64 upscale) with random-init ViT bodies. |
| **G3** `g3_episodic_eval.json` | OA to 6 decimals **and** a SHA-256 over the full per-episode per-query argmin assignment matrix, plus a `key_report` recording exactly which checkpoint keys `fix_state_dict_keys` discards. |
| **G4** `fixtures/*.pth.fixture` | The two pre-refactor checkpoints, loaded through the live loader on every run. |
| **G5** `g5_masking.json` | Realized band/token mask rates (4 decimals, exact), union-mask rate, masked reconstruction loss, and the centre-weight mean — pinning Eq. 1. |

G5 builds its pretraining model with the **pretraining** projection setting, not
the eval one: the projection head sits inside the masked-reconstruction forward
(`masked_modeling_enhanced.py:196`), so an eval-shaped encoder would have pinned
a model pretraining never ran. With that corrected the SimMIM masked losses move
(e.g. `simmim_token` 0.3557 → 0.3214) while MAE's is unchanged at 1.3556 — MAE
has no projection head, exactly as `run_pretrain` forces
(`pretrain_enhanced.py:304-309`). Mask rates are unaffected either way.

Sanity anchors that came out right on the first run: the CoFFE eval encoder has
**579,328** parameters, matching canon §2 exactly; HyperSIGMA is 180.4 M
(`patch_native`) / 188.3 M (`backbone_native`), matching canon's "~180M"; the
realized mask rates land on 0.8499 (band 0.85), 0.7521 (token 0.75 →
`round(0.75·121)=91`, 91/121) and 0.9628 union (analytic
`1−0.15·0.25 = 0.9625`); centre weights have mean 1.0 to fp32.

Tolerances: eval assignments, realized mask rates and state_dict **key sets** —
exact; floats — `rtol=1e-6, atol=1e-8`. Nothing was loosened.

### Determinism and stability

CPU only; `random`/`numpy`/`torch` seeded; `torch.use_deterministic_algorithms(True)`
(raised nothing on any path); `num_workers=0`; no AMP; episode seed 42. The
switch is restored on teardown so the rest of `tests/` does not inherit it.
Two no-op env settings were deliberately **not** included: `PYTHONHASHSEED`
(inert once the interpreter is running) and `TQDM_DISABLE` (tqdm 4.67.3 does not
read it) — golden JSON is written `sort_keys=True` and every key comparison is
over a sorted set, so nothing depends on hash ordering.

Stability evidence:

- `pytest tests/equivalence` — **32 passed** twice back-to-back (35.8 s, 36.1 s),
  and again with the default `addopts` coverage flags on (40.8 s).
- The goldens are generated in one process and re-derived in another, so every
  fingerprint already round-trips across process boundaries.
- Independent check: retraining both fixtures from scratch in a fresh process
  gives **bit-identical** weights (`torch.equal` on all 52 / 55 tensors).
- Strongest check: a full `make_golden.py` re-run from scratch reproduced all
  four golden JSONs *and* both fixture checkpoints **byte-identically**
  (`md5sum -c`, 6/6 OK).
- Full suite `pytest tests -q`: **94 passed** = the 62 of the phase-0 baseline
  plus 32 new. No pre-existing test changed behavior.
- Marker filters work: `-m "not gpu and not data"` selects all 94;
  `-m "not slow"` deselects the 2 HyperSIGMA ViT builds.

**The harness demonstrably bites.** Two mutations were applied to the real code,
run against the goldens, and reverted (`git checkout`, tree verified clean):

| Mutation | Result |
|---|---|
| `models/mft_cpea_cosine.py:294`, the λ term scaled by `1.0000001` (relative 1e-7) | **3 failed** — `test_g2_encoder_forward` for both CoFFE variants and `test_g4_fixture_forward[coffe]`. G3 correctly did not fire: 1e-7 flips no argmin. |
| `pretrain/masked_modeling.py:95`, `int(round(0.75·N))` → `int(0.75·N)` (91 → 90 masked tokens) | **7 failed** — all four G1 trajectories that use token masking, plus the three G5 entries. |

So the fine-grained detectors (G2/G4 fingerprints, G5 rates) catch sub-ULP drift
while G3's assignment hash catches decision-level change — the intended division
of labour.

`make_golden.py` guards were tested by deliberately tripping them: a dirty
tracked file outside the phase-2 scope refuses (`README.md` probe), and an
extra `[phase 3]` commit refuses with the re-baseline instructions. `--rebaseline`
waives only the commit-range check, still refuses on an out-of-scope dirty tree,
and stamps `rebaselined_from` into `meta.json`.

### Optional real-checkpoint fingerprint — captured

`COFFE_EXPERIMENTS_DIR` is unset by default, but the real experiment tree and
the headline checkpoint are on this machine, so it was worth running:

```
COFFE_EXPERIMENTS_DIR=./experiments python tests/equivalence/make_golden.py --real
```

50 fixed-seed episodes (`k_query=100`, 15-way, K=5, euclidean, CPU) on
`houston_enhanced_spatial_mask_test_run1_seed52/checkpoints/checkpoint_epoch_950.pth`
— the run behind Table 2's headline Houston cell (D14):

**OA = 74.9493 ± 0.7093** vs. the paper's **75.30** (1000 episodes).

Within the CI, and consistent with phase 1's independent CPU probe (74.85 on
40 episodes). This is the one measurement that ties the harness's code path to a
real paper number rather than to a synthetic scene. Written to
`golden/real_local.json`, excluded by `tests/equivalence/golden/.gitignore`,
never asserted against (machine-specific).

`golden/meta.json` records python 3.12.3 / torch 2.11.0+cu128 / numpy 2.4.4, the
git SHA (`4d42f64`) and a **per-group `groups` record**, so a partial
regeneration (`--only g5`) can no longer erase the provenance of the other
groups — a flaw the verifier caught and which is now fixed. `"dirty": true` is
expected and documented in the harness README: the harness is necessarily
uncommitted when it first generates its own goldens, and `make_golden.py`
separately refuses if any tracked file *outside* `tests/equivalence/` is
modified, so the code under test is never dirty. `test_golden_metadata_recorded` fails loudly with re-baseline
instructions if the torch version ever moves, rather than letting the harness
drift into tolerance-loosening.

### Surprises reported, not fixed (hard rule 7)

- **D19 (new) — the Houston band+token cell used a mask rate the paper does not
  state.** `PAPER_CANON` §1 defines SimMIM band+token as `(r_b, r_s) =
  (0.85, 0.75)`, and D13 records that the paper's rates "match all six canonical
  run configs". They do not. Five of the six do; the sixth —
  `experiments/houston_enhanced_spec_spat_combined`, which produces Table 2's
  CoFFE SimMIM band+token / HSI+LiDAR / Houston = **64.63** — used
  `band_mask_ratio: 0.75`. Verified across all six:

  | run | band | token |
  |---|---|---|
  | `houston_enhanced_spec_spat_combined` | **0.75** | 0.75 |
  | `trento_enhanced_spectral_spatial_run2` | 0.85 | 0.75 |
  | `muufl_enhanced_spectral_spatial_run2` | 0.85 | 0.75 |
  | `houston_enhanced_spectral_spatial_no_lidar` | 0.85 | 0.75 |
  | `trento_enhanced_spectral_spatial_no_lidar` | 0.85 | 0.75 |
  | `muufl_enhanced_spectral_spatial_no_lidar` | 0.85 | 0.75 |

  Nothing was changed. The harness pins **both** rates (G1/G5 entries
  `simmim_band_token` at 0.85 and `simmim_band_token_houston_run` at 0.75) so
  neither can drift. Needs a decision at the gate: whether the reproduction docs
  state the per-cell rate, and whether D13's claim in `PAPER_CANON` §8 is
  corrected.

### Scope note for a later phase

`fixtures/` checkpoints are named `*.pth.fixture`, not `*.pth`, because
`.gitignore` has a blanket `*.pth` rule that would silently drop them — the same
trap D5 records for `lib/`. Phase 2 is scoped out of touching `.gitignore`, so
the workaround stands and is documented in `fixtures/README.md`. **Proposed for
phase 5:** add `!/tests/equivalence/fixtures/*.pth` and rename them back.
`golden/real_local.json` is excluded via a nested
`tests/equivalence/golden/.gitignore` (inside the phase's scope).

### Open question for the gate

Only D19 above. The harness itself has no open questions.

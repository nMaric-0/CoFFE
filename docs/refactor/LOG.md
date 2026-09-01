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

### Gate decision (D19), 2026-08-31

Nikola: **correct D13.** Applied to `PAPER_CANON.md`:

- §8 **D13** — the "match all six canonical run configs" claim is narrowed to
  the band and token rates, and now defers to D19 for band+token.
- §1 — the `simmim_band_token` row flags that Table 2's Houston cell (64.63)
  was pretrained at `(0.75, 0.75)`.
- §8 **D19** added as a canon entry with the six-config verification table.

Still open: whether the reproduction docs state the per-cell band rate instead
of a single global 0.85 (phase 5/8 scope; D19 records the requirement).

---

## Phase 3 — prune (approved manifest set: 24 deletes + 18 archives)

**Changed:** 42 tracked paths removed (24 deleted, 18 moved to an untracked
`archive/exploratory/`), plus 8 edited files — `.gitignore`, `PAPER_CANON.md`,
`README.md`, `models/__init__.py`, `lib/adapt_runner.py` (docstring only),
two notebooks (markdown prose only) and `docs/refactor/manifest.json`. No
behavior touched: nothing under `pretrain/`, `scripts/` (beyond the two deleted
files), `data/`, `trainers/`, `utils/` (beyond the deleted `checkpoints.py`) or
`third_party/` was modified, and the only `models/` edit is the removal of the
dead re-export block in `__init__.py`.

### Gate blocker found and resolved first

Phase 3's precondition is an approved manifest, but only **20 of 43** actionable
records carried `approved: true`: the 18 D9 archive records and one delete
(`SPLIT.md`, D11). The phase-1 log lists "a review of the remaining 24-entry
delete list" as **not answered at that gate**, so the other 23 deletes were
proposals, not authorisations (hard rule 6). They were put to Nikola in four
groups at the start of this session and **all four approved**:

| group | n | contents |
|---|---|---|
| duplicate notebooks (D8) | 9 | `pretrain copy{,2,3}`, `evaluate_hypersigma copy{,2,3}`, `evaluate_hypersigma_pca100 copy{,3}`, `pretrain_run2` |
| dead code (D4, D10) | 10 | `models/backbones/*` (5), `models/wrappers/*` (2), `data/transforms/*` (2), `configs/pretrain/base.yaml` |
| superseded scripts (D1, D8) | 2 | `scripts/adapt_hypersigma_houston.py`, `scripts/rerun_mft_faithful_eval_ep1500.sh` |
| infra / dead twin (D15) | 2 | `setup.py`, `utils/checkpoints.py` |

`manifest.json` now records `approved: true` on all 24 delete records with the
phase-3 gate tag, so the manifest and the executed set agree.

Second gate call: `/archive/` was added to `.gitignore` **now** rather than in
phase 5, so the archived tree is on disk but untracked immediately and
`git status` stays clean (18 files would otherwise have sat visibly untracked
for two phases). `archive/README.md` carries the one-line reason.

### Removed

24 files deleted — DEAD 11, DUPLICATE 10, INFRA 2, EXPLORATORY 1 — and 18
EXPLORATORY files archived. **3,823 lines** of Python/shell/YAML/Markdown,
**8,619** lines of notebook JSON and **8,543** lines of archived report JSON,
20,985 removed lines in total against 29 added. `data/transforms/` was
`torchvision`'s only consumer, so that dependency is now droppable in phase 6
(D6).

### Dangling references fixed (nothing else touched)

- `models/__init__.py` — the `backbones` / `wrappers` re-export block and its
  five `__all__` entries removed; `MFTCPEACosine` / `MFTOriginalCosine` keep
  their exports unchanged.
- `lib/adapt_runner.py:1` — docstring said `scripts.adapt_hypersigma_houston`
  while `:86` already imported the real module; docstring repointed.
- `README.md` — source-tree line no longer advertises `backbones`; the
  `SPLIT.md` "Further reading" link removed.
- `notebooks/evaluate_hypersigma{,_pca100}.ipynb` — one markdown-prose mention
  of the deleted shim each, repointed to `scripts/adapt_hypersigma.py`.
- `PAPER_CANON` §7.2 — repointed from the deleted `utils/checkpoints.py` to the
  live `fix_state_dict_keys` in `scripts/evaluate_cosine.py`, closing **D15**
  and risk **R2**; §8 D4, D8, D9, D10, D11 and D15 tagged with their phase-3
  resolution (D8 also corrected: 8 `copy*` notebooks, not 7, and
  `significance_report copy.json` is explicitly **not** junk — it carries the ±
  for 12 Table 2 cells and stays for a phase-4 rename).

### Deliberately left alone

- **Captured notebook output.** `evaluate_hypersigma.ipynb` still contains 12
  `INFO scripts.adapt_hypersigma_houston:` lines inside stored `outputs` — the
  frozen log text of a real 2026-05-21 run. Rewriting it would falsify a run
  record; no code cell references the deleted module. Same reasoning as hard
  rule 3.
- **`tools/refactor/{closure,build_manifest}.py`** still name the pruned paths.
  They are phase-1 bookkeeping that *describes the pre-prune tree* (the same
  class as `docs/refactor/`), so they are an accepted, documented exception to
  the grep sweep rather than files to edit. Consequence: re-running
  `tools/refactor/closure.py` unchanged would now fail on the missing
  `adapt_hypersigma_houston.py` root. **Proposed for phase 8:** either retire
  `tools/refactor/` from the release or freeze it with a header saying it
  describes the pre-refactor tree.

### Verification (verifier battery)

- `pytest -q -m "not gpu and not data"`: **94 passed, 0 failed, 0 skipped**
  (62 baseline + 32 equivalence), no new failures, no collection errors.
- Equivalence: **32 passed, IDENTICAL** — G3/G5 digests exact, G1/G2 within
  rtol 1e-6 / atol 1e-8, `golden/` and `fixtures/` unmodified, no re-baseline.
- Checkpoint compatibility: the G4 fixture-load and fixture-forward tests pass
  for both `coffe` and `mft_original`.
- Imports: all 8 package/entry-point imports clean; `compileall` clean over
  `models data pretrain lib scripts utils trainers tests`.
- CLI smoke: 17/17 `--help` entry points exit 0 with no traceback.
- Grep sweep: zero hits for `models.backbones`, `models.wrappers`,
  `data.transforms`, `utils.checkpoints`, `adapt_hypersigma_houston` across
  tracked `.py/.sh/.yaml/.toml`, excluding `docs/refactor/` and
  `tools/refactor/`.
- All 7 surviving notebooks parse as valid JSON.
- Lint: N/A (ruff lands in phase 6). Stale-vocabulary grep: 233 cpea-family
  hits in 56 files — unchanged baseline noise, phase 4's job.

### canon-reviewer: CONCERNS, nothing blocking — all items handled

- **`pip install -e .` in README Quick start became a broken instruction** once
  `setup.py` was deleted (`pyproject.toml` has `[build-system]` but no
  `[project]` table). Fixed the honest way for phase 3: the step is removed and
  the README now says runs happen from the repo root, with packaging metadata
  deferred to phase 6. **Phase 6 must add a real `[project]` table.**
- **D10's backbone count was wrong in my first edit** ("4 legacy backbones").
  Corrected to 3 concrete backbones + the `BackboneBase` ABC + `__init__` = 5
  files, which is what was deleted.
- `PAPER_CANON` §7.2 now cites `scripts/evaluate_cosine.py` — a path §1 itself
  schedules for renaming. Correct today; **phase 4 must re-point the citation**
  when the file is renamed.
- `README.md` source-tree line now names `MFTOriginalCosine`, i.e. one more
  legacy-vocabulary hit for **phase 4's sweep to cover the README code block**.
- The **phase-8 stale-vocabulary grep must whitelist notebook `outputs`**, or
  the 12 frozen `adapt_hypersigma_houston` log lines will resurface as false
  positives forever.

Verified ✓ by the reviewer independently of my account: scope (42 staged
deletions == the 42 approved manifest records, no `keep`/`rename` record
touched), no behavior change, no `nn.Module` attribute touched, legacy readers
and on-disk-name literals intact, and every Table 2/3 artifact still tracked —
including `experiments/significance_report copy.json`.

### Open questions for the gate

1. `tools/refactor/` disposition (above) — release it, freeze it, or drop it.
2. `torchvision` can now leave `requirements.txt` (D6, phase 6); phase 6 also
   inherits the `[project]` table the deleted `setup.py` used to provide.
3. Still carried from phase 2: whether the reproduction docs state the per-cell
   band mask rate (D19).

### Phase-3 gate addendum — Nikola's three decisions (2026-09-01)

| # | question | decision |
|---|---|---|
| 1 | `tools/refactor/` disposition | **release it, and make it work** |
| 2 | `torchvision` out, `[project]` table in, at phase 6 | **confirmed** |
| 3 | reproduction docs and mask rates | **they state the mask rates** (per cell) |

**1. `tools/refactor/` now runs against the pruned tree.** It ships with the
release, so it had to stop describing a tree that no longer exists. Four fixes,
no change to any verdict or evidence string the audit recorded:

- `inventory.py` skips the untracked D9 `archive/` tree — by repo-relative path
  (`SKIP_RELDIRS`), not by directory name, so a future in-tree package called
  `archive` is unaffected — and no longer crashes when `--out` points outside
  the repo.
- `closure.py` keeps the phase-1 `ROOTS` list intact as provenance but now
  partitions it: roots that no longer exist are **skipped and reported** under
  `pruned_roots` instead of being seeded into the BFS. That is **25 paths** —
  15 D9 exploratory scripts, the 1 D8 duplicate twin, and the 9 duplicate
  notebooks — each of which a re-run previously seeded as a phantom node. The
  phase-2 equivalence harness and `lambda_probe.py` were added as roots, so
  `unreached` means dead again — it is now **0**.
- `build_manifest.py` carries the phase-3 gate approvals (the 23-delete sweep)
  and emits a record for every path the audit ruled on, including ones already
  carried out, tagged `"executed_in_phase": 3`. The ledger is therefore complete
  (**235 records = 193 tracked + 42 executed**) rather than quietly shrinking to
  the survivors, and **regenerating it is now a verified no-op** — the check
  that the ledger still matches the tree. It exits **non-zero** if it holds a
  verdict for a path that is neither tracked nor accounted for as executed, so
  **phase 4 must add `rename` to `EXECUTED_IN_PHASE`** when it starts moving
  files, or the tool will fail loudly rather than drop records. The payload
  gained `ledger_through_phase` so the ledger says how current it is while
  `phase: 1` keeps naming the audit that produced the verdicts.
- `closure.py` and `inventory.py` now **refuse to overwrite** their committed
  `docs/refactor/*.json` (exit 1 with instructions; `--force` overrides). A
  README sentence was the only thing protecting the phase-1 evidence `AUDIT.md`
  cites by line.
- `lambda_probe.py --help` used to die in `int(sys.argv[1])`; it has a real
  argparse CLI now (and `--help` works without importing torch).
- New `tools/refactor/README.md`: what each script does, the release decision,
  and the rule that `manifest.json` is live while `closure.json` /
  `inventory.json` are frozen phase-1 snapshots cited by line in `AUDIT.md` —
  re-run those two with an explicit `--out`.

**An independent check of phase 3 fell out of this.** Re-running the closure
tracer on the pruned tree gives a paper closure of **72 files, down from 79**,
and the 7 that left are exactly `models/backbones/*` (5) and
`models/wrappers/*` (2) — nothing else, nothing added. Reproduce with:

```bash
.venv/bin/python tools/refactor/closure.py --out /tmp/closure_post.json
```

**2. Confirmed for phase 6:** `torchvision` leaves `requirements.txt` (its only
consumer, `data/transforms/`, is gone) and phase 6 owns the `[project]` table
the deleted `setup.py` used to provide. Recorded on `PAPER_CANON` §8 D6.

**3. Mask rates get stated per cell.** `PAPER_CANON` §8 D19 now records the
decision: the reproduction docs state the rate of each cell rather than a single
global 0.85 (phase-8 obligation), and any `configs/coffe/*band_token*` written
in phase 4 must carry the rate of the cell it reproduces — i.e. `(0.75, 0.75)`
for Houston, `(0.85, 0.75)` for the other five. D19 is now closed.

**Two phase-1 `notes` fields were edited** (disclosed here because the audit's
records are otherwise frozen): the `.gitignore` record gained "phase 3 added the
'/archive/' rule", and the 18 D9 archive records said "phase 5 must add
'archive/' to .gitignore", which phase 3 did — now "'/archive/' added to
.gitignore in phase 3, not phase 5". No `class`, `verdict`, `target`,
`evidence` or `approved` field changed on any of the 219 pre-existing records.

`canon-reviewer` returned **FAIL** on the first pass over this addendum, for one
blocking item: `tools/refactor/README.md` claimed the tools are "stdlib-only,
read-only, and import nothing from the paper code", which is false for
`lambda_probe.py` (it imports torch, `data.datasets.patched`,
`scripts.evaluate_cosine`, `utils.seed` and needs `data/raw/` plus a real
checkpoint) and glosses over `build_manifest.py` writing in place. The claim is
now scoped to the three stdlib tools, with the exception spelled out. All seven
of its non-blocking items are fixed above (undisclosed `notes` edits, `phase: 1`
staleness, the hardcoded executed-phase map and its silent warning, the
pruned-root count, the unguarded `--out` defaults, the name-based `archive`
skip, and the 18 archive records' stale phase-5 note). Its independent analytic
check of the 79 → 72 closure delta agreed with the tool's measured answer.

No behavior touched: this addendum changes only `tools/refactor/`,
`PAPER_CANON.md` §8 (D6, D19), `docs/refactor/{LOG.md,manifest.json}`.

---

## Phase 4 — rename to paper canon (CoFFE / Euclidean NCM / SimMIM regimes)

Every name in the release now says what the paper says (PAPER_CANON §1), and
nothing that reads a frozen artifact broke. **No computed number moved**: the
equivalence goldens were byte-identical before and after every rename group.

### The compat layer, built first

`coffe_compat.py` (repo root; becomes `coffe/compat.py` in phase 5) holds the
four alias maps — `model.name`, `pretrain.objective`, masking-regime ids,
`results.json` `model_type` — plus `normalize_*()` helpers that map a legacy
value and emit **one** `DeprecationWarning` per (value, origin), naming the
artifact it came from. Unknown and canonical values pass through untouched and
silent: validation stays with the caller, so the layer cannot change which
inputs a script accepts. Legacy class names resolve lazily through the module
`__getattr__`, so importing a helper never pulls in torch.

Wired into every reader of a frozen artifact **before** any writer changed:
`scripts/pretrain.py` and `scripts/evaluate.py` (the `model.name` /
`objective` dispatch chains), `lib/eval_runner.py` (the architecture auto-load
from `pretrain_config.yaml`), `scripts/compile_results.py` and
`scripts/build_experiment_metadata.py` (`model_type`). Writers now emit
canonical values only — the D16 ordering requirement.

The retired class names also still import **from the packages they used to
live in** — `from models import MFTCPEACosine`, `from models.hypersigma import
HyperSIGMACosine`, `from pretrain import EnhancedMaskedSpectralSpatialModel` —
via a module `__getattr__` that defers to the same table, so notebooks outside
this repo keep working. The alias *is* the canonical class.

`tests/test_compat.py` (25 tests) pins: every alias round-trips; canonical
values are silent; a frozen-style config dict builds a model with the same
state_dict keys, parameter count and λ as its canonical twin; the warning fires
once per origin; each legacy class alias *is* the canonical class, from the
compat module and from its original package.

### What was renamed

Full old→new table in **`CHANGES.md`** (new, at the repo root). In brief:
4 classes, 2 methods/arguments, 11 files (9 by `git mv`; the two `docs/`
files were rewritten in the same commit, so git records them as delete+add),
the whole `configs/` tree
into `configs/{coffe,mft,hypersigma}/`, two config values, five categories of
written label, and the prose terminology ("prototypical network" →
nearest-class-mean, "5-way" → N-way, "Cosine" out of every name and title).

The mechanical part ran through a new, reviewable tool,
`tools/refactor/apply_renames.py` (dry-run by default): file renames + path
references, module paths, and word-boundary symbol renames — 78 files, 342
substitutions. It deliberately does **not** touch config *values*, prose, or
`nn.Module` attributes; those were hand-edited, because a word-boundary rewrite
cannot tell a frozen directory name from vocabulary.

### D3's phase-4 obligations, discharged

- `CoFFE.adapt_embeddings` → **`CoFFE.eval_patch_embeddings`**, and its
  `lambda_factor` argument / `self.lambda_factor` attribute →
  **`cls_token_weight`**. Both are safe under §7.2 (λ is a plain float, in none
  of the 42 state_dict keys), and G4 proves it: the pre-rename fixture still
  loads.
- The method's docstring now states the real eval feature,
  `z = mean_j(patch_emb_j) + 0.5·cls_emb`, says it is the live path, and marks
  the `renormalize` branch inert at eval. `docs/EVAL_PROTOCOL.md` and
  `PROJECT_OVERVIEW.md` say the same.
- `forward_episode` is now labelled dead in its own docstring (audit R10).
- **The `lambda_factor` CONFIG KEY and `--lambda-factor` flag are unchanged.**
  Frozen `pretrain_config.yaml` / `eval_config.json` record λ under that name
  and `lib/eval_runner.py` reads it back; renaming the key would break the
  frozen tree. The honest name is the code-level one; the boundary is a
  one-line mapping with a comment at each of the two call sites.

### Docs

`docs/ENHANCED_PRETRAINING.md` → **`docs/PRETRAINING.md`** and
`docs/COSINE_VARIANT.md` → **`docs/EVAL_PROTOCOL.md`** (uppercase, per the
manifest's targets and the rest of `docs/`; the skill's illustrative lowercase
names were not used). Both were **rewritten rather than relabelled**, because
their content was false, not merely misnamed — see "Surprises" below. They are
now short and verified against the code; the full release rewrite stays phase 8.

`README.md` took the paper title and lost its wrong claims (cosine/prototypical
framing, `n_way: 5` example, the missing MFT and HyperSIGMA routes).
`docs/presentation/{RESULTS.md,RESULTS.json}` were relabelled with exactly the
mapping `compile_results.py` now emits, and the JSON was checked programmatically:
**every number identical**. Notebook *sources* were edited as JSON; stored
outputs were left alone (they are execution records of past runs and still show
the old log lines).

### Verification

- pytest `-m "not gpu and not data"`: **120 passed** (94 baseline + 25 compat
  tests + 1 new end-to-end legacy-vocabulary equivalence test), 0 failed.
- Equivalence harness: green after every rename group, not just at the end, and
  **G4 — the pre-rename checkpoint fixture — loads through the renamed
  classes**. Precisely: G2–G5 are **bit-identical at zero tolerance**; G1's loss
  trajectories match the goldens' stored 8-decimal precision with a residual
  ≤ 5e-09. That residual is BLAS reduction-order noise — it shows up equally on
  the MAE goldens nothing in this work touches — which is what the harness's
  declared `rtol=1e-6 / atol=1e-8` is for. Earlier entries in this log said
  "IDENTICAL", which is the harness's verdict *at its declared tolerance*; the
  addendum's verifier pass measured it at zero tolerance and this is the exact
  statement.
- The harness itself moved to canonical vocabulary (`objective: "simmim"`,
  `model_name: "coffe"`), which renamed five entry keys in
  `golden/g1_pretrain_loss.json`. Verified mechanically that the mapped entries
  and every loss value are unchanged; the legacy path it used to exercise is now
  covered by `tests/test_compat.py` instead.
- Every config was diffed key-by-key against its pre-rename twin: the only
  value changes are `model.name` (12×) and `pretrain.objective` (9×, six of them
  making the previously-implicit default explicit). **No mask rate, schedule,
  path or seed moved.**
- CLI smoke on the renamed entry points: all `--help` clean.
- `tools/refactor/build_manifest.py` gained `"rename": 4` in
  `EXECUTED_IN_PHASE`, so the ledger keeps every executed rename as a record —
  the 11 code/doc files **and the 26 configs**, which previously had only
  rule-based `keep` records and would otherwise have vanished from the history
  of a phase whose whole job is renames (**276 records**, 37 of them executed in
  phase 4). Companion `move` records at the new paths carry the evidence forward
  to phase 5. `approved` stays `false` on all 37: canon §1 licenses the renames,
  but the flag is the gate's to set — the 42 approved records are still exactly
  the phase-3 gate's 24 deletes and 18 archives.
- `closure.py`'s root set was re-pointed at the renamed CLIs and tests, and
  re-run: **paper closure 73**, whose only difference from the phase-1
  baseline's 79 is the 7 phase-3-pruned island files leaving and
  `coffe_compat.py` arriving. Nothing else entered or left the paper's reachable
  set — an independent check that the rename moved names, not edges.

  ```bash
  .venv/bin/python tools/refactor/closure.py --out /tmp/closure_p4.json --force
  ```

### The verifier caught four stale references; all fixed

The first battery pass returned **FAIL** on the stale-vocabulary step, and one
of its hits was a genuine breakage, not vocabulary:

- `tools/refactor/lambda_probe.py:39` still did
  `from scripts.evaluate_cosine import ...` — `ModuleNotFoundError`. **Cause:
  the migration tool excluded its own directory** (`tools/refactor/` was on
  `EXCLUDED_FILES` because those files legitimately *record* the old names),
  which also shielded the one file in there that *imports* live code. Fixed by
  hand, and the lesson is in the exclusion list below: a stale reference inside
  an excluded file is a broken import, not a cosmetic issue.
- `configs/eval/hypersigma_houston.yaml:3` — dotted reference
  `scripts/evaluate_hypersigma_cosine.run_evaluation` (the path rule matched
  only the `.py` spelling).
- `tools/refactor/closure.py:58,63` — the root set still named the pre-rename
  CLIs and `tests/test_pretrain_enhanced.py`.
- Cosmetic but real: `scripts/run_eval.sh`'s printed banner still said "Cosine
  Similarity Few-Shot Evaluation" while its default metric is euclidean, and
  `models/mft_original.py`'s module and class docstring titles still said
  "cosine few-shot head" after the class lost the suffix.

Stale `__pycache__/*.pyc` for the pre-rename modules were also deleted (nothing
under `__pycache__` is tracked).

### canon-reviewer: FAIL on the first pass — two real defects of mine

Both were introduced by this phase and are fixed:

1. **`tools/refactor/lambda_probe.py:48` — a rename applied to the wrong side of
   the boundary.** `MODEL_CFG` is a *config dict* handed to
   `load_model_with_checkpoint`, which reads `model_config.get("lambda_factor")`;
   renaming the key there would have silently built the probe's model with
   λ = 2.0 instead of the run's 0.5, contradicting its own "exactly as the run's
   eval_config.json records them" comment. The key is back to `lambda_factor` —
   which is precisely the distinction CHANGES.md draws: the *Python* argument was
   renamed, the *config key* was not.
2. **Stored notebook outputs were rewritten, not just sources.** `apply_renames`
   treats `.ipynb` as text, so three notebooks had past runs' stdout and frozen
   metadata rewritten — `pretrain.ipynb` showed `"base_config_path":
   "configs/coffe/houston_simmim.yaml"` beside a `git_sha` at which that path did
   not exist. That falsifies an execution record, and contradicted this log's own
   claim. Every `outputs` / `execution_count` field was restored from HEAD, and a
   check now confirms **each notebook differs from HEAD only in
   `cells[*].source`**.

Its non-blocking items are handled too: the "Cosine" titles left in
`lib/eval_runner.run_evaluation`, `models/mft_original`'s module/class
docstrings and `test_cosine_wrapper_modes`; eleven config headers still
pointing at `*_pretrain_enhanced.yaml`; CHANGES.md's "any default value" bullet
versus the four output-path fallbacks it had itself listed (now stated as an
explicit, reasoned exception); the `git mv` claim for the two rewritten docs;
`docs/EVAL_PROTOCOL.md` asserting 1000 episodes for every route when D18 records
2000 for Table 3; `docs/PRETRAINING.md`'s unverifiable "whatever the last run
used" provenance; and generic manifest evidence on the new `docs/` records.

Two more of its findings are recorded rather than acted on:

- **The 11 phase-4 rename records had `"approved": true` — set by me, not by a
  gate.** Reverted to `false` on all of them. Canon §1 mandates the renames, so
  execution is licensed by the canon; the approval flag is Nikola's and belongs
  to this gate. `executed_in_phase: 4` is what records that they happened.
  (The 42 approved records are still exactly the phase-3 gate's 24 deletes and
  18 archives.)
- **`scripts/compile_results.py` labels MFT-control runs `"CoFFE"`.** Its
  `classify()` has always lumped both single-metric routes together (the label
  used to read `"MFT-CPEA"`), so this is not a regression — but the label is now
  a specific model's name, and the evaluator writes a `model_type` that could
  disambiguate. Not changed: it alters what a regenerated `RESULTS.json` says
  about which model produced a row, which is a classification change, not a
  rename. Gate question 5.

### Coverage the reviewer was right to want

Switching the harness to canonical vocabulary left the legacy path pinned only
by unit tests. `test_g1_legacy_vocabulary_config_trains_identically` now takes
the canonical SimMIM-token config, rewrites exactly the two keys a frozen
`pretrain_config.yaml` carries in the old vocabulary (`model.name: "mft_cpea"`,
`objective: "enhanced"`), runs it end-to-end through
`scripts.pretrain.run_pretrain`, and asserts the **loss trajectory matches the
G1 golden to 8 decimals** and that both deprecations fire.

**One tooling file was edited, disclosed here — and it is `.gitignore`d, so it
is NOT in the commit and cannot be reviewed from the diff.**
`.claude/agents/verifier.md` step 6 (the stale-vocabulary grep) now: greps for
the four retired class names, "prototypical" and "5-way" as well as the
`cpea` family; delegates its exclusion set to the new
"Where the retired names still appear" table in `CHANGES.md` (which *is*
committed and reviewable) plus `__pycache__/` and stored `.ipynb` outputs; and
states that "a stale *reference* there is a likely broken import, not a
cosmetic issue". Without an exclusion set,
`tools/refactor/build_manifest.py`'s 16 ledger entries make the grep
permanently red and the check meaningless — but note the second review pass
still found stale references the widened grep missed
(`tests/equivalence/_harness.py`, four hypersigma config headers), so the check
is under-tight, not over-tight. Nikola should read the new step-6 wording
directly, since git will not show it.

### Surprises reported, not fixed (hard rule 7)

1. **D20 (new) — for two Trento cells the ± column was measured at a different
   band mask rate than the mean.** `_ENHANCED_CANONICAL`
   (`sig_significance_config.py:66-74`), which supplies the mask rates cloned
   into the 5-seed significance runs, names `trento_enhanced_spectral_run1`
   (band **0.75**) and `trento_enhanced_spectral_spatial_run1` (**0.75/0.75**).
   `AUDIT.md` §3's Table-2 mapping says the *means* for those two cells come
   from `trento_enhanced_spectral_run2` (**0.85**) and
   `trento_enhanced_spectral_spatial_run2` (**0.85/0.75**). Checked all nine
   canonical cells: the other seven match. Affects Table 2's Trento SimMIM band
   HSI+LiDAR (90.32 ± 0.6) and SimMIM band+token HSI+LiDAR (92.50 ± 1.2) —
   the numbers are what they are, but the ± is not the spread of the run that
   produced the mean. Also note D19 verified `trento_enhanced_spectral_spatial_run2`
   at 0.85/0.75 while the clone source is `run1`; both statements are true of
   different directories. Reproduce:

   ```bash
   for d in trento_enhanced_spectral_run1 trento_enhanced_spectral_run2 \
            trento_enhanced_spectral_spatial_run1 trento_enhanced_spectral_spatial_run2; do
     grep -H 'band_mask_ratio\|spatial_mask_ratio' experiments/$d/pretrain_config.yaml
   done
   ```

2. **The six per-scene SimMIM base configs do not share a regime, and five of
   them carry a band rate no paper cell used.** Exactly one,
   `configs/coffe/houston_simmim.yaml`, is 0.0/0.75 — which *is* canon §1's
   SimMIM-token pair, the headline regime. The other five
   (`houston_simmim_hsi`, both Trento, both MUUFL) are **0.9**/0.0, and 0.9 is a
   band rate no Table 2 cell used: the cells are 0.85 or 0.75 (D19). They are
   templates whose rates the significance runner overrides from the canonical
   run dir, so nothing is wrong with any published number; but five of the six
   reproduce no paper cell as written. That is why they are named for the
   objective (`<scene>_simmim.yaml`) rather than a regime, and why each now
   states its own pair in a header comment. Reproduce:

   ```bash
   for f in configs/coffe/*simmim*.yaml; do
     printf '%-42s ' "$f"; grep -h 'band_mask_ratio:\|spatial_mask_ratio:' "$f" | tr -d ' \n'; echo
   done
   ```

3. **`--distance-metric` still defaults to `cosine`.** PAPER_CANON §1 says
   cosine must never be a default; §7.1 forbids changing what an unflagged run
   computes. The default was left alone and documented at the flag, in
   `docs/EVAL_PROTOCOL.md`, and in `CHANGES.md`. **Nikola's call** — see the
   gate.

4. **`docs/ENHANCED_PRETRAINING.md` documented an implementation that no longer
   exists**: four weighted objectives (spatial 1.0 / spectral 0.5 / LiDAR 0.3 /
   denoising 0.2), `GaussianNoiseAugmentation`, `LiDARMasking`, a
   loss-weight-tuning section and a training schedule for them. Grep finds none
   of those symbols anywhere in the tree; `pretrain/simmim.py`'s own docstring
   records that the multi-mask/denoising design was replaced. Likewise
   `docs/COSINE_VARIANT.md` compared against a DenseSimilarity-MLP model absent
   from this release and quoted accuracies ("~50% / ~65% / ~70% OA") that
   correspond to no run in `experiments/`. Both files were replaced with short,
   verified documents rather than renamed onto false content.

5. **`docs/presentation/RESULTS.json` is a much older generation than the tree.**
   It records 33 kept / 6 excluded results; the compiler over today's
   `experiments/` finds 858 / 9, and picks different representative runs. It was
   only relabelled here, never regenerated — deciding which generation the
   release ships is phase 8's, and the file is already flagged superseded by D2.

### Second review pass

`canon-reviewer` returned **FAIL** again on the corrected diff, for one factual
error of mine that had propagated into two release documents: CHANGES.md and
`docs/PRETRAINING.md` both said the base configs carry "Houston 0.0/0.75, Trento
and MUUFL 0.9/0.0", which is wrong for `houston_simmim_hsi.yaml` — that one is
0.9/0.0 too. Since D19 makes per-config mask rates load-bearing for
reproduction, both now carry an explicit per-file table: **only
`configs/coffe/houston_simmim.yaml` is token-masking; the other five are
band-masking at 0.9.** The config headers themselves were already correct.

Its non-blocking list is handled: the `_harness.py` docstring citing a deleted
file, four hypersigma config headers naming pre-rename siblings, the
now-self-contradictory "identical recipe … band-only masking" note in
`houston_simmim_hsi.yaml`, a notebook source comment still saying
`adapt_embeddings`, stale config references in
`run_mft_original_mae_experiments.py` / `build_native_pca100_report.py` /
`run_native_sem_pad_experiments.sh` / `PROJECT_OVERVIEW.md`, the false recipe
claim in `scripts/adapt_hypersigma.py`'s docstring, `test_enhanced_model_*` →
`test_simmim_model_*`, the "11 files (git mv)" shorthand, and the new legacy
test's dependence on process-global warning state (it now resets it). The empty
`configs/pretrain/` directory left behind by the moves was removed.

A **third** pass found that the same mask-rate error I had corrected in
CHANGES.md and `docs/PRETRAINING.md` was still sitting in this log's own
Surprise #2 — including a second mistake, "none carries a paper rate", when
`houston_simmim.yaml`'s 0.0/0.75 *is* canon §1's SimMIM-token pair. Surprise #2
above is the corrected text. Its other items are fixed too: two shell-driver
examples still passing `5`-way after the N-way edit, the `houston_simmim_hsi`
comparability note (the two files also differ 10× in `lr` and 4× in
`save_interval`, not only in masking), the same "identical recipe"
claim in the two Houston MAE configs (their lr is 1.5e-4 against
`houston_simmim.yaml`'s 1.5e-5), the "band-only, matching the proven Houston
regime" line in all four band-masking configs that carried it, a dangling
"the team's" and a self-referential schedule comment in `configs/mft/`,
`tests/equivalence/fixtures/README.md` naming today's module for a pre-refactor
run, "enhanced-pretraining" in `lib/pretrain_runner`'s docstring, two runner
docstrings naming pre-rename config basenames, three `configs/mft/` headers
still titled "Spatial" masking, a legacy regime word in `RESULTS.md` prose,
both `(an ``CoFFE``)` instances, a half-edited paragraph in `pretrain.ipynb`,
and the off-by-one counts in CHANGES.md, this log and the manifest note
(**26** configs were renamed, not 27; 26 carry a `paths:` block, the eval config
has none). The notebook-outputs invariant was re-checked after the last notebook
edit and still holds.

A **fourth** pass found that three of those claims had been written before the
corresponding edit was complete — the Houston MAE configs, one of the four
"proven Houston regime" lines, and the second `an ``CoFFE``` — plus a new
inaccuracy of mine in `pretrain.ipynb` ("lengthen the schedule": the overrides
in fact *shorten* it, 2000 epochs against the config's 3000, and double the
batch). All are now actually done, and this paragraph describes the finished
state. It also flagged three pre-existing claims that this phase's renaming had
made newly misleading, now corrected: `scripts/run_eval.sh`'s "paper protocol"
example (the script defaults to `k_query=30` / 600 episodes, not 100 / 1000),
`scripts/run_eval_trento.sh`'s header (its built-in 8 heads / 4 layers /
lambda 1.0 / k_query 19 / projection-on defaults are not the paper
configuration), a stale "the base config uses 8 heads / 4 layers" in
`PROJECT_OVERVIEW.md` (`configs/pretrain/base.yaml` was deleted in phase 3),
and the "0.9 means ~90% of all values are hidden" comment sitting above
`band_mask_ratio: 0.0` in `houston_simmim.yaml`.

A **fifth** pass returned **CONCERNS with nothing blocking** and verified all
eleven fixes present. Its remaining items are handled: `adapt_hypersigma.py`'s
recipe line named two keys the HyperSIGMA pipeline does not have; CHANGES.md and
`coffe_compat.py` said "writers emit canonical only" without disclosing
`aggregate_significance.py`, which writes the significance experiment's own
`group`/`variant` dir-name components (now an explicit exception);
CHANGES.md's "none of the four directories exists" (two of the new fallbacks are
the parent directories the configs write into); `PROJECT_OVERVIEW.md` asserting
one `lr=1.5e-4` for both objectives and "Houston trains 3000 epochs" (D17), plus
two retired regime words two lines below a bullet this phase had canonicalised;
`docs/EVAL_PROTOCOL.md` saying the shell drivers "wrap this" without the caveat
the same diff added to those scripts; `run_eval.sh`'s header claiming the paper
protocol above non-paper episode defaults; and a `docs/PRETRAINING.md` snippet
that called `torch.load` without importing torch and never loaded the state dict.

Two of its observations are worth the gate's attention rather than a fix: the
`models/hypersigma/few_shot.py` basename is the phase-4 skill's, not the
manifest's original `hypersigma_fewshot.py` (the manifest was updated to match —
ratify or reverse), and `.claude/agents/verifier.md` is `.gitignore`d, so the
step-6 edit disclosed above cannot be reviewed from the commit.

### Deviations from the phase-4 skill, stated for the record

- Doc targets are `docs/PRETRAINING.md` / `docs/EVAL_PROTOCOL.md` (the
  manifest's, and the convention of every other file in `docs/`), not the
  skill's lowercase `docs/pretraining.md` / `docs/evaluation.md`.
- The HyperSIGMA evaluator module is `models/hypersigma/few_shot.py` (the
  skill's basename); the manifest's earlier `hypersigma_fewshot.py` target was
  updated to match, since inside package `hypersigma/` the prefix is redundant.
- `tests/test_pretrain_enhanced.py` → `tests/test_pretrain_simmim.py` was added
  to the rename set (not in the phase-1 list) because the stale-vocabulary grep
  covers test filenames.
- No `configs/coffe/*band_token*` file was written, so D19's conditional
  phase-4 obligation ("must carry the rate of the cell it reproduces") did not
  fire. Per-cell reproduction configs are still owed; see the gate.

### Open questions for the gate

1. **D20** — accept as a documented discrepancy (reproduction docs state which
   run produced the mean and which produced the ±), or do something else?
   Nothing was changed.
2. **`--distance-metric` default** — leave at `cosine` (behaviour frozen, as
   now) or flip to `euclidean` (canon §1, but it changes what an unflagged
   invocation computes)? A flip is a one-line change plus a golden re-check.
3. **Per-cell reproduction configs** — should phase 5/8 add
   `configs/coffe/<scene>_simmim_{band,token,band_token}[_hsi].yaml` with each
   cell's exact rates (Houston band+token 0.75/0.75, the rest per D19), so the
   base configs stop being the only entry point?
4. **`docs/presentation/RESULTS.{md,json}`** — regenerate against today's
   `experiments/` at some point, or freeze and mark superseded?
5. **`compile_results.py`'s model label** — should `classify()` split CoFFE from
   the MFT control using the `model_type` the evaluator writes, instead of
   labelling every single-metric run `"CoFFE"`? That changes what a regenerated
   `RESULTS.json` claims, so it was not done here.


### Phase-4 gate addendum — Nikola's five decisions (2026-09-01)

**1. D20 — use the mean's run.** The two Trento cells whose ± was measured on a
different mask rate than their mean keep the published numbers, and their
reproduction configs carry **the mean's rate (band 0.85)**. Confirmed against
the eval JSONs before deciding: `trento_enhanced_spectral_run2` @ 0.85 gives
exactly 90.32 and `..._spectral_spatial_run2` @ 0.85/0.75 gives exactly 92.50,
while the 5-seed ± was cloned from the `_run1` dirs at 0.75, which evaluate to
87.47 and 88.52. Both cell configs state the mismatch in their header.
`PAPER_CANON` §8 gains **D20** with that decision.

**2. `distance_metric` now defaults to `euclidean`.** Cosine stays a selectable
option. This is the refactor's **one deliberate default change**, and it is
behaviour-visible for an unflagged invocation — which is exactly why it needed
the gate. Changed in `scripts/evaluate.py` (CLI, `_DEFAULT_ARGS`, both
`model_config.get(...)` fallbacks), `scripts/evaluate_hypersigma.py`,
`lib/eval_runner.py`'s primary-metric pick, and the `CoFFE` / `MFTOriginal` /
`HyperSIGMAFewShot` constructors. **The equivalence goldens are unchanged**,
which is the proof that no paper run went through the default: every paper run
passes `distance_metric` explicitly. (The canon-reviewer was right to challenge
the first wording of this — not every paper run passes *euclidean*: the Table 3
Houston 11×11 spectral cell records `cosine`, D18. The conclusion holds because
what matters is that the flag is always given, not which value it takes.) Recorded in `PAPER_CANON` §1.

**3. Per-cell reproduction configs — all 30 Table-2 cells now have one.**
`tools/refactor/make_cell_configs.py` parses the cell → run mapping out of
`AUDIT.md` §3 (so it cannot drift from the audit), reads each cell's frozen
`pretrain_config.yaml`, and emits a config whose every recipe value is copied
verbatim — only `model.name` and `pretrain.objective` are rewritten to canon
vocabulary, through `coffe_compat`. 18 CoFFE SimMIM configs are new; the 12 MAE
and MFT configs already matched their cell exactly and only gained a provenance
header (verified: comments and blank lines only). `paths:` is not copied into the 18 generated files. The 12 stamped ones keep the
`paths:` they had (config `paths:` values are DO-NOT-RENAME); none of those
directories exists on disk — the paper's checkpoints are under
`experiments/<run>/checkpoints/` — but a raw CLI run would create them, which
their headers now say.

The tool re-verifies all 30 on every run and **caught a bug it had introduced
itself**: `min_lr` rendered as `1e-06`, which PyYAML reads back as the *string*
`"1e-06"`, not a float — it would have reached the scheduler as one. `_scalar()`
now forces `1.0e-06`, and `render()` round-trips through `yaml.safe_load` and
compares against the run's recipe before anything is written, so a formatting
slip can no longer ship a config that trains something else.

The six `configs/coffe/<scene>_simmim[_hsi].yaml` files are **not** redundant
after this: `scripts/sig_significance_config.py:114,122` uses them as the
significance experiment's base configs. They keep their names and now carry a
ROLE banner saying what they are and pointing at the per-cell configs.

**4. `RESULTS.{md,json}` — explained at the gate, no change made.** See the
answer written up for Nikola; the short version is that the committed file is a
June-04 generation (33 kept results, 26 headline rows) while today's tree yields
858 kept / 41 rows, it is missing the headline Houston cell because
`_TEST_MARKERS` filters the run's directory name (D14), and its HyperSIGMA rows
are the superseded generation D2 records. Regenerating is a phase-8 decision
that also needs D14's filter settled.

**5. `compile_results.py` mislabelled the MFT control — fixed.** `classify()`
now splits on the `model_type` the evaluator writes: `MFTOriginal` →
`"MFT (original)"`, everything else single-metric → `"CoFFE"` (Nikola's note is
right that the MFT-CPEA variant *is* CoFFE — that half was already correct).
Before the fix, 47 MFT-control results were labelled `"CoFFE"`, and in six
(dataset, model, regime, modality) groups an MFT-control run competed with
genuine CoFFE runs to be the cell's representative. The committed
`RESULTS.json` predates those runs, so no published row was affected.

`canon-reviewer` returned **FAIL** on the addendum, for two claims of mine that
overreached, both now corrected:

- **"`paths:` is deliberately not copied"** was true of the 18 generated configs
  and false of the 12 stamped ones, which keep their original `paths:` (config
  `paths:` values are DO-NOT-RENAME). Checked: none of those twelve directories
  exists — the paper's checkpoints live under `experiments/<run>/checkpoints/` —
  so nothing can be clobbered, but a raw CLI run would create them. The claim is
  now scoped, and the twelve headers say where they write.
- **"no paper run is affected, since all of them pass `euclidean` explicitly"**
  — not all of them do: the Table 3 Houston 11×11 spectral cell records
  `cosine` (D18). The conclusion stands (every paper run passes the flag
  *explicitly*, so no default is consulted), but the justification was wrong in
  `PAPER_CANON` §1 itself, at the one place recording the refactor's only
  behaviour change. Reworded there, in CHANGES.md and above.

Its non-blocking items are handled: the generator documented a `--check` flag it
does not have; the header duplicated the modality and, for one cell, the D17
note; `compile_results.py`'s new comment reintroduced a retired name; and three
notebooks still carried the `.get("distance_metric", "cosine")` primary-metric
fallback that `lib/eval_runner.py` had moved. The double duty of the twelve
MAE/MFT configs — they are *also* the significance runner's base configs for the
`enhanced_mae`/`mft_mae`/`mft_spatial` groups (`clone_mask=False`, so the values
used are the cell's) — is now stated in CHANGES.md.

It also found three silent-drop paths in the generator that the round-trip guard
could not catch, all now closed: `assert_no_dropped_keys()` refuses to emit if a
frozen config carries a key outside the emit lists, and `_objective_of()`
replaces the old "band_mask_ratio present ⇒ SimMIM" inference — an MAE run that
omitted `objective` would have been silently relabelled `simmim`, and the guard
would have agreed with itself. It now raises instead of guessing.

The verifier's addendum pass returned **BATTERY: PASS** and added three items,
all handled: `configs/eval/hypersigma_houston.yaml` was the one shipped artifact
still selecting `cosine` (it is the "default HyperSIGMA eval config", and that
key picks which block `lib/eval_runner` records as primary) — now `euclidean`,
with its non-paper episode constants spelled out in the header; CHANGES.md's
exclusion table gained `.claude/` and the deliberate legacy-vocabulary G1 test;
and its zero-tolerance measurement of the goldens corrected the "IDENTICAL"
shorthand used earlier in this log (G2–G5 bit-identical, G1 within ≤ 5e-09 of
its 8-decimal goldens — reduction-order noise present on untouched goldens too).

It also confirmed independently, without using the generator, that all 30 cell
configs match their frozen runs: 45 deltas in total, every one of them in the
allowed set (23 × `model.name`, 22 × `pretrain.objective`). The twelve
MUUFL/Trento SimMIM cell configs omit the frozen `data.hsi_channels` /
`data.aux_channels`; that is inert (`scripts/pretrain.py:214-220` derives both
from the dataset and never reads them from the config) and Houston's frozen
configs never carried them.

Verification for this addendum: 120 tests pass, equivalence goldens unchanged
(the default flip is invisible to them by construction), all 30 cell configs
re-verified value-by-value against their frozen runs, and the six base configs
plus the twelve stamped cell configs confirmed to differ from the phase-4 commit
in comments and blank lines only.

---

## Phase 5 — restructure into the `coffe` package + thin CLIs + `results/`

Seven top-level packages became one installable package; the four fat scripts
became argparse-only entry points over package modules; the paper-provenance
JSONs left the runtime tree. **No computed number moved:** every Python move was
verbatim (proved below), and the equivalence goldens' numeric fields are
byte-identical.

### Layout approved at the gate

`coffe/{models,pretrain,data,eval,runners,utils}` + `coffe/compat.py`,
`scripts/` (CLIs) with `scripts/reproduce/` (experiment drivers) and
`scripts/reports/` (provenance builders), `results/` (paper JSONs),
`experiments/` runtime-only. Nikola also chose: ship `third_party` as an
installed package (no vendored file edited), and take all three deferred
renames (`significance_report copy.json`, the `.pth.fixture` fixtures, the
stray `configs/eval/` config).

Deliberate deviation from the phase skill's default layout: it proposed writing
new `scripts/reproduce/table2_*.sh` / `table23_*.sh` wrappers. No new pipeline
code was invented in a restructure phase — the existing drivers moved into
`scripts/reproduce/` with a README mapping each to its table cells instead.

### What moved

| Before | After |
|---|---|
| `models/`, `pretrain/`, `data/{datasets,samplers}/`, `utils/` | `coffe/<same>/` |
| `lib/` | `coffe/runners/` |
| `trainers/pretrain_trainer.py` | `coffe/pretrain/trainer.py` |
| `coffe_compat.py` | `coffe/compat.py` |
| body of `scripts/evaluate.py` | `coffe/eval/episodic.py` |
| body of `scripts/evaluate_hypersigma.py` | `coffe/eval/hypersigma.py` |
| body of `scripts/pretrain.py` | `coffe/pretrain/loop.py` |
| body of `scripts/adapt_hypersigma.py` | `coffe/pretrain/hypersigma_adapt.py` |
| 16 experiment drivers / 7 report builders | `scripts/reproduce/`, `scripts/reports/` |
| 8 `experiments/*.json` | `results/` (+ `results/README.md`) |
| `experiments/significance_report copy.json` | `results/significance_report_enhanced_v1.json` |
| `configs/eval/hypersigma_houston.yaml` | `configs/hypersigma/houston_eval.yaml` |
| `tests/equivalence/fixtures/*.pth.fixture` | `…/fixtures/*.pth` |

162 import lines across 54 files were rewritten to absolute `coffe.<subpkg>`
form; every `sys.path` bootstrap inside the package is gone (the scripts keep
theirs, since `scripts/reproduce/*` import `scripts.reproduce.sig_significance_config`
when run as files).

### The four script splits are verbatim

Each split was verified mechanically against `HEAD`: the `if __name__ ==
"__main__":` block is **byte-identical** in all four new scripts, and the moved
body differs from the old script only in (a) import lines, (b) the removed
`sys.path` bootstrap and (c) path strings inside docstrings and comments
(`coffe_compat` → `coffe.compat`, `data/datasets/registry.py` →
`coffe/data/datasets/registry.py`). No executable line was reordered, rewritten
or "fixed while I was in there".

### Surprises found (reported, not fixed)

1. **`create_pretrain_dataloaders` was quietly broken.**
   `trainers/pretrain_trainer.py:472` did `from .masked_modeling import …`,
   i.e. `trainers.masked_modeling`, which never existed — any call would have
   raised `ModuleNotFoundError`. Nothing calls it (the only reference was the
   `trainers/__init__` re-export), so no paper run touched it. Moving the file
   into `coffe/pretrain/` makes that relative import resolve, so the function
   goes from latently broken to working **as a side effect of the move** —
   not a code change, and not something any caller can observe today.
2. **Legacy top-level imports no longer resolve.** `from models import
   MFTCPEACosine` used to work through the phase-4 alias layer; the package
   `models` no longer exists, so it is now `from coffe.models import
   MFTCPEACosine`. The legacy *class names* still resolve with their
   `DeprecationWarning` — only the module prefix changed.
   `tests/test_compat.py`'s parametrisation was updated accordingly, and
   `coffe/compat.py`'s `LEGACY_CLASSES` now names `coffe.models…`.
3. **`--output results/eval/…` in `configs/hypersigma/houston_eval.yaml`.** That
   comment's example output path now collides with the new tracked `results/`.
   Retargeted to `experiments/<run>/evaluations/<eval>/results.json`, the
   convention the runner actually uses. Comment only; no config value changed.
4. **Two `parents[1]`/`parent.parent` classes of bug** would have shipped
   silently: every script that moved one level deeper computed the wrong repo
   root, and the five `run_*.sh` wrappers `cd`'d to `scripts/` instead of the
   repo root. Both fixed; caught by the from-another-cwd CLI smoke, which is why
   that check earns its place. A third instance (`DATASETS` / `HyperSIGMAFewShot`
   referenced by `scripts/evaluate_hypersigma.py`'s parser but left behind in the
   package module) was caught the same way.

### Golden files: three metadata strings changed

The goldens' numeric content is untouched. Three provenance strings inside them
name modules or paths that moved, and were updated so the record stays true:
`g1_pretrain_loss.json`'s description (`scripts.pretrain.run_pretrain` →
`coffe.pretrain.loop.run_pretrain`), `g3_episodic_eval.json`'s description
(`scripts.evaluate.run_evaluation` → `coffe.eval.episodic.run_evaluation`) and
its two `fixtures[*].path` fields (`.pth.fixture` → `.pth`). No test reads any
of the three. Diffable in one line: `git diff HEAD -- tests/equivalence/golden/`.

### Packaging

`pyproject.toml` gained `[project]` (name `coffe`, version `0.9.0`, MIT,
`requires-python >=3.9`, dependencies copied unchanged from `requirements.txt`
— pruning is phase 6) and `[tool.setuptools.packages.find] include =
["coffe*", "third_party*"]`. `pytest`'s `addopts` now measures `--cov=coffe`.
`setuptools-scm` was dropped from the build requirements at the same time: the
version is static and no `[tool.setuptools_scm]` section exists, so it was never
doing anything. `.gitignore` lost the venv-layout `lib/`/`lib64/`/`parts/`/`eggs/`
block and its
`!/lib/**` counter-rules (**D5 closed**: the trap is gone, not neutralised),
lost the now-unneeded `!/experiments/*.json` exception, and gained
`!/tests/equivalence/fixtures/*.pth` plus an explicit ignore for
`tests/equivalence/golden/real_local.json`. The removal is slightly wider than
D5 strictly required — `lib64/`, `parts/`, `eggs/`, `.eggs/` went with `lib/`,
as none of them names anything this repo builds.

`tests/equivalence/make_golden.py` emits the two golden `description` strings,
so it was updated in step with them; otherwise a regeneration would silently
revert those provenance lines. `run_eval.sh` / `run_eval_trento.sh` had their
default output directory moved from `results/eval/…` to `experiments/eval/…`,
since `results/` is now tracked paper provenance — disclosed in CHANGES.md
under "One default moved".

### Ledger and closure

`tools/refactor/build_manifest.py` gained `"move": 5` in `EXECUTED_IN_PHASE`,
path rules for the new tree, and phase-5 move records; the four CLI records
became `keep` with a note that they supersede phase-4's never-approved
`coffe/cli/<x>.py` proposals. 340 records, **the 42 approved ones unchanged**.
`closure.py`'s root set was re-pointed at both the CLIs and the package modules
that now hold their bodies. Normalised to pre-move names, the paper closure is
**74** against phase 4's 73: `coffe/__init__.py` and `coffe/eval/__init__.py`
entered, `trainers/__init__.py` left with the dissolved package. Nothing else
entered or left the paper's reachable set.

### Incident: four frozen artifacts were overwritten, then restored

While smoke-testing `--help` on every script, four report builders **ran their
full pipelines** and rewrote their outputs against today's `experiments/` tree:
`results/_report_raw.json`, `results/hypersigma_native_sem_pca100_report.json`,
`results/gathered_results.json` and `docs/presentation/RESULTS.json`. The last
one had its curated Table-2 entries replaced wholesale (33 kept results → 858),
which is precisely what PAPER_CANON §6 forbids. `canon-reviewer` caught it as a
FAIL; all four were restored from `HEAD` and re-verified byte-identical
(`sha256` per file), so the committed tree carries the frozen artifacts
unchanged. No paper number is affected.

**The underlying hazard is pre-existing and worth fixing in phase 6:** four
scripts take no arguments at all —
`scripts/compile_results.py`, `scripts/reports/build_native_pca100_report.py`,
`scripts/reports/gather_native_pca100_raw.py`,
`scripts/reports/gather_requested_results.py` (the last has no `__main__` guard
either). Passing them *any* flag, `--help` included, executes the pipeline and
overwrites a frozen artifact. They are now excluded from CLI smoke by design
(byte-compile only), and the recommendation is: give each an `argparse` front
end with an explicit `--out`, and a `--force` before any overwrite of a
`results/` path.

### Two defects the reviews caught after the first pass

1. **Every notebook was broken.** All seven bootstrap cells locate the repo root
   with `while not (REPO / "lib" / "experiments.py").exists()`. With `lib/`
   gone, each would have walked to `/` and raised `RuntimeError` in its first
   cell — the README's primary workflow, dead. Fixed in all 11 such loops
   across the seven notebooks (several carry two) to
   `coffe/runners/experiments.py`, and verified by executing each bootstrap
   from `notebooks/` and from the repo root.
2. **The four thin CLIs required an install; the other 13 entry points did
   not.** Dropping their `sys.path` bootstrap made
   `python scripts/evaluate.py` fail with `ModuleNotFoundError: coffe` in any
   tree that had not been `pip install -e .`-ed — including Nikola's `.venv`.
   The bootstrap is back in all four (a script-level `sys.path.insert`, matching
   the rest of `scripts/`; the *package* still has none), so every entry point
   behaves the same way with or without an install. README's install section
   says so explicitly now.

### Verification

- `pytest -q -m "not gpu and not data"`: **120 passed** (same count as phase 4).
- Equivalence: G1–G5 green; goldens' numeric fields byte-identical.
- G4 checkpoint fixtures load through the moved key-mapping code under their
  new `.pth` names.
- `pip install -e .` in a clean scratch venv from `pyproject.toml` alone
  (`--no-deps`, so the ~2 GB torch download is skipped): `import coffe`,
  `coffe.{models,data,eval,pretrain,runners,utils}` and
  `third_party.HyperSIGMA.ImageClassification.model` all resolve.
- CLI `--help` smoke on all **17 argparse entry points**, from the repo root and
  from `/tmp`, in the project `.venv` (no install) **and** in the scratch venv
  (installed): 17/17 both ways. The five argument-less scripts are byte-compiled
  only, never executed — see the incident above.
- Stale grep: no `from models`/`from lib`/`import utils`/`coffe_compat` imports
  remain outside `docs/refactor/` (frozen history) and `CHANGES.md` (which
  documents the retired forms on purpose); every relative markdown link in the
  release docs resolves; the equivalence harness's own docstrings, README and
  line-number citations were re-anchored to `coffe/eval/episodic.py`.
- `verifier` battery: pytest 120 passed, equivalence **IDENTICAL** (a recursive
  key-by-key comparison against `HEAD` found **0 numeric diffs**; exactly four
  changed lines, all provenance strings), G4 6/6, imports 11/11.
- `canon-reviewer`: returned **FAIL** on the first pass — the four regenerated
  artifacts and the dead notebook bootstrap, both fixed above — plus 12
  non-blocking items, all applied: harness/README stale module paths,
  CHANGES.md's exclusion table naming pre-move files, two live-code citations in
  PAPER_CANON (`lib/eval_runner.py:209`, `scripts/build_experiment_metadata.py:112-115`),
  the `run_eval.sh` output-dir collision, prose corrupted by the mechanical path
  rewrite ("the verbatim **coffe/pretrain**/adapt config"), the epoch-950/975
  over-simplification in the two new READMEs (one cell is epoch 800; all six MFT
  cells are 950), an unsupported "21 of 30" count, `pyproject`'s
  `setuptools>=45` against the SPDX `license` string (now `>=77`), and three
  cosmetics.

### Open questions for the gate

1. **`pip install -e .` was verified with `--no-deps`** in a scratch venv (the
   dependency set is unchanged from `requirements.txt`, and a full resolve would
   re-download torch). Nikola's `.venv` was not modified — installing the
   package editable there is his call.
2. **`requires-python = ">=3.9"`** was chosen to match the existing
   `black`/badge targets; the dev `.venv` is 3.12. Confirm or narrow in phase 6
   with the dependency prune.
3. **Config coverage gaps for Table 3** (skill step 6 asks for the list, not
   for invented settings). Table 2: all 30 cells have a per-cell config. Table
   3: the 3 SEM-only pad cells and the 3 joint+SEM cells have configs, the 3
   11×11 spatial cells have a script path
   (`run_hypersigma_spatial_pca100.py`, which overrides `adapt_mode` in code
   over the joint_sem config); **12 frozen 64×64 cells and the 3 11×11 spectral
   cells have neither** — they were driven from
   `notebooks/evaluate_hypersigma_native.ipynb` and
   `notebooks/evaluate_hypersigma.ipynb` respectively. The frozen twelve
   involve no adaptation at all (eval-flag variations over the released
   checkpoints), so a config would only carry eval settings. Recorded in
   `scripts/reproduce/README.md`; reconstructing configs for them is a phase-8
   decision, not something phase 5 should guess.
4. **`docs/presentation/RESULTS.{md,json}`** still carry phase-4's open
   question (regenerate or mark superseded). Phase 5 leaves both byte-identical
   to phase 4 — see the incident above for why that sentence needed checking.
5. Surprise 1 above (`create_pretrain_dataloaders`) — leave the now-working
   helper as it is, or delete it as dead code in phase 6?

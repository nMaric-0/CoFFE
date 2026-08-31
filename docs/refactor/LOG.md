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

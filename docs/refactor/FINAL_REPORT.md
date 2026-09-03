# FINAL_REPORT.md — the CoFFE release cleanup, phases 0–8

What this cleanup did, what it deliberately did not do, and what is still open.
`LOG.md` is the per-phase narrative with the evidence; `AUDIT.md` is the
paper↔code trace; `manifest.json` is the file-by-file ledger. This file is the
summary a reader should be able to stop at.

- **Branch:** `refactor/cleanup`, 2026-08-31 → 2026-09-03: 19 commits through
  phase 7, plus phase 8's three (two for the release documentation, one for the
  gate decisions).
- **Baseline:** `9541085` "Final experiments" (2026-07-30), tagged
  **`pre-refactor`**.
- **The one hard rule:** no refactor commit may change a computed number. The
  arbiter is `tests/equivalence/`, and it stayed green on every commit that
  touched code — G2–G5 bit-exact at zero tolerance, G1's loss trajectories
  within the declared `rtol=1e-6` (residual ≤ 4.6e-09, BLAS reduction-order
  noise from an unpinned thread count; see `LOG.md`'s phase-8 finding 0). Evaluator *defaults* changed at three gates — phase
  4's `distance_metric`, phase 7's seven `_DEFAULT_ARGS` values, and phase 8's
  HyperSIGMA trio plus the `use_projection` de-inheritance — every one signed
  off, none reachable on the paper path (`CHANGES.md`, "What deliberately did
  **not** change").

## Phase by phase

| Phase | What happened | Verification |
|---|---|---|
| **0** — baseline | Inventoried the tree (108 Python files / 19,889 LOC), recorded the environment, ran the suite as-found (**62 passed**), tagged `pre-refactor`. No code touched. | `docs/refactor/BASELINE.md`, `baseline_pytest.txt`, `ENV.md`, `inventory.json` |
| **1** — audit | Import-closure trace over the whole tree; verdicts on D1–D12 with file:line evidence; a 340-file manifest (24 delete / 18 archive / 34 move / 39 rename / 225 keep). Resolved **D3**: the CoFFE eval feature is `z = mean_j(patch_emb_j) + 0.5·cls_emb`, not "patch tokens pooled" — measured that removing λ costs 1.70 pp, so removal is barred by the freeze; Nikola chose to keep it and document it honestly. | `AUDIT.md`, `closure.json`, `manifest.json` |
| **2** — harness | Built `tests/equivalence/`: goldens **G1–G5** — pretraining loss (G1), encoder forward (G2), episodic eval (G3), the two committed pre-refactor checkpoint fixtures (G4), masking (G5) — driven through the *real* entry points over three synthetic scenes written in the real on-disk layout. Pins the **live** eval path, not the dead `forward_episode`. | 32 tests, goldens generated pre-prune on `4d42f64` |
| **3** — prune | Executed the approved manifest set: 24 files deleted, 18 exploratory files moved to an untracked `archive/`. 20,985 lines removed against 29 added. The remaining 23 deletes were re-approved in four groups at the gate first, because phase 1 had left them as proposals. | equivalence IDENTICAL; battery green |
| **4** — rename | The paper's vocabulary throughout: `MFT-CPEA*` → **CoFFE**, "Cosine" → **Euclidean NCM**, `enhanced` → `simmim`, `spectral`/`spatial`/`both` → `simmim_band`/`simmim_token`/`simmim_band_token`, `MFTOriginalCosine` → `MFTOriginal`, `HyperSIGMACosine` → `HyperSIGMAFewShot`. Legacy names still read via `coffe/compat.py` (one `DeprecationWarning` per value). state_dict attribute names untouched, so checkpoints load unchanged. Gate addendum: `euclidean` became the default, and every Table 2 cell got a per-cell reproduction config. | equivalence IDENTICAL; pre-rename checkpoint fixture still loads |
| **5** — restructure | One installable package: `coffe/{models,pretrain,data,eval,runners,utils}`, `scripts/` reduced to thin CLIs, paper artifacts moved out of the runtime tree into `results/`. Closed **D5** — `.gitignore` no longer shadows first-party source. | equivalence IDENTICAL; `pip install -e .` in a clean venv |
| **6** — quality | black+isort+flake8 → **ruff**; **mypy** adopted lenient; dependencies pruned to the measured import set (D6); public API declared; docstrings brought up to release standard. Gate addendum: `--out`/`--force` on the four report scripts that used to overwrite committed artifacts on any invocation. | ruff + ruff format + mypy clean; equivalence IDENTICAL |
| **7** — tests | `tests/{unit,integration,equivalence}` with `gpu`/`data`/`slow` markers: **527 collected, 521 passed, 6 skipped** under `-m "not gpu and not data"`; coverage of `coffe/` 52 % → 69 %; CI added (ruff + mypy + tests on 3.11/3.12, CPU-only torch). Gate: eight questions answered, four applied — numeric checkpoint sort, one evaluation entry point per route (`scripts/evaluate_mft.py` is new), six stale evaluator defaults aligned, plus a real-data GPU smoke. | equivalence IDENTICAL 33/33 throughout |
| **8** — release | README rewritten to release standard from `PAPER_CANON.md` §§2–6; `docs/hypersigma.md` and `CITATION.cff` new; `pretraining.md`/`evaluation.md` gained the reproduction sections they had promised; `docs/presentation/` marked superseded; `CHANGES.md` finalised with an archaeology section; the three carried-over phase-7 obligations applied. Every README command executed, most of them in a fresh clone with a fresh venv. | 8-check fresh-clone gate: 7 PASS, plus one FAIL that was an **environment** finding rather than a behaviour change — resolved at the gate (item 7), after which a fresh clone skips the goldens with a reason instead of failing. Equivalence 33/33 on the reference environment throughout. |

## Metrics, before → after

First-party source excludes `tests/` and `tools/refactor/` (one-shot tooling
kept for provenance) and never included the vendored `third_party/`.

| | phase 0 | now |
|---|---|---|
| first-party source files | 100 | **81** |
| first-party source LOC | 18,895 | **16,646** |
| test files / LOC | 8 / 994 | **30 / 6,856** |
| tests collected (`-m "not gpu and not data"`) | 62 (whole suite) | **527** |
| line coverage of the package | 50 % (`models`+`data`) | **69 %** (`coffe/`) |
| equivalence fingerprints | — | **33**, IDENTICAL |
| top-level packages | 8 (`models`, `data`, `pretrain`, `lib`, `utils`, `trainers`, `scripts`, `tests`) | **1** (`coffe`) + `scripts/`, `tests/` |
| declared dependencies | 23 (`requirements.txt` kitchen sink) | **12** (measured import set) |
| lint / type tooling | black + isort + flake8, none enforced | **ruff + mypy**, enforced in CI |

Two paper-path modules went from 0 % to covered in phase 7:
`coffe/eval/hypersigma.py` (74 %), the module every Table 3 cell came through,
and `coffe/pretrain/hypersigma_adapt.py` (81 %).

## What the cleanup found that the paper does not say

Every one of these was documented, none was "fixed" (hard rule: behaviour is
frozen; PAPER_CANON §8 carries the full record with evidence).

1. **D3** — the CoFFE eval feature adds a weighted class-agnostic token,
   `z = mean_j(patch_emb_j) + 0.5·cls_emb`; the paper describes pooling patch
   tokens. Removing it moves OA by −1.70 pp, so it is load-bearing.
2. **D17** — the evaluated checkpoint is mid-schedule (950 Houston /
   975 Trento+MUUFL / 800 for one cell), and phase 7 traced *why*: a
   lexicographic checkpoint sort, not a model-selection decision. The sort is
   numeric now; reproducing a cell requires passing its epoch, which the
   per-cell configs instruct.
3. **D18** — the protocol constants must be read per table: Table 2 ran 1000
   episodes, Table 3 ran 2000, and one Table 3 cell used `k_query=30`.
4. **D19 / D20** — the Houston band+token cell used `(0.75, 0.75)`, not the
   paper's stated `(0.85, 0.75)`; and for two Trento cells the ± was measured on
   a different mask rate than the mean.
5. **D14** — the headline Houston run (75.30) sits in a directory the results
   compiler filters out as a test run, and its `seed52` name is a misnomer (it
   ran seed 42).
6. **D13** — the committed `configs/` did not reproduce the paper; the frozen
   `experiments/*/pretrain_config.yaml` did. Phase 4 regenerated per-cell
   configs *from* those.
7. **Phase-7 S3** — the paper's "border-padded at edges" happens upstream, in
   the MFT data preparation; this repo's own raw-image path drops border pixels
   and is on no paper path (PAPER_CANON §5 footnote).

## Unresolved / open items

**Release blanks Nikola has to fill (all tagged `TODO(release)`):**

1. ~~Paper venue~~ **filled after the release push attempt**: the venue was
   recorded in a commit that existed only on `origin/main`
   (`9196879 "Fix README"`, 2026-07-30) and would have been discarded by the
   force-push — MACLEAN workshop, ECML PKDD 2026, "to appear". `README.md` and
   `CITATION.cff` now carry it, together with that commit's attributions
   (HyperSIGMA: Wang et al., IEEE TPAMI 2025; the MFT control: Roy et al., IEEE
   TGRS 2023; the scene providers). Still open there: pages, DOI and the
   proceedings URL, once published.
2. The **CoFFE / MFT checkpoint hosting link** — the paper promises public
   checkpoints and there is nowhere to point yet.
3. The funding acknowledgement text.

**Decided at the phase-8 gate (2026-09-03) and applied.** Nikola's answers to
the six questions the gate raised, and what each turned into:

4. **The D17 addendum is accepted** — `PAPER_CANON.md` §8 D17 now records that
   the evaluated epochs (950/975/800) are what a **string** sort of checkpoint
   filenames returned, not a mid-training model-selection decision, and says
   release documentation must not present it as one. `README.md`,
   `docs/evaluation.md`, `docs/pretraining.md`, `results/README.md` and
   `scripts/reproduce/README.md` carry that provenance.
5. **The HyperSIGMA evaluator's defaults are aligned** with the table it serves:
   `k_query` 100, 2000 episodes, `split "all"` (were 30 / 600 / `"test"`, which
   reproduced no published cell). Pinned by
   `tests/unit/test_hypersigma_contracts.py`. No published number moves — every
   paper run passes these explicitly.
6. **`use_projection` is no longer inherited** by `coffe/runners/eval_runner.py`,
   so a bare `run_evaluation(...)` evaluates with the head **off**, which is the
   protocol. The `proj_*` shaping keys are still inherited for a caller who asks
   for the head. Pinned by `tests/unit/test_eval_runner.py`.
7. **The environment the goldens need is now shippable and CI-safe.**
   `constraints/verification.txt` pins `torch==2.11.0`, `torchvision==0.26.0`
   and `numpy==2.4.4`; `tests/equivalence/conftest.py` compares the running
   torch/numpy **release** and the thread count against that reference and
   **skips the package with the difference in the message** instead of
   reporting float drift as a behaviour change; and `.github/workflows/ci.yml`
   installs through the constraints file and asserts that the package skipped
   *for that reason*. Note the workflow itself has **still never run** — it
   triggers on push and the branch has not been pushed — so its behaviour is
   reasoned from the local measurements, not observed. Locally the inverse
   guard applies: `COFFE_EQUIVALENCE_STRICT=1` turns the environment skip into
   an error, and the `verifier` battery sets it — so a skip can never be
   mistaken for a green freeze on the reference machine. The measurements behind it, all in fresh clones: today's
   resolved versions 19/33 fail; torch pinned alone, 2 fail on a torchvision
   build mismatch; the full pin, 33 pass; the same pin on **CPU-only** wheels,
   33 pass (so the `+cpu` / `+cu128` suffix is irrelevant, the release is not);
   and at `OMP_NUM_THREADS` 4 / 2 / 1, 31 / 31 / 26 pass — **the thread count
   matters as much as the version**, which is why a 2-core CI runner can never
   reproduce them and why the guard exists.
8. **The HSI-only error now names the flag that fixes it.** A band-count
   mismatch that is exactly the scene's aux channels says the checkpoint was
   pretrained HSI-only and to re-run with `--no-aux` (or, in the mirror case, to
   drop it), instead of sending the reader to `--dataset`. Pinned end-to-end by
   `tests/integration/test_cli.py`, which also checks the flag it names works.
   `--no-aux` is documented in the README, `docs/evaluation.md` and
   `docs/pretraining.md`.
9. **`CLAUDE.md` and `WORKFLOW.md` are untracked** (still on disk, gitignored):
   they describe the cleanup process, not the release — the same reason
   `SPLIT.md` was deleted in phase 3.
10. **The route docs took their release names**: `docs/pretraining.md`,
    `docs/evaluation.md`, `docs/hypersigma.md`.

**Still open:**

11. **D14's compiler filter.** `scripts/compile_results.py` still excludes the
    headline Houston run as a scratch run. Changing the filter needs sign-off.
12. **Table 3 reproduction gaps.** The twelve 64×64 frozen cells and the three
    11×11 spectral cells have no committed driver (the frozen ones are
    reproducible from CLI flags alone — the README gives the command; the
    spectral ones' adaptation settings were never reconstructed, deliberately).
    Trento's and MUUFL's 11×11 joint+SEM configs lack their runs' launch-time
    overrides, which live in those runs' `pretrain_overrides.yaml`. No
    `backbone_native` adaptation was ever completed for Trento or MUUFL — a gap
    in the paper's sweep, not a missing file.
13. **The 20-epoch observation, and a discrepancy inside it.** Phase 7's smoke
    reached OA **74.43 ± 0.66** on Houston at 20 of 1500 epochs — ~99 % of the
    headline 75.30, in 1.3 % of the schedule — and flagged that as worth a
    second look. Phase 8 re-ran a 20-epoch smoke to put executable commands in
    the README and measured **71.15 ± 0.67** from
    `configs/coffe/houston_simmim_token.yaml` with the head off. The two runs
    state the same settings, phase 7's exact invocation was not recorded, and
    the 3.3 pp difference is **unattributed**; one candidate was tested and
    ruled out (the 5-seed runner's base config gives 56.65 ± 0.96 at 20
    epochs). Phase 7's finding therefore stands as flagged — a truncated
    schedule reaching most of the headline OA would be a claim the paper does
    not make — and needs a real run, not a smoke. **Not investigated**: that is
    new science, not cleanup.
14. **A literal-IDENTICAL harness would need a thread pin and a re-baseline.**
    On the reference environment G2–G5 are bit-exact at zero tolerance while
    G1's loss trajectories sit ≤ 4.6e-09 off the goldens — stable across
    repeats, and traceable to the unpinned thread count. Making them bit-exact
    means `torch.set_num_threads(1)` plus regenerating the goldens, i.e.
    re-baselining the freeze, which must not be done casually: the goldens'
    value is that they were captured from **pre-refactor** code.

**Accepted, recorded, not open:**

15. mypy `ignore_errors` on 16 modules (175 findings, all typing friction — the
    other 39 modules are clean and must stay so), and
    `coffe/utils/visualization.py` at 0 % coverage (it produces figures, not
    numbers; every significance eval ran with `no_plots: True`).
16. `docs/presentation/` stays in the tree with SUPERSEDED banners — the
    manifest's verdict is `keep`, and `RESULTS.json` is left byte-for-byte as
    the compiler wrote it.
17. Legacy vocabulary survives exactly where `CHANGES.md`'s "Where the retired
    names still appear" table says it does: the alias table, the rename ledger,
    on-disk-name literals, stored notebook outputs, and the documents about the
    retirement.
18. `scripts/reproduce/run_eval{,_trento}.sh` and
    `configs/hypersigma/houston_eval.yaml` carry non-paper constants; each says
    so in its own header. The *CLIs* are all aligned now — these are committed
    config/driver files whose values are on the audit's DO-NOT-RENAME footing,
    and changing them would change what a config-driven run computes.

## If you are picking this up cold

- `PAPER_CANON.md` first — names, protocol constants, published tables, and §8's
  known paper↔code gaps.
- `README.md` for how to install, get data, and reproduce a table.
- `CHANGES.md` for what a name used to be and what still accepts the old one.
- `tests/equivalence/README.md` before touching anything numeric.

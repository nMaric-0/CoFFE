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

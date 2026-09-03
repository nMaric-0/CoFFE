# BASELINE.md — the pre-refactor "before" picture (phase 0)

| Field | Value |
|---|---|
| Date | 2026-08-31 |
| Baseline commit | `95410851e47bb0f2171fff9206d19eb46e26b636` — "Final experiments" (2026-07-30) |
| Baseline branch | `main` |
| Tag | `pre-refactor` |
| Work branch | `refactor/cleanup` |
| Environment | see `ENV.md` (repo-local uv venv, Python 3.12.3, torch 2.11.0+cu128) |

No code was changed in phase 0. The only additions are `docs/refactor/*` and
`tools/refactor/inventory.py`.

---

## 1. Baseline test run

Command (from the repo root, venv active):

```bash
pytest -q --tb=short 2>&1 | tee docs/refactor/baseline_pytest.txt
```

Coverage plugins **were** available (`pytest-cov` 7.1.0), so the pyproject
`addopts = "-v --cov=models --cov=data --cov-report=term-missing"` ran as
configured; `-p no:cov` was **not** needed.

### Result

```
62 passed, 2 warnings in 86.79s
```

**0 failures, 0 errors, 0 skips.** There is no failure list to carry forward —
the baseline suite is green.

| Test file | Tests |
|---|---|
| `tests/test_data.py` | 1 |
| `tests/test_hypersigma_native_shapes.py` | 25 |
| `tests/test_hypersigma_shapes.py` | 4 |
| `tests/test_mft_original_shapes.py` | 13 |
| `tests/test_models.py` | 3 |
| `tests/test_pretrain_enhanced.py` | 4 |
| `tests/test_spatial_weights.py` | 12 |
| **Total** | **62** |

Warnings (both benign, third-party):

1. `timm.models.layers` deprecated-import `FutureWarning`.
2. `torch.meshgrid` missing-`indexing` `UserWarning`, from
   `test_native_spatial_forward[144-upscale]`.

Baseline coverage of the two measured packages: **50 %** of 2168 statements
(`models` + `data`). Notable zero/near-zero areas, recorded because phase 3
prune decisions will lean on them: `data/transforms/augmentations.py` 0 %,
`data/samplers/patched_episode_sampler.py` 13 %, `models/backbones/*` 16–17 %,
`models/wrappers/pretrain_wrapper.py` 22 %. Low coverage is **not** evidence of
dead code on its own — the import-closure trace in phase 1 is.

Marker note for phase 2: `pyproject.toml` declares **no** `markers` section, so
the `gpu` / `data` markers required by PAPER_CANON §7.5 and the verification
battery do not exist yet and must be registered.

---

## 2. Inventory

`tools/refactor/inventory.py` (stdlib + `ast` only) → `docs/refactor/inventory.json`.
Excluded: `.git`, `.venv`, `third_party/`, caches, and per-run subtrees under
`experiments/*/` (all of which are symlinks to the archive).

| Metric | Value |
|---|---|
| Python files | 108 |
| Python LOC | 19,889 |
| Top-level defs | 260 |
| Top-level classes | 82 |
| Parse errors | none |
| Non-Python files | 88 (`.yaml` 28, `.md` 20, `.ipynb` 16, `.json` 13, `.sh` 11) |

Python files / LOC by top-level directory:

| Directory | Files | LOC |
|---|---|---|
| `scripts/` | 39 | 8,053 |
| `models/` | 23 | 4,752 |
| `data/` | 13 | 1,374 |
| `pretrain/` | 8 | 2,090 |
| `tests/` | 8 | 994 |
| `utils/` | 8 | 843 |
| `lib/` | 5 | 995 |
| `trainers/` | 2 | 528 |
| `tools/` | 1 | 228 |
| `setup.py` | 1 | 32 |

`scripts/` is 40 % of the file count and 40 % of the LOC — the single largest
surface, and the one PAPER_CANON §8 D9 flags as mostly exploratory.

---

## 3. Known-junk snapshot (for the record only — nothing was deleted)

### 3.1 Tracked files matching `* copy*` / `*_run2*`

Verbatim, from `git ls-files`:

```
experiments/significance_report copy.json
notebooks/evaluate_hypersigma copy 2.ipynb
notebooks/evaluate_hypersigma copy 3.ipynb
notebooks/evaluate_hypersigma copy.ipynb
notebooks/evaluate_hypersigma_pca100 copy 3.ipynb
notebooks/evaluate_hypersigma_pca100 copy.ipynb
notebooks/pretrain copy 2.ipynb
notebooks/pretrain copy 3.ipynb
notebooks/pretrain copy.ipynb
notebooks/pretrain_run2.ipynb
```

That is 9 notebook duplicates (PAPER_CANON §8 D8 says 7 `* copy*` notebooks —
the true count is **8** `* copy*` plus `pretrain_run2.ipynb`) and one JSON
duplicate. D8 to be refined in phase 1.

### 3.2 Untracked `*_run2*` paths on disk (symlinked experiment runs)

```
experiments/houston_enhanced_spatial_run2
experiments/houston_enhanced_spectral_run2
experiments/houston_enhanced_test_run2
experiments/hypersigma_adapt_houston_spatial_only_run2
experiments/hypersigma_baseline_spat_only_run2
experiments/hypersigma_houston_spatial_only_run2
experiments/muufl_enhanced_spectral_spatial_run2
experiments/trento_enhanced_spatial_run2
experiments/trento_enhanced_spectral_run2
experiments/trento_enhanced_spectral_spatial_run2
```

These are **frozen artifacts** (symlinks into `/ceph/home/nmaric/CoFFE`), not
source. They fall under PAPER_CANON §7.3 (DO-NOT-RENAME on-disk names).

### 3.3 `.gitignore` `lib/` trap (PAPER_CANON §8 D5)

`.gitignore:16` still contains the packaging-boilerplate rule:

```
lib/
```

**However, D5 as written is already neutralised at baseline.** Lines 78–84 of
the committed `.gitignore` add explicit negations:

```
# The repo-root `lib/` is a source package and must not be matched by the
# venv-related `lib/` / `lib64/` rules above.
!/lib/
!/lib/**
# ...but keep __pycache__ ignored everywhere under lib/.
/lib/**/__pycache__/
/lib/__pycache__/
```

Verified: `git check-ignore -v lib`, `lib/eval_runner.py`, `lib/pretrain_runner.py`
all report **not ignored**, and all five `lib/*.py` files are tracked
(`lib/__init__.py`, `adapt_runner.py`, `eval_runner.py`, `experiments.py`,
`pretrain_runner.py`). The remaining work for D5 is cosmetic (drop the
now-shadowed bare `lib/` rule) and belongs to phase 5 restructure. Recorded so
phase 1 does not re-litigate it as an active bug.

---

## 4. Phase-0 discrepancies found (reported, not fixed — hard rule 7)

- **E1 — `CLAUDE.md` environment facts are wrong.** It says "conda env
  (`coffe`)"; there is no conda on the machine and the project env is a
  repo-local uv `.venv`. It also says "NVIDIA RTX 4090 available"; there are
  **four**. Fix `CLAUDE.md` at a gate, not mid-phase.
- **E2 — the baseline working tree was not clean.** At the moment phase 0
  started, `main` carried an uncommitted modification to `.gitignore` (the
  archive-symlink rules quoted in §3.3) and three untracked governance files:
  `CLAUDE.md`, `PAPER_CANON.md`, `WORKFLOW.md`. See §5 for how this was
  resolved.
- **E3 — no pytest markers registered.** `gpu` / `data` markers required by
  PAPER_CANON §7.5 do not exist in `pyproject.toml`; phase 2 must add them.
- **E4 — `.claude/` is gitignored** (`.gitignore:55`), so the refactor's own
  phase skills and subagent definitions (12 files) are not versioned with the
  code they govern. Not a blocker for phase 0; Nikola's call whether the
  release should carry them.

---

## 5. Disposition of the unclean baseline tree

Decided by Nikola at the start of phase 0: **carry the pending files onto
`refactor/cleanup`; leave `main` untouched.**

Concretely:

- `git tag pre-refactor 9541085` — the tag points at the unmodified baseline
  commit on `main`, so it marks the true frozen "before" code state. `main`
  itself received no commit.
- `git checkout -b refactor/cleanup` — the modified `.gitignore` and the three
  untracked governance docs followed the checkout unchanged.
- The phase-0 commit on `refactor/cleanup` therefore contains: the `.gitignore`
  archive-symlink rules, `CLAUDE.md`, `PAPER_CANON.md`, `WORKFLOW.md`, and the
  phase-0 additions (`docs/refactor/*`, `tools/refactor/inventory.py`).

Consequence for later phases: a diff against `pre-refactor` will show the
`.gitignore` change and the three docs as part of the refactor, even though
they predate it. `git diff pre-refactor..refactor/cleanup -- .gitignore` is the
one place to remember this.

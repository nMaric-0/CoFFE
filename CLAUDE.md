# CLAUDE.md — CoFFE repository

## What this repo is

Public code release for the paper *"A Compact In-Domain Fusion Encoder versus a
Hyperspectral Foundation Model for Few-Shot HSI-LiDAR Land-Cover
Classification"* (Marić & Kocev). It contains three routes compared in the
paper: the compact **CoFFE** encoder (per-scene SimMIM/MAE pretraining,
input-level HSI+LiDAR fusion), the **MFT** architectural control, and the
**HyperSIGMA** foundation-model route (label-free adaptation) — all evaluated
by one frozen-encoder, 5-shot N-way, **Euclidean nearest-class-mean** protocol.

**`PAPER_CANON.md` at the repo root is the source of truth** for names,
protocol constants, results tables, invariants, and known discrepancies. Read
it before any change. Repo history note: the codebase predates the paper's
final naming — legacy "MFT-CPEA"/"Cosine" vocabulary is being retired per the
canon's naming law.

## Refactor protocol (while the cleanup is in progress)

The cleanup runs as gated phases, each a manually invoked skill:
`/phase0-baseline` → `/phase1-audit` → `/phase2-harness` → `/phase3-prune` →
`/phase4-rename` → `/phase5-restructure` → `/phase6-quality` →
`/phase7-tests` → `/phase8-release`.

- **One phase per session.** Start each phase in a fresh session (or after
  `/clear`). Never start phase N+1 in the same session as phase N.
- Do not run a phase whose preconditions (stated in its skill) aren't met, and
  never skip a gate.
- All work happens on the `refactor/cleanup` branch. Commit messages:
  `[phase N] <imperative summary>`. Never push, never force-push, never touch
  `main` — Nikola pushes.
- Every phase ends by appending a short entry to `docs/refactor/LOG.md`
  (what changed, verification results, open questions) and printing a gate
  checklist for Nikola.

## Hard rules (never violated, in any session, refactor or not)

1. **Behavior is frozen.** A refactor commit must not change any computed
   number. `tests/equivalence/` (once it exists, after phase 2) is the arbiter
   and must pass before every commit that touches code.
2. **Checkpoint compatibility.** `nn.Module` attribute names define state_dict
   keys — frozen unless a key-map shim + loading test is added
   (PAPER_CANON §7.2).
3. **Legacy readers.** Frozen experiment artifacts use old vocabulary
   (`mft_cpea`, `enhanced`, `spatial`, dir names like
   `houston_enhanced_spatial_run1`). Readers accept both old and new via the
   alias map; writers emit canonical only. On-disk-name literals are
   DO-NOT-RENAME (PAPER_CANON §7.3).
4. **`third_party/HyperSIGMA/` is vendored** — imports referencing it may
   change; its contents may not.
5. **No data/checkpoints assumed.** Tests must pass without datasets or
   HyperSIGMA checkpoints; real-data checks live behind `-m data` / `-m gpu`
   skip markers.
6. **Deletions and renames come only from the approved audit manifest
   (`docs/refactor/manifest.json`) and PAPER_CANON §1.** Anything else:
   propose, don't do.
7. **Surprises get reported, not fixed.** Any paper↔code behavior mismatch
   discovered mid-phase is written to the log and surfaced at the gate.

## Environment facts

- Dev machine: Linux, conda env (`coffe`), NVIDIA RTX 4090 available but **not
  required** — default all verification to CPU, mark GPU tests.
- Datasets live under `data/raw/` (never committed). HyperSIGMA ViT-Base
  checkpoints are downloaded via `scripts/download_hypersigma_checkpoints.sh`
  (never committed).
- Verification battery (run via the `verifier` subagent):
  `pytest -q -m "not gpu and not data"`, the equivalence harness, `ruff check`
  + `ruff format --check` (after phase 6), and a stale-vocabulary grep.

## Subagents

- `canon-reviewer` — read-only reviewer; call it at the end of phases 3–8 on
  the phase's diff before committing.
- `verifier` — runs the verification battery and returns a compact report;
  call it before every commit that touches code.

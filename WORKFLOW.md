# CoFFE Repository Cleanup — Claude Code Workflow

An agentic, gated workflow that takes `nMaric-0/CoFFE` from its current state
(legacy MFT-CPEA/Cosine naming, dead exploratory code, wrong README) to a
paper-grade public release for *"A Compact In-Domain Fusion Encoder versus a
Hyperspectral Foundation Model for Few-Shot HSI-LiDAR Land-Cover
Classification"* — with proof at every step that behavior didn't change.

## What's in this package

```
CLAUDE.md                      → repo root   (project memory, loaded every session)
PAPER_CANON.md                 → repo root   (distilled paper spec: naming law,
                                              protocol constants, results tables,
                                              invariants, 12 pre-seeded discrepancies)
.claude/agents/canon-reviewer.md            (read-only diff reviewer vs the canon)
.claude/agents/verifier.md                  (runs the verification battery)
.claude/skills/phase{0..8}-*/SKILL.md       (the nine gated phases, /phaseN-… commands)
WORKFLOW.md                    (this file — keep it anywhere)
```

## Design in one paragraph

The paper is frozen; the code must adopt its vocabulary without changing what
it computes. So the workflow is: **freeze a baseline (0) → decide everything
with evidence (1) → build a numeric safety net (2) → then, and only then,
delete (3), rename (4), restructure (5), polish (6)** — each step proven
no-op numerically by the phase-2 equivalence harness and reviewed by an
independent read-only subagent — **then make correctness permanent with a real
test suite and test runs (7) and rewrite the docs against the canon with a
fresh-clone release gate (8).** Deletions come only from a human-approved
manifest; renames only from the canon's naming law; legacy artifacts (frozen
configs with `model.name: "mft_cpea"`, existing experiment dirs, old
checkpoints) stay readable through explicit alias shims that the tests pin.

## Prerequisites

- Claude Code installed and signed in (https://code.claude.com/docs/en/quickstart).
- A clone of the repo with a working env: `conda activate coffe` (or venv) in
  which `python -c "import torch"` succeeds. GPU not required — verification is
  CPU; the optional real-data smoke in phase 7 uses the 4090.
- Optional but valuable: your real experiment trees / checkpoints on the same
  machine (enables the real-checkpoint fingerprint in phase 2 and smoke
  reproductions in phase 7). Datasets under `data/raw/` likewise optional.

## Install (once)

From the repo root, on a clean `main`:

```bash
cp /path/to/package/CLAUDE.md /path/to/package/PAPER_CANON.md .
cp -r /path/to/package/.claude .
git add CLAUDE.md PAPER_CANON.md .claude
git commit -m "Add Claude Code refactor workflow (canon, skills, agents)"
```

Then read `PAPER_CANON.md` yourself once, top to bottom — especially §8: those
twelve discrepancies (D1–D12) are decisions the workflow will put in front of
you, and you'll answer them faster having seen them coming.

## Run

Start `claude` in the repo root. **One phase per session** — after a phase's
gate, exit (or `/clear`) and start fresh; this keeps each phase's context clean
and its instructions authoritative. Type the skill name to run a phase:

| # | Command | What it does | Gate (you) | Feel |
|---|---|---|---|---|
| 0 | `/phase0-baseline` | Env record, baseline pytest, inventory, `pre-refactor` tag, `refactor/cleanup` branch | sanity check | ~15 min |
| 1 | `/phase1-audit` | Import-closure trace from every paper entry point; classify every file; resolve D1–D12 with evidence; write `docs/refactor/manifest.json` | **decisions**: D1 epochs, D3 λ, D9 exploratory fate, D11 SPLIT.md, notebook keep-list, delete-list review | the big read; 1–2 h of agent work, 20 min of yours |
| 2 | `/phase2-harness` | Golden fingerprints G1–G5 on synthetic mini-scenes via the real code paths + pre-refactor checkpoint fixture | confirm stability + fixture size | the safety net |
| 3 | `/phase3-prune` | Delete/archive exactly the approved manifest set; sweep dangling refs | diff vs manifest | destructive but boring — as it should be |
| 4 | `/phase4-rename` | Compat/alias layer first, then classes → files → configs → strings/docs to canon §1; CHANGES.md starts | rename table, alias tests, old-checkpoint load | the one you asked for (MFT-CPEA→CoFFE, cosine→Euclidean NCM) |
| 5 | `/phase5-restructure` | Proposes package layout, **waits for your approval**, then `coffe/` package + thin CLIs + `results/` + `.gitignore` fix | approve layout; clean-venv install log | biggest diff, still numerically no-op |
| 6 | `/phase6-quality` | ruff (replaces black/isort/flake8), type hints, docstrings, dependency pruning to the measured import set | dropped-deps list | polish |
| 7 | `/phase7-tests` | Unit/integration/e2e suite (masking, Eq. 1, NCM vs sklearn, config-parity with paper constants, aliases, CLI e2e), full runs, optional GPU smoke + optional CI | coverage, smoke y/n, CI y/n | "make sure it all works", permanently |
| 8 | `/phase8-release` | README/docs rewrite from the canon, CITATION.cff, fresh-clone release gate (7 checks), FINAL_REPORT | fill links/acks, review README, you merge+tag+push | ship |

Permission mode: the default interactive mode is fine — you'll be approving
file edits and commands as they come. If you trust a phase (3, 6 are the most
mechanical), `Shift+Tab` to auto-accept edits speeds it up; phases 1 and 5 are
deliberately conversational (plan → your approval → execution). If you prefer,
run phase 1 in plan mode first.

## Invariants the whole thing rides on

(Enforced by CLAUDE.md + the canon + both subagents; worth knowing as the human.)

1. **Numerics frozen** — the equivalence harness must report IDENTICAL before
   any code-touching commit in phases 3–8.
2. **Checkpoint compatibility** — `nn.Module` attribute names are state_dict
   keys and don't change; a committed pre-refactor checkpoint fixture proves
   old checkpoints load forever (test G4).
3. **Frozen artifacts stay readable** — readers accept legacy vocabulary via
   an alias map (`mft_cpea`→`coffe`, `enhanced`→`simmim`,
   `spatial/spectral/both`→`simmim_token/band/band_token`); writers emit
   canonical only; literals naming on-disk dirs are DO-NOT-RENAME.
4. **`third_party/HyperSIGMA/` untouched**; license/NOTICE surface in the README.
5. **No data required** — the whole verification story runs on synthetic
   tensors; real-data checks are opt-in and marker-gated.
6. **Claude never pushes** — merging `refactor/cleanup`, tagging `v1.0.0`, and
   pushing are your commands, printed for you at the phase-8 gate.

## If something goes wrong

- A phase's verifier reports DIVERGED → the phase must find the offending hunk
  (bisect its own uncommitted changes) and revert it; a divergence is never
  committed. If the divergence reveals a pre-existing bug, it goes to you as a
  decision (fix = behavior change = deliberate, isolated, re-baselined).
- A session dies mid-phase → start a new session, run the same `/phaseN-…`
  command; each skill re-checks preconditions and git state and continues.
- You dislike a phase's outcome after commit → `git revert` the phase commits
  on `refactor/cleanup` (or reset to the previous phase's last commit — the
  branch is yours until merge), rerun with amended instructions.
- Escape hatch: `git checkout main` / tag `pre-refactor` is always the world
  before anything happened.

## Notes

- Claude Code also has *dynamic workflows* (`ultracode`) for fanning out many
  parallel subagents; this cleanup is intentionally sequential-with-gates
  instead — each phase depends on the previous one's approved output, and you
  want review points, not throughput. The two subagents here are used within
  phases where independence matters (review) or logs are noisy (verification).
- After phase 8, keep `CLAUDE.md` (trim the "Refactor protocol" section, keep
  the hard rules — they're good permanent repo policy), keep `PAPER_CANON.md`
  (it's the best onboarding doc the repo will ever have), and keep the
  `verifier`/`canon-reviewer` agents for future changes. The phase skills can
  be deleted or left as archaeology.

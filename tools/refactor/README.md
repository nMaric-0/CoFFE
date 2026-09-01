# `tools/refactor/` — the audit tooling

**Released with the repo** (Nikola's decision at the phase-3 gate, 2026-09-01):
these are the scripts that produced the refactor's evidence, so anyone checking
the release can re-derive it instead of taking `docs/refactor/` on trust.

`inventory.py`, `closure.py` and `build_manifest.py` are stdlib-only and import
nothing from the paper code — they read the tree as text (`ast`, `git ls-files`)
and write only the JSON at their `--out`. `lambda_probe.py` is the exception on
both counts: it imports `numpy`, `torch`, `data.datasets.patched`,
`scripts.evaluate_cosine` and `utils.seed`, and it needs `data/raw/` plus a real
checkpoint, so it does not run in the default battery.

| script | what it does | default output |
|---|---|---|
| `inventory.py` | Python surface snapshot: per-file LOC, top-level defs/classes, resolved imports | `docs/refactor/inventory.json` |
| `closure.py` | static import closure from each named root set (paper entry points, tests, tooling); reports what nothing reaches | `docs/refactor/closure.json` |
| `build_manifest.py` | the per-file verdict ledger (class / verdict / target / evidence / approval) | `docs/refactor/manifest.json` |
| `lambda_probe.py` | measures the eval λ term (`z = mean_j(patch_emb_j) + λ·cls_emb`) — the D3 measurement | prints; needs `data/raw/` + a real checkpoint |

Run them from the repo root with the project venv:

```bash
.venv/bin/python tools/refactor/build_manifest.py                  # regenerates the ledger in place
.venv/bin/python tools/refactor/closure.py   --out /tmp/closure.json
.venv/bin/python tools/refactor/inventory.py --out /tmp/inventory.json
```

## Two things to know

**`manifest.json` is live; `closure.json` and `inventory.json` are frozen
phase-1 snapshots.** `build_manifest.py` is kept in sync with the tree and with
each gate's approvals, so re-running it is expected to be a no-op — that is the
check that the ledger still matches reality. It is the one tool that writes in
place, and it exits non-zero if it holds a verdict for a path that is neither
tracked nor accounted for as executed — so a phase that carries out a record
without teaching the tool about it fails loudly instead of dropping the record.

The other two committed JSONs describe the **pre-prune** tree and are cited by
line in `AUDIT.md`, so both scripts now refuse to overwrite an existing `--out`
(exit 1; `--force` overrides). Re-run them to a scratch path instead.

**Pruned entry points stay listed.** `closure.py`'s `ROOTS` still names all 25
entry points phase 3 removed — the D8 duplicate twin, the 15 D9 exploratory
scripts, and the 9 duplicate notebooks; that list is provenance. Roots that no
longer exist are skipped and reported under `pruned_roots`
(`summary.pruned_root_paths` = 25 today), never seeded into the BFS.
Likewise `build_manifest.py` keeps a verdict record for every file the audit
ruled on: paths already carried out are emitted from the verdict table with
`"executed_in_phase"`, so the ledger stays complete instead of quietly shrinking
to the survivors.

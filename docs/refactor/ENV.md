# ENV.md — pre-refactor environment record (phase 0)

Recorded 2026-08-31 on the dev machine, at baseline commit
`9541085` ("Final experiments").

## Interpreter and activation

The project environment is a **repo-local `uv`-managed virtualenv**, not a
conda env:

```bash
cd /work/nmaric/CoFFE/CoFFE
source .venv/bin/activate      # prompt: (CoFFE)
```

`CLAUDE.md` states "conda env (`coffe`)". **No conda is installed on this
machine** (`conda` is not on `PATH`; no `~/miniconda3`, `~/anaconda3`, or
`/opt/conda`). Recorded as discrepancy **E1** in `BASELINE.md`; `CLAUDE.md`
should be corrected at a later gate.

Note: `.venv` has **no `pip`** (`No module named pip`) — it is uv-managed.
Install packages with `uv pip install ...`, and read the installed set from
`packages.txt` (generated via `importlib.metadata`, not `pip freeze`).

| Item | Value |
|---|---|
| Python | 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0] |
| `sys.executable` | `/work/nmaric/CoFFE/CoFFE/.venv/bin/python` |
| venv manager | `uv` 0.6.3 (`.venv/pyvenv.cfg`), base interpreter `/usr/bin` |
| system-site-packages | false |

## Core scientific stack

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.3 |
| matplotlib | 3.10.9 |
| einops | 0.8.2 |
| timm | 1.0.27 |
| PyYAML | 6.0.3 |
| pytest | 9.0.3 |
| pytest-cov | 7.1.0 |

Full list of 205 installed distributions: `docs/refactor/packages.txt`.

## CUDA / hardware

| Item | Value |
|---|---|
| `torch.cuda.is_available()` | **True** |
| `torch.version.cuda` | 12.8 |
| GPUs | 4 × NVIDIA GeForce RTX 4090, 24564 MiB each, driver 610.43.02 |
| CPU | 64 logical cores |
| RAM | 251 GiB total |
| OS | Linux-6.8.0-138-generic-x86_64-with-glibc2.39 |

**CUDA is present but the refactor verifies on CPU.** Every phase's
verification battery runs `-m "not gpu and not data"`; GPU-dependent checks are
gated behind the `gpu` marker (PAPER_CANON §7.5). `CLAUDE.md` describes a
single RTX 4090; four are visible here.

## Storage layout

Bulk artefacts were archived to `/ceph/home/nmaric/CoFFE` on 2026-08-28 and the
in-repo paths are now **symlinks**: `checkpoints/`, `logs/`, `data/raw/`, and
every per-run directory under `experiments/`. The uncommitted `.gitignore`
change at baseline exists to keep those symlinks ignored (see `BASELINE.md`).

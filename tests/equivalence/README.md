# `tests/equivalence/` — the behaviour-equivalence harness

Created in **phase 2** of the cleanup. Its job is narrow and absolute:

> capture what the code computed **before** the refactor, so every later phase
> can prove it still computes exactly that.

PAPER_CANON §7.1 freezes numerics — no refactor step may change a loss value,
a feature, a prototype or an OA. This directory is the arbiter of that rule.

## Running it

```bash
pytest tests/equivalence -q                  # the whole harness
pytest tests/equivalence -q -m "not slow"    # skip the HyperSIGMA ViT builds
```

No datasets and no HyperSIGMA checkpoints are required (PAPER_CANON §7.5):
scenes are synthesised, and the HyperSIGMA ViT bodies are randomly initialised.

### The goldens are bit-exact on **one** environment (read this first)

`golden/meta.json` records it: **Python 3.12.3, torch 2.11.0+cu128,
numpy 2.4.4**. `pyproject.toml` declares floors, not pins, so a fresh
`pip install -e ".[dev]"` resolves something newer — and the goldens do not
survive that. Measured in the phase-8 release gate, in a fresh clone:

| Environment | Result |
|---|---|
| as `pip install -e ".[dev]"` resolved it (torch 2.14.0, numpy 2.5.2) | **19 of 33 failed** — G1/G2/G5 numeric drift (e.g. `G5[simmim_token].masked_loss` off by 2.6e-04 against the 1e-06 tolerance) plus the `meta.json` environment assertion |
| `torch==2.11.0` (+ `torchvision==0.26.0`, which `timm` requires), `numpy==2.4.4` | **33 passed** |

The rest of the suite does not care: on the newer environment the same fresh
clone runs `-m "not gpu and not data" --ignore=tests/equivalence` at
**486 passed, 0 failed**, and on the pinned one the whole selection is
**519 passed, 8 skipped**.

So a failure here means one of two things, and the first is far more likely:
**your environment differs from the goldens'**, or the code's behaviour actually
changed. Check `python -c "import torch, numpy; print(torch.__version__,
numpy.__version__)"` against `golden/meta.json` before concluding anything.
Note `timm` pulls `torchvision`, so pinning torch means pinning torchvision to
its matching release too — a torch/torchvision mismatch fails these tests with
`RuntimeError: operator torchvision::nms does not exist`, which is not numeric
drift at all.

Nothing about this is a defect in the code under test: bit-exactness on a fixed
environment is exactly what the harness promises (`conftest.py`'s determinism
contract). What the release does not yet ship is a way to *get* that
environment — recorded as an open item in
[`docs/refactor/FINAL_REPORT.md`](../../docs/refactor/FINAL_REPORT.md).

## What is pinned

| Golden | File | What it fixes |
|---|---|---|
| **G1** | `golden/g1_pretrain_loss.json` | Per-epoch mean pretraining loss (8 decimals) for every Table 2 objective — `simmim_band`, `simmim_token`, `simmim_band_token`, `mae` — plus the Houston band+token rate actually used (D19), for CoFFE and the MFT control. Via `coffe.pretrain.loop.run_pretrain`. |
| **G2** | `golden/g2_encoder_forward.json` | Every state_dict tensor of a freshly-seeded model (pins init *and* the §7.2 key contract) and the pooled eval feature `z` for CoFFE HSI+LiDAR, CoFFE HSI-only, `MFTOriginal`, and the HyperSIGMA wrapper in both paper input regimes. |
| **G3** | `golden/g3_episodic_eval.json` | OA (6 decimals) and a SHA-256 over the **full per-episode per-query argmin assignment matrix**, via `coffe.eval.episodic.run_evaluation`. |
| **G4** | `fixtures/*.pth` | The pre-refactor checkpoints themselves. Loaded through the live key-mapping code on every run; if anyone breaks state_dict keys or loader plumbing, G4 fails. |
| **G5** | `golden/g5_masking.json` | Realized band/token mask rates (4 decimals, exact), the union-mask rate, and the masked reconstruction loss — pins Eq. 1 (union of the two masks, centre weights with mean one). |

`golden/meta.json` records python/torch/numpy versions, the git SHA and the
dirty flag at generation time, plus a per-group `groups` record so a partial
regeneration (`--only g5`) cannot erase the provenance of the others.

`"dirty": true` in `meta.json` is **expected and not a warning sign**: the
harness itself is necessarily uncommitted at the moment it first generates its
own goldens. What matters is `git_sha`, which must be a phase-0/1/2 commit —
`make_golden.py` refuses to run otherwise, and separately refuses if any tracked
file outside this directory has uncommitted modifications, so the *code under
test* is never dirty even when the harness is. `golden/real_local.json` is machine-specific,
gitignored, and never asserted against.

## The paper path, not a convenient path

The harness drives the same functions the paper runs went through
(`coffe/runners/*_runner.py` are thin wrappers over these):

* `coffe.pretrain.loop.run_pretrain`
* `coffe.eval.episodic.run_evaluation`
* `coffe.eval.episodic.load_model_with_checkpoint` (and its
  `fix_state_dict_keys`, which is the *live* key-mapping code — PAPER_CANON §8 D15)
* `coffe.data.datasets.*PatchedDataset`, fed synthetic `.mat` files in the real
  on-disk layout, so patch handling and min-max normalisation are the repo's

In particular the eval feature is captured off the **live** path
(`z = mean_j(patch_emb_j) + 0.5 · cls_emb`, PAPER_CANON §8 D3), not off the dead
`CoFFE.forward_episode`. A dedicated test records that the two still
agree, so pruning the dead one later can be shown to be behaviour-preserving.

## Configs

Semantics come from the **frozen run configs** in `experiments/*/pretrain_config.yaml`,
not from `configs/` (PAPER_CANON §8 D13 — `configs/` is stale). Only the
schedule is scaled down: 3 epochs instead of 1500-3000, batch 16, 20 episodes
with `k_query=10` instead of 1000 episodes with `k_query=100`. Every semantic
knob — architecture, mask rates, loss, optimiser, `distance_metric: euclidean`,
`use_projection: False`, `pool_sigma: None`, N-way full-class, K=5 — is the
paper's.

`make_golden.py` writes the trainer's tqdm progress bars to stderr (tqdm 4.67
has no env switch for them); redirect with `2>/dev/null` if you want only the
fingerprint log.

## Regenerating

Don't, unless you are re-baselining after a torch upgrade. `make_golden.py`
refuses to run on a tree with commits beyond phases 0-2 or with modifications to
tracked files outside this directory; the deliberate re-baseline procedure is in
its module docstring.

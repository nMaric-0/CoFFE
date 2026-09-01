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

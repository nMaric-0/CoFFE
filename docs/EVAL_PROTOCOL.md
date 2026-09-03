# Evaluation protocol — frozen encoder, Euclidean nearest-class-mean

Every route in this repository (CoFFE, the MFT control, HyperSIGMA) is measured
by the same *kind* of protocol — frozen encoder, N-way 5-shot episodes,
Euclidean nearest-class-mean. Nothing is finetuned. The episode count and query
count differ between the paper's two tables; see the note in the table below.

> **Scope.** This file documents *what the code does*.
> [`PAPER_CANON.md`](../PAPER_CANON.md) §4 is the source of truth for the
> constants below, and the [README](../README.md#reproduce) maps each paper
> table to the command that reproduces it.

## The protocol

| Constant | Value |
|---|---|
| Episode | **N-way**, N = the scene's full class count (Houston 15, Trento 6, MUUFL 11) |
| Support | K = 5 per class |
| Queries | 100 per class, class-balanced |
| Episodes | 1000 per run for Table 2 (CoFFE / MFT). Table 3's HyperSIGMA sweep used **2000**, and one cell used `k_query=30` — PAPER_CANON §8 D18: read the protocol constants per table |
| Classifier | **Euclidean nearest-class-mean**: the class mean is the average of its 5 support features; a query goes to the nearest class mean |
| Encoder | Frozen; projection head **off** (`use_projection: False`) |
| Metric | OA. Query sets are class-balanced, so OA = AA; κ is reported in the JSON but omitted from the paper by design |
| Reporting | OA mean ± 95 % CI over episodes; across-seed std over seeds `[42, 123, 456, 789, 1011]` for CoFFE/MFT |

This is a nearest-class-mean classifier on fixed features (Mensink et al.), not
a prototypical network (Snell et al.): nothing is trained episodically, and
there are no learnable similarity parameters at all.

`distance_metric` also accepts `"cosine"` (with a `temperature`), and the
HyperSIGMA evaluator reports both blocks. **The paper is Euclidean throughout**
— `distance_metric="euclidean"` in every Table 2 run and every Table 3 cell but
one, and even that cell's quoted value is its `euclidean` sub-block; only its
`eval_config.json` records `cosine` (PAPER_CANON §8 D18). Since the phase-4 gate
the CLI default is `euclidean` too, so an unflagged invocation is already the
paper protocol. That default move is one of the refactor's two signed-off
default changes (the other is the phase-7 gate's alignment of six further
evaluator defaults, plus `num_episodes` 2000 → 1000, Table 2's count), and
neither changes a paper number, because every paper run passes these values
explicitly (CHANGES.md, "What deliberately did **not**
change").

## What the eval feature actually is

For CoFFE the pooled feature is **not** simply the mean of the patch tokens.
The live path
([`coffe/eval/episodic.py`](../coffe/eval/episodic.py) →
[`CoFFE.eval_patch_embeddings`](../coffe/models/coffe.py)) folds the weighted
class-agnostic token into every patch token first:

```
z = mean_j(patch_emb_j) + lambda_factor * cls_emb        # lambda_factor = 0.5 in every paper run
```

This is PAPER_CANON §8 **D3**: the behaviour is frozen and documented, and the
paper's §3 description ("patch tokens pooled") is the incomplete one. The MFT
control and HyperSIGMA override the same method with a pass-through, so the gap
is CoFFE-only.

Optional pooling knob: `pool_sigma` selects Gaussian centre-weighted pooling
instead of the uniform mean. Every Table 2 run records `pool_sigma: null`.

## Running it

One entry point per route, each defaulting to that route's published
architecture, all three sharing this evaluator body:

```bash
# CoFFE — the defaults ARE the paper protocol, so nothing needs spelling out
python scripts/evaluate.py \
    --checkpoint experiments/<run>/checkpoints/checkpoint_epoch_950.pth \
    --dataset houston

# the same run, protocol written out in full (identical result)
python scripts/evaluate.py \
    --checkpoint experiments/<run>/checkpoints/checkpoint_epoch_950.pth \
    --dataset houston \
    --k-shot 5 --k-query 100 --num-episodes 1000 --split all \
    --distance-metric euclidean --no-projection

# the MFT architectural control (64/8/2 + mlp 512 + mcross by default)
python scripts/evaluate_mft.py \
    --checkpoint experiments/<run>/checkpoints/checkpoint_epoch_950.pth \
    --dataset houston

# HyperSIGMA — same episodes, but its own defaults are NOT the paper's:
# pass --split all --k-query 100 --num-episodes 2000 for a Table 3 cell
python scripts/evaluate_hypersigma.py --dataset houston --mode fused \
    --split all --k-query 100 --num-episodes 2000
```

The evaluated checkpoint is **not** the final one: epoch 950 on Houston, 975 on
Trento/MUUFL for CoFFE, 950 for all six MFT cells, with one cell at 800
(PAPER_CANON §8 D17). Each `configs/{coffe,mft}/` cell config names its own
epoch in its header.

`--n-way` defaults to the scene's full class count, which is the paper setting.
Passing `--name mft_original` to `scripts/evaluate.py` is no longer how the
control is evaluated — it has its own pipeline (phase-7 gate).
The shell drivers [`scripts/reproduce/run_eval.sh`](../scripts/reproduce/run_eval.sh) and
[`scripts/reproduce/run_eval_trento.sh`](../scripts/reproduce/run_eval_trento.sh) wrap the same
call, but **their built-in defaults are not the paper's** (`k_query` 30 / 600
episodes; the Trento one also defaults to 8 heads, 4 layers, λ 1.0, `k_query`
19 and the projection head on) — each script's header lists them. Meanwhile
[`coffe/runners/eval_runner.py`](../coffe/runners/eval_runner.py) is the notebook entry point — it
reads the architecture back out of the pretraining run's
`pretrain_config.yaml`, so eval cannot silently drift from the checkpoint.

**One caveat on that inheritance: `use_projection` is inherited too.** The
SimMIM pretrain configs keep the head on (it is a pretraining part), so a bare
`run_evaluation(...)` that does not say otherwise evaluates *with* the head —
not the paper protocol. `notebooks/evaluate.ipynb` surfaces `use_projection` as
a top-level parameter set to `False`, `scripts/reproduce/sig_significance_config.py`
passes `use_projection: False` in its shared `eval_params`, and every frozen
paper `eval_config.json` records `false`; so no published number is affected,
but a new caller of the runner has to pass it. The CLIs default to off.

## Checkpoint compatibility

Checkpoints predating the phase-4 rename load unchanged: only class, file and
config-value names changed, never `nn.Module` attribute names (PAPER_CANON
§7.2). `fix_state_dict_keys` in
[`coffe/eval/episodic.py`](../coffe/eval/episodic.py) strips the `encoder.`
prefix left by the pretraining wrappers, and channel-dependent weights are
skipped with a loud error if the checkpoint's band count does not match the
dataset. `tests/equivalence/` pins a pre-refactor fixture checkpoint loading
through the renamed classes.

Old configs that still say `model.name: "mft_cpea"` or
`objective: "enhanced"` are accepted via [`coffe/compat.py`](../coffe/compat.py),
which maps them and warns once.

## See also

- [`docs/PRETRAINING.md`](PRETRAINING.md) — the objectives that produce the
  encoders.
- [`PAPER_CANON.md`](../PAPER_CANON.md) §4 (protocol), §8 D2/D3/D18 (protocol
  drift found in the audit).
- [`README.md`](../README.md#results) and [`PAPER_CANON.md`](../PAPER_CANON.md)
  §6 — the published tables; [`results/README.md`](../results/README.md) — the
  JSONs behind them. (`docs/presentation/RESULTS.md` is an earlier compilation,
  superseded — see its header.)
- [`docs/HYPERSIGMA.md`](HYPERSIGMA.md) — the foundation-model route's own
  input regimes, adaptations and evaluator quirks.

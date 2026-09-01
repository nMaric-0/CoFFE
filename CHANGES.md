# CHANGES

Renames applied while preparing this repository for release, and what still
accepts the old names.

**No computed number changed.** Every entry below is a name: a class, a file, a
config value, or a printed label — with one signed-off exception, the
`distance_metric` default (see the end of this file). The behaviour-equivalence
harness (`tests/equivalence/`) stayed green throughout: G2–G5 are **bit-identical**
at zero tolerance, and G1's loss trajectories match the goldens' stored 8-decimal
precision (residual ≤ 5e-09, BLAS reduction-order noise that the harness's
declared tolerance exists for — it is present on goldens this work never touched).
A pre-rename checkpoint fixture still loads through the renamed classes.

---

## Phase 4 — rename to the paper's vocabulary (2026-09-01)

The repository predates the paper's final naming. `PAPER_CANON.md` §1 is the
naming law; this table is what was carried out.

### Public API — classes

| Old | New | Where |
|---|---|---|
| `MFTCPEACosine` | `CoFFE` | `models/coffe.py` |
| `MFTOriginalCosine` | `MFTOriginal` | `models/mft_original.py` |
| `HyperSIGMACosine` | `HyperSIGMAFewShot` | `models/hypersigma/few_shot.py` |
| `EnhancedMaskedSpectralSpatialModel` | `SimMIMPretrainModel` | `pretrain/simmim.py` |

`HyperSIGMADual` and `MFTMAEPretrainModel` / `MFTSpatialMaskPretrainModel` /
`MAEPretrainModel` keep their names.

### Public API — methods and arguments

| Old | New | Note |
|---|---|---|
| `CoFFE.adapt_embeddings(...)` | `CoFFE.eval_patch_embeddings(...)` | also on `MFTOriginal` and `HyperSIGMAFewShot`, where it is a pass-through |
| `CoFFE(lambda_factor=...)`, `self.lambda_factor` | `CoFFE(cls_token_weight=...)`, `self.cls_token_weight` | constructor argument and attribute only — **the config key and CLI flag stay `lambda_factor`** (frozen artifacts record it) |

Both renames are mandated by PAPER_CANON §8 D3 and are safe under §7.2:
`lambda_factor` is a plain float and appears in none of the 42 state_dict keys.

### Files

| Old | New |
|---|---|
| `models/mft_cpea_cosine.py` | `models/coffe.py` |
| `models/hypersigma/hypersigma_cosine.py` | `models/hypersigma/few_shot.py` |
| `pretrain/masked_modeling_enhanced.py` | `pretrain/simmim.py` |
| `scripts/evaluate_cosine.py` | `scripts/evaluate.py` |
| `scripts/pretrain_enhanced.py` | `scripts/pretrain.py` |
| `scripts/evaluate_hypersigma_cosine.py` | `scripts/evaluate_hypersigma.py` |
| `scripts/run_cosine_eval.sh` | `scripts/run_eval.sh` |
| `scripts/run_trento_cosine_eval.sh` | `scripts/run_eval_trento.sh` |
| `tests/test_pretrain_enhanced.py` | `tests/test_pretrain_simmim.py` |
| `docs/ENHANCED_PRETRAINING.md` | `docs/PRETRAINING.md` |
| `docs/COSINE_VARIANT.md` | `docs/EVAL_PROTOCOL.md` |

All were `git mv`d. History follows for the nine code/test files; the two
`docs/` files were also rewritten in the same commit (their content described
code that no longer exists — see below), so git records them as delete+add. A
later phase moves the packages into a single `coffe/` package; these basenames
will not change again.

### Configs

Reorganised into `configs/{coffe,mft,hypersigma}/` per PAPER_CANON §9.
**No value inside any config changed except `model.name` and
`pretrain.objective`** (verified key-by-key against the pre-rename files).

| Old | New |
|---|---|
| `configs/pretrain/<scene>_pretrain_enhanced.yaml` | `configs/coffe/<scene>_simmim.yaml` |
| `configs/pretrain/<scene>_pretrain_hsi_only.yaml` | `configs/coffe/<scene>_simmim_hsi.yaml` |
| `configs/pretrain/<scene>_pretrain_mae.yaml` | `configs/coffe/<scene>_mae.yaml` |
| `configs/pretrain/<scene>_pretrain_mae_hsi_only.yaml` | `configs/coffe/<scene>_mae_hsi.yaml` |
| `configs/pretrain/mft_original_<scene>_spatial.yaml` | `configs/mft/<scene>_simmim_token.yaml` |
| `configs/pretrain/mft_original_<scene>_mae.yaml` | `configs/mft/<scene>_mae.yaml` |
| `configs/pretrain/hypersigma_<scene>_adapt.yaml` | `configs/hypersigma/<scene>_patchnative_joint_sem.yaml` |
| `configs/pretrain/hypersigma_houston_adapt_pca100.yaml` | `configs/hypersigma/houston_patchnative_pca100_joint_sem.yaml` |
| `configs/pretrain/hypersigma_houston_adapt_native_sem.yaml` | `configs/hypersigma/houston_backbonenative_upscale_sem_only.yaml` |
| `configs/pretrain/hypersigma_<scene>_adapt_native_sem_pad.yaml` | `configs/hypersigma/<scene>_backbonenative_pad_sem_only.yaml` |
| `configs/eval/hypersigma_houston.yaml` | unchanged |

The six `configs/coffe/<scene>_simmim*.yaml` files are **base** configs, not
per-cell reproduction recipes. Exactly one of them is token-masking; the rest
are band-masking at 0.9:

| Config | `band_mask_ratio` | `spatial_mask_ratio` | i.e. |
|---|---|---|---|
| `configs/coffe/houston_simmim.yaml` | 0.0 | 0.75 | SimMIM token |
| every other `configs/coffe/*_simmim*.yaml` (Houston HSI-only, both Trento, both MUUFL) | 0.9 | 0.0 | SimMIM band, at a rate **no Table 2 cell used** |

The 5-seed significance experiment overrides the pair at runtime from the
canonical run dir, so these rates are a starting point, not a claim about any
paper cell. Each file now states its own pair in a header comment. They are
named for the objective (`simmim`) rather than a regime precisely because the
regime is not fixed by the file — set both rates explicitly to reproduce a
specific cell.

### Config values

| Key | Old | New |
|---|---|---|
| `model.name` | `"mft_cpea"` | `"coffe"` |
| `pretrain.objective` | `"enhanced"` | `"simmim"` |
| `pretrain.objective` | *(absent, defaulted to `"enhanced"`)* | now written explicitly as `"simmim"` |

Regime ids used in prose and labels: `spectral` → **SimMIM band**, `spatial` →
**SimMIM token**, `both` → **SimMIM band+token**. Those three strings are *not*
renamed where they are components of on-disk directory names
(`houston_enhanced_spatial_run1`, `_ENHANCED_CANONICAL`, and the substring tests
in `scripts/compile_results.py`) — those name files that exist.

### Written values and printed labels

| Where | Old | New |
|---|---|---|
| `results.json` `model_type` | `"MFTCPEACosine"` / `"MFTOriginalCosine"` | `"CoFFE"` / `"MFTOriginal"` |
| eval log / table header | `"MFT-CPEA-Cosine"`, `"original-MFT"` | `"CoFFE"`, `"MFT (original)"`, `"nearest-class-mean"` |
| `docs/presentation/RESULTS.json` model / regime | `"MFT-CPEA"`, `"Enhanced: spatial"` … | `"CoFFE"`, `"SimMIM token"` … |
| `scripts/build_experiment_metadata.py` family id | `mft_cpea_enhanced` / `mft_cpea_mae` | `coffe_simmim` / `coffe_mae` |
| `scripts/gather_requested_results.py` output key | `mft_cpea_baselines` | `coffe_baselines` |
| default plot dir | `outputs/cosine_eval/…`, `results/cosine_eval/…` | `outputs/eval/…`, `results/eval/…` |
| default fallback dirs in `scripts/pretrain.py` | `checkpoints/pretrained_enhanced`, `logs/pretrain_enhanced` | `checkpoints/pretrained`, `logs/pretrain` |

`docs/presentation/RESULTS.json` and `RESULTS.md` were relabelled in place with
exactly the mapping the generator now emits; every number in them is unchanged.
The generated files under `experiments/` were **not** touched — they are frozen
run artifacts.

### Terminology

- "prototypical network" → **nearest-class-mean (NCM) on frozen features**. The
  cited prototypical-network method (Snell et al.) is not what this repository
  evaluates; there is no episodic training.
- "5-way" → **N-way**, N = the scene's full class count (15 / 6 / 11).
- "Cosine" as a name, title or heading → dropped. `cosine` survives only as an
  option *value* of `distance_metric`.

---

## Compatibility guarantees

**1. Checkpoints load unchanged.** No `nn.Module` attribute name was renamed, so
state_dict keys are identical (PAPER_CANON §7.2). A pre-rename checkpoint
fixture is committed under `tests/equivalence/fixtures/` and a test loads it
through the renamed classes on every run.

**2. Frozen experiment trees still read.** `coffe_compat.py` maps the retired
vocabulary and emits one `DeprecationWarning` per value, naming the artifact it
came from:

```python
from coffe_compat import normalize_model_name, normalize_objective
normalize_model_name("mft_cpea")     # -> "coffe"
normalize_objective("enhanced")      # -> "simmim"
normalize_variant("spatial")         # -> "simmim_token"
normalize_model_type("MFTCPEACosine")  # -> "CoFFE"
```

It is wired into every reader of a frozen artifact: the model dispatch in
`scripts/pretrain.py` and `scripts/evaluate.py`, the architecture auto-load in
`lib/eval_runner.py`, `scripts/compile_results.py`, and
`scripts/build_experiment_metadata.py`. Writers emit canonical values only,
with one disclosed exception: `scripts/aggregate_significance.py` reproduces the
significance experiment's own `group`/`variant` directory-name components (see
the table below).

**3. Old imports still work.** The retired class names still resolve from the
packages they used to live in, and from the compat module, each with a
deprecation warning:

```python
from models import MFTCPEACosine                     # -> models.coffe.CoFFE
from models import MFTOriginalCosine                 # -> MFTOriginal
from models.hypersigma import HyperSIGMACosine       # -> HyperSIGMAFewShot
from pretrain import EnhancedMaskedSpectralSpatialModel  # -> SimMIMPretrainModel
from coffe_compat import MFTCPEACosine               # same object
```

The alias *is* the canonical class, so `isinstance` checks and checkpoint loads
behave identically either way. `tests/test_compat.py` (25 tests) pins all three guarantees, and
`tests/equivalence/test_equivalence.py::test_g1_legacy_vocabulary_config_trains_identically`
runs a frozen-vocabulary config end-to-end through `run_pretrain` and checks the
loss trajectory against the golden.

### Per-cell reproduction configs (added at the gate)

Every one of the paper's 30 Table-2 cells now has a config carrying **the exact
recipe of the run that produced its published mean**, generated from that run's
frozen `pretrain_config.yaml` by `tools/refactor/make_cell_configs.py`:

| Route | Files |
|---|---|
| `configs/coffe/` | `<scene>_simmim_{band,token,band_token}[_hsi].yaml` (18, new) and `<scene>_mae[_hsi].yaml` (6, already matched their cell — stamped with provenance) |
| `configs/mft/` | `<scene>_simmim_token.yaml`, `<scene>_mae.yaml` (6, already matched — stamped) |

Each header names the cell, its paper OA, the source run, and the **evaluated
checkpoint epoch** (950/975, not the final one — PAPER_CANON §8 D17).

`paths:` is deliberately *not* copied into the **18 generated** files — the
frozen values name the run's own output directories. The **12 stamped** files
(the MAE and MFT cells, which already matched their run) keep the `paths:` they
already had, because config `paths:` values are on the audit's DO-NOT-RENAME
list; none of those directories exists on disk today (the paper's checkpoints
live under `experiments/<run>/checkpoints/`), but a raw
`python scripts/pretrain.py --config ...` will create and write into them.

The six `configs/coffe/<scene>_simmim[_hsi].yaml` files are **not** cell recipes
— they are the 5-seed significance runner's base configs (it clones per-variant
mask rates at launch), and each now says so in its header. Note the twelve MAE
and MFT cell configs do double duty: `sig_significance_config.py:131-147` also
uses them as base configs for the `enhanced_mae`, `mft_mae` and `mft_spatial`
groups. That is harmless — those groups set `clone_mask=False`, so the runner
uses the file's values, which are exactly the cell's.

### `compile_results.py` no longer calls the MFT control "CoFFE"

`classify()` split the two apart using the `model_type` the evaluator writes.
Before, every single-metric result got one label (`"MFT-CPEA"`, then `"CoFFE"`),
so 47 MFT-control results were labelled as CoFFE and could out-rank a genuine
CoFFE run as a cell's representative in six groups. The committed
`RESULTS.json` predates those runs and is unaffected.

## Where the retired names still appear

By design, in exactly these places — this is the list the verification battery's
stale-vocabulary grep excludes:

| File | Why |
|---|---|
| `coffe_compat.py`, `tests/test_compat.py` | the alias table itself, and its tests |
| `tests/equivalence/test_equivalence.py`'s `test_g1_legacy_vocabulary_config_trains_identically` | drives a frozen-vocabulary config end to end on purpose |
| `.claude/agents/*.md`, `.claude/skills/*` | the refactor tooling's own specification of the retirement (same category as `docs/refactor/`) |
| `_LEGACY_ALIASES` in `models/__init__.py`, `models/hypersigma/__init__.py`, `pretrain/__init__.py` | the package-level import shims, which delegate to that table |
| `tools/refactor/apply_renames.py`, `tools/refactor/build_manifest.py` | the rename table and the audit ledger — they exist to record old→new |
| `CHANGES.md`, `PAPER_CANON.md`, `docs/refactor/*`, `CLAUDE.md`, `WORKFLOW.md` | documents about the retirement |
| on-disk-name literals (`_ENHANCED_CANONICAL`, dir-name substring tests, `paths:` values, experiment names in notebook parameter cells) | they name files that exist |
| stored notebook **outputs** | execution records of runs made before the rename |
| `scripts/aggregate_significance.py`'s emitted `group` / `variant` keys | they reproduce the significance experiment's own directory-name components and the frozen `significance_report.json` schema |
| `experiments/**` | frozen run artifacts |

Anything else is a bug: a stale *reference* to a renamed module is a broken
import, not a cosmetic issue.

## What deliberately did **not** change

- Any default that affects a computed number — every hyperparameter, constant
  and piece of maths, with **one deliberate, signed-off exception**:
  `distance_metric` now defaults to `"euclidean"` instead of `"cosine"`
  (phase-4 gate, 2026-09-01). It is the paper protocol, PAPER_CANON §1 forbids
  cosine as a default, and no paper run is affected because **every paper run
  passes `distance_metric` explicitly** — so no default is ever consulted. (Not
  every paper run passes *euclidean*: one Table 3 cell records `cosine`, see
  PAPER_CANON §8 D18.) That is why the equivalence goldens are unchanged.
  Cosine remains fully selectable via `--distance-metric cosine`. Changed in
  `scripts/evaluate.py` (CLI + `_DEFAULT_ARGS` + config fallbacks),
  `scripts/evaluate_hypersigma.py`, `lib/eval_runner.py`, and the `CoFFE` /
  `MFTOriginal` / `HyperSIGMAFewShot` constructors.
- The only defaults that did move are four **output-path fallbacks** whose names
  carried retired vocabulary and which no shipped config or reader relies on
  (all 26 pretraining/adaptation configs set `paths.checkpoint_dir` /
  `log_dir` explicitly — the eval config has no `paths:` block — so nothing
  shipped reaches the fallback; note two of the new fallback paths are the
  parent directories those configs write into): they are the last two rows of the label table
  above. They change where a bare invocation would *write*, never what it
  computes.
- The `lambda_factor` config key and `--lambda-factor` flag (frozen artifacts
  record them under that name).
- On-disk artifact names: experiment directories, `checkpoint_dir` / `log_dir`
  values in configs, `_ENHANCED_CANONICAL`, `EVAL_NAME`, and the dir-name
  substring tests in the aggregators.
- `third_party/HyperSIGMA/`, which is vendored.
- Stored notebook *outputs*: they are execution records of past runs and still
  show the old log lines. Only notebook sources were edited.

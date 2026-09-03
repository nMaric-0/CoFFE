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

## Phase 5 — restructure into an installable package (2026-09-01)

The release is now `pip install -e .`-able: one package, `coffe/`, plus thin
command-line entry points under `scripts/`. **No computed number changed** —
every Python move below was verbatim, and the equivalence goldens are
unchanged. Paths in the phase-4 table above name pre-move locations; this table
is where those files live now.

### Packages

| Before | After |
|---|---|
| `models/` | `coffe/models/` |
| `pretrain/` | `coffe/pretrain/` |
| `data/{datasets,samplers}/` | `coffe/data/{datasets,samplers}/` |
| `utils/` | `coffe/utils/` |
| `lib/` | `coffe/runners/` |
| `trainers/pretrain_trainer.py` | `coffe/pretrain/trainer.py` |
| `coffe_compat.py` | `coffe/compat.py` |

The `trainers` package is dissolved. `PretrainTrainer` is re-exported from
`coffe.pretrain`. Its only other public name, `create_pretrain_dataloaders`, was
**deleted** at the phase-5 gate: it had no caller anywhere in the repo, and its
`from .masked_modeling import …` resolved to `trainers.masked_modeling`, which
never existed — so any call would have raised `ModuleNotFoundError`. The move
into `coffe/pretrain/` would have silently repaired it; deleting it instead
keeps the release free of code no paper run ever executed.

### Scripts became thin CLIs

The paper logic moved into the package; each script keeps only its argparse
block and calls `main()`. The script paths themselves are unchanged, so every
documented invocation (`python scripts/evaluate.py …`) still works.

| Script | Logic now lives in |
|---|---|
| `scripts/evaluate.py` | `coffe/eval/episodic.py` |
| `scripts/evaluate_hypersigma.py` | `coffe/eval/hypersigma.py` |
| `scripts/pretrain.py` | `coffe/pretrain/loop.py` |
| `scripts/adapt_hypersigma.py` | `coffe/pretrain/hypersigma_adapt.py` |

`fit_pca_hypersigma.py`, `compile_results.py` and the two `download_*.sh` stay
as they are. The experiment drivers moved to `scripts/reproduce/` (the 5-seed
significance runner and its workers, the per-family `run_*` pairs, the eval
shell drivers) and the provenance/aggregation builders to `scripts/reports/`.

### Paper artifacts

The eight JSONs that provenance Tables 2 and 3 moved out of the runtime tree
into `results/`, with `results/README.md` mapping each to the table cells it
feeds. `experiments/` is now runtime-only (gitignored, `_example/` excepted) and
still resolves existing run directories by name, unchanged.

| Before | After |
|---|---|
| `experiments/aggregated_results.json` | `results/aggregated_results.json` |
| `experiments/experiment_metadata.json` | `results/experiment_metadata.json` |
| `experiments/mft_faithful_results.json` | `results/mft_faithful_results.json` |
| `experiments/significance_report.json` | `results/significance_report.json` |
| `experiments/significance_report copy.json` | `results/significance_report_enhanced_v1.json` |
| `experiments/_report_raw.json` | `results/_report_raw.json` |
| `experiments/hypersigma_native_sem_pca100_report.json` | `results/hypersigma_native_sem_pca100_report.json` |
| `experiments/gathered_results.json` | `results/gathered_results.json` |
| `configs/eval/hypersigma_houston.yaml` | `configs/hypersigma/houston_eval.yaml` |
| `tests/equivalence/fixtures/*.pth.fixture` | `tests/equivalence/fixtures/*.pth` |

The `copy` file's rescue (it is the only source of the across-seed std for the
12 HSI+LiDAR SimMIM cells, including all three headline numbers) is recorded in
`docs/refactor/AUDIT.md` §3 D8.

### Two defaults moved

`scripts/reproduce/run_eval.sh` and `run_eval_trento.sh` defaulted their output
directory to `results/eval/<scene>_<N>way_<K>shot`. `results/` is now the
tracked home of the paper-provenance JSONs, so that default became
`experiments/eval/…` (gitignored, like every other run output). It changes where
a bare invocation *writes*, never what it computes, and both scripts still take
an explicit output directory as a positional argument.

The second is the writer side of the artifact move above: the `--output`
fallbacks of `scripts/reports/aggregate_experiment_results.py`,
`build_experiment_metadata.py` and `compile_mft_faithful_results.py` now default
to `results/<name>.json` instead of `experiments/<name>.json`, so a regeneration
lands where the file it regenerates actually lives. `gather_native_pca100_raw.py`,
`build_native_pca100_report.py` and `gather_requested_results.py` take no
arguments at all; their hard-coded output paths moved with them.

### Packaging

`pyproject.toml` gained `[project]` (name `coffe`, version `0.9.0`, MIT,
dependencies copied unchanged from `requirements.txt`) and installs
`coffe*` plus `third_party*`, so the vendored HyperSIGMA sources stay importable
as `third_party.HyperSIGMA…` from any working directory with no vendored file
edited. `.gitignore` lost the venv-layout `lib/`/`lib64/`/`parts/`/`eggs/` rules
that used to shadow the real `lib/` source package (PAPER_CANON §8 D5), and now
tracks the equivalence checkpoint fixtures explicitly.

## Phase 6 — release quality (2026-09-02)

Zero behaviour change: the equivalence harness reports every fingerprint
IDENTICAL, re-checked with its tolerances forced to zero (the only residual
deltas were the goldens' own 8-decimal storage rounding, ~4e-9).

### Tooling: black + isort + flake8 -> ruff

One tool, one config block in `pyproject.toml`: line length 100,
`target-version = "py311"`, `select = ["E","F","W","I","UP","B","SIM","RUF"]`,
`third_party`/`archive`/`experiments`/`results`/`notebooks` excluded. `ruff
format` reformatted 89 files and `ruff check --fix` applied 683 safe fixes
(import sorting, PEP 585/604 annotations, unused imports, `dict.get(k, None)`
-> `dict.get(k)`); the remaining 111 findings were resolved by hand. **No
unsafe fix was applied.**

Three rules are switched off repo-wide, each because the "fix" would be a
behaviour edit or a documentation loss (reasons inline in `pyproject.toml`):
`B905` (`zip(strict=)` would turn silent truncation into an exception),
`RUF046` (`int(round(x))` is not a no-op when `x` is a NumPy scalar), and
`RUF059` (`B, N, D = x.shape` documents the tensor layout even where a name is
unused). `E402` is per-file-ignored for `scripts/`, `tests/` and `tools/`,
whose entry points insert the repo root on `sys.path` before importing
`coffe`. Six `# noqa` comments carry their reason at the site (`F822` for
`coffe/compat.py`'s lazily-served aliases, `SIM108` for the three fusion
branches whose per-branch shape comments a ternary would drop, `B018` for the
two `pytest.raises` attribute probes, `SIM115` for the log handle a child
process owns, `RUF022` for two grouped `__all__` lists).

### Dependencies pruned to the measured import set (PAPER_CANON D6)

`requires-python` is now **`>=3.11`** and the floors are the versions this
release was developed and verified against (`docs/refactor/ENV.md`) rather than
a compatibility survey — Nikola's decision at the phase-6 start, since NumPy
2.4.4 itself requires 3.11 and nothing older has ever been run against the
paper's numbers. The README badges and the environment line in Quick start
follow.

| | |
|---|---|
| Runtime | `torch`, `numpy`, `scipy`, `scikit-learn`, `PyYAML`, `omegaconf`, `matplotlib`, `seaborn`, `tqdm`, plus `einops` + `timm` (imported by the vendored HyperSIGMA ViT sources) |
| Dropped | `torchvision` (its only consumer died in phase 3), `hydra-core`, `h5py`, `scikit-image`, `spectral`, `rasterio`, `wandb` — none is imported anywhere in `coffe/`, `scripts/`, `tests/` or the notebooks |
| Moved to extras | `tensorboard` (the trainer already warns and continues without it), `pandas` + a Jupyter kernel (`notebooks` extra, for `compare.ipynb`), `pytest`/`pytest-cov`/`ruff`/`mypy` (`dev`) |
| Kept but unused by the package | none |

`mmengine` is **not** a dependency: the two vendored modules import it inside a
`try/except ImportError`.

`requirements.txt` is now a one-line mirror of `pyproject.toml` (`-e .[dev]`),
so the dependency set has exactly one source of truth. It was kept rather than
deleted because the README documents it as an install path.

### Public API

`coffe/__init__.py` exports a curated surface, still lazily so that `import
coffe` does not pull in torch (verified): `CoFFE`, `MFTOriginal`,
`HyperSIGMAFewShot`, the four runner entry points (`run_pretrain`,
`run_adapt_hypersigma`, `run_evaluation`, `run_hypersigma_evaluation`) and
`__version__`. `coffe.runners` re-exports the same four run functions beside
the `ExperimentLogger` / `PretrainExperiment` / `EvalRun` classes. Nothing was
removed, so every existing `from coffe.x.y import z` in the notebooks still
works.

Type hints were added to the 45 public functions and classes that lacked them,
and docstrings to the paper-relevant part of the API (Eq. 1's union-mask loss,
the N-way NCM prototypes, the backbone-native vs patch-native regimes, why λ is
inert on the HyperSIGMA route).

### mypy: adopted, lenient, partially enforced

`[tool.mypy]` runs over `coffe/` with `ignore_missing_imports`, no strict mode,
and `third_party` neither checked nor followed. `mypy` exits 0. Of the 55
modules, **39 are checked and clean**; the 16 listed in the overrides block are
not enforced (175 findings, all typing friction — `register_buffer` attributes
typed as `Module`, `config: dict` values arriving as `object`, ndarray/Tensor
swaps). Silencing those would mean casts inside forward passes and the training
loop, which this phase may not touch. 19 findings *were* fixed where the fix
was an annotation (`list[nn.Module]`, `dict[int, list[int]]`, a `cast` on
`OmegaConf.to_container`) plus four `# type: ignore[...]` with reasons.

### Consistency

- `print` stays in exactly one place — `coffe/eval/episodic.py`'s results table
  and aggregate banner, which are the CLI's user-facing report; a timestamped
  log prefix would break the table, and the docstring now says so. Everything
  else in the package logs.
- One seeding utility (`coffe.utils.seed.set_seed`) was already the only one;
  its function-local import in the evaluator was hoisted to module level.
- No bare `except`, no commented-out code, no `%`/`.format` string building
  outside logging format specifiers: checked, nothing to change.
- `os.path` in the dataset loaders was **left alone** (see "What deliberately
  did not change").

### Config headers

All 45 configs now carry a `RUNTIME` line citing PAPER_CANON §3's published
timing and this config's own epoch count. The eight HyperSIGMA adaptation
configs also carry a `REPRODUCES` block naming the Table 3 cell — including the four that
reproduce **no** published cell, which now say so and why. `base.yaml` needed
no decision: PAPER_CANON D4 records it as deleted in phase 3.

### Gate addendum: the report scripts got a CLI, and one reader was fixed

Applied at the phase-6 gate (2026-09-03).

**Four scripts that used to take no arguments now have an `argparse` front end**
with `--out` and `--force`: `scripts/compile_results.py`,
`scripts/reports/build_native_pca100_report.py`,
`scripts/reports/gather_native_pca100_raw.py` and
`scripts/reports/gather_requested_results.py` (the last also gained a `main()`
and a `__main__` guard — its pipeline used to write at import time). Each
refuses to overwrite an existing `--out` unless `--force` is passed, and the
check runs before any work, so a mistaken invocation costs nothing. This closes
the hazard that regenerated `docs/presentation/RESULTS.json` during routine CLI
smoke in both phase 5 and phase 6.

**Behaviour note:** refreshing one of these artifacts now requires
`--force` (or `--out PATH` to write elsewhere). That is the point.

**`scripts/compile_results.py` labelled HSI-only significance runs as
HSI+LiDAR.** It classified modality by the substring `no_lidar` alone, but the
5-seed significance runs are named `<scene>_hsi_only_<variant>_seed<s>` and
`<scene>_enhanced_mae_hsi_only_seed<s>`. Both conventions are frozen on disk
(PAPER_CANON §7.3), so `HSI_ONLY_MARKERS` now holds both and the reader accepts
either. Re-running the compiler to a scratch path shows 117 entries sourced
from an HSI-only run directory and **0** mislabelled (previously the
`_hsi_only` ones were all wrong). No paper number is affected: Table 2's means
come from the `*_no_lidar` dirs, which the old substring did match, and the ±
column is aggregated by `scripts/reports/aggregate_significance.py`, which
reads the declared group structure instead of sniffing directory names.

`tests/integration/test_compile_results.py` pins both: the two naming conventions, and that
`--help` does no work while an existing `--out` is refused and left
byte-identical.

### Two side effects of the tooling, recorded

- **ruff formats Python code blocks inside Markdown**, so
  `docs/PRETRAINING.md`'s load-a-checkpoint example was restyled (wrapped
  arguments, magic trailing comma). Prose is untouched; the example runs the
  same.
- **Live `file.py:NNN` citations were re-anchored** after the format pass moved
  code: three in `PAPER_CANON.md` (`compile_results.py:45`, `:75-76`, `:72,124`,
  `scripts/reports/build_experiment_metadata.py:121-123`,
  `sig_significance_config.py:45` and `:66-76`), seven in
  `tests/equivalence/_harness.py`, one in `tests/equivalence/test_equivalence.py`.
  Each new anchor was asserted to land on the line it claims. Three of them
  still pointed into `scripts/pretrain.py` from before phase 5 moved that body
  to `coffe/pretrain/loop.py`. Editing PAPER_CANON is normally out of scope; the
  edits here are citation line numbers only — no verdict, constant or table
  value was touched.

### One documentation error corrected

`docs/EVAL_PROTOCOL.md` still told readers the CLI default was `cosine`. It has
been `euclidean` since the phase-4 gate. Fixed.

`scripts/reproduce/README.md`'s Table 3 coverage row for `11x11 joint+SEM` said
the three cells reproduce from `configs/hypersigma/<scene>_patchnative_joint_sem.yaml`.
They do not: all three published runs used the **100-band** spatial front-end,
and the committed Trento/MUUFL configs are 3-band. Corrected there, and each
affected config's own header now states its true relationship to the cell.

---

## Phase 7 — test suite (2026-09-03)

Nothing in `coffe/`, `scripts/` or `configs/` changed. This phase only added
tests and moved existing ones, so **no computed number could move**; the
equivalence harness confirms it.

### Structure

`tests/` is now three layers. `tests/equivalence/` keeps its contents; the
only edit inside it is two comment-only path references in `_harness.py`:

| Layer | Meaning |
|---|---|
| `tests/unit/` | one module under test, synthetic tensors, no disk, no training loop |
| `tests/integration/` | several modules composed — dataset → sampler → model, the pretrain/adapt loops, the CLIs, the report scripts |
| `tests/equivalence/` | the behaviour-equivalence harness (no behavioural change) |

Nine surviving tests were `git mv`d into place with no change of substance:

| Old | New |
|---|---|
| `tests/test_models.py` | `tests/unit/test_tokenizers.py` |
| `tests/test_spatial_weights.py` | `tests/unit/test_spatial_weights.py` |
| `tests/test_compat.py` | `tests/unit/test_compat.py` |
| `tests/test_mft_original_shapes.py` | `tests/unit/test_mft_original_shapes.py` |
| `tests/test_hypersigma_shapes.py` | `tests/unit/test_hypersigma_shapes.py` |
| `tests/test_hypersigma_native_shapes.py` | `tests/unit/test_hypersigma_native_shapes.py` |
| `tests/test_data.py` | `tests/integration/test_episode_sampler.py` |
| `tests/test_pretrain_simmim.py` | `tests/integration/test_pretrain_simmim.py` |
| `tests/test_compile_results.py` | `tests/integration/test_compile_results.py` |

`test_pretrain_simmim.py` also lost its `sys.path` bootstrap, its `print()`
reporting and its `main()`/`__main__` runner — pytest is the only runner now.
Its assertions are byte-identical.

`tests/conftest.py` is new and holds two things: the paper's §5 dataset table
written out as a literal (the constant the code is checked *against*), and a
session-scoped fixture that writes the three shape-faithful synthetic
mini-scenes. The scene builders are **reused from
`tests/equivalence/_harness.py`** rather than re-implemented, so the test tree
has exactly one description of the on-disk scene layout.

### What the new tests pin

| File | Canon clause |
|---|---|
| `unit/test_masking_loss.py` | §3 Eq. 1 — exact token-mask counts, band-mask rates, the mask **union** (captured with forward hooks on both masking submodules), the masked-mean reduction, and the mean-one Gaussian centre weight |
| `unit/test_ncm_protocol.py` | §4 — assignments equal `sklearn.neighbors.NearestCentroid`, prototype = mean of the K = 5 supports, Euclidean the default everywhere, cosine still selectable. Also splits `_DEFAULT_ARGS` into the defaults that are the protocol and the five that contradict it (`docs/refactor/LOG.md` S4) |
| `unit/test_model_contracts.py` | §2 — forward shapes for all three dataset specs, 121 + 1 tokens, HSI-only sizing, pooled feature dim 128, Houston parameter count **579,328** (exactly) |
| `unit/test_hypersigma_contracts.py` | §1/§6 — the two input regimes, the three Table 3 feature widths, and the `requires_grad` partition: no adaptation may unfreeze a released ViT-Base block |
| `unit/test_config_parity.py` | §1/§2/§4/§9 — every committed config's arch and per-cell mask rates, including D19's Houston exception; the reproduce pipeline's K = 5 / 100 queries / 1000 episodes / euclidean; the seeds `[42, 123, 456, 789, 1011]` |
| `unit/test_checkpoint_keys.py` | §7.2 — the live `fix_state_dict_keys` rules, and that a real pretrain state_dict leaves no eval parameter randomly initialised |
| `unit/test_eval_runner.py` | §7.3 — a frozen `model.name: "mft_cpea"` config still seeds the right architecture, with one `DeprecationWarning`; checkpoint lookup by epoch |
| `integration/test_eval_protocol.py` | §4/§5 — N-way = 15/6/11, five supports per class, class-balanced queries |
| `integration/test_data_plumbing.py` | §5 — 11×11 patches, per-band min-max to [0, 1] asserted against the un-normalised tensors, PCA 144→100, linear resample 63/64→100. Also pins the one §5 clause the code does **not** implement: the paper's "border-padded at edges" happened upstream in the MFT data preparation, and this repo's (non-paper-path) raw-image loader drops border pixels instead — see `docs/refactor/LOG.md` S3 |
| `integration/test_hypersigma_eval.py` | §6/§8 D18 — the Table 3 evaluator end to end, and that `distance_metric` selects a *label*: both blocks are computed either way |
| `integration/test_hypersigma_adapt.py` | §1/§3 — one label-free adaptation epoch per mode, and no gradient reaching a released body after a real backward |
| `integration/test_cli.py` | `--help` on all 21 argparse entry points (leaving `docs/presentation/`, `results/` and `experiments/` unchanged, proved by a path+size+mtime digest taken before and after the sweep), plus pretrain → checkpoint → evaluate → `results.json` for both routes |

### Marker discipline

`slow` marks the 21 new tests that build a 180 M-parameter ViT body, or that
shell out to `scripts/pretrain.py` + `scripts/evaluate.py` (23 in the suite,
counting the two pre-existing equivalence ones). The 22 `--help`-driving tests
are *not* marked `slow`: the 21 per-script ones take under a second each, and
the 22nd (which re-runs all 21 subprocesses to digest-check the frozen artifact
trees) takes ~20 s.

Nothing new needs `data` or `gpu`. Verified rather than asserted: run from a
working directory where `data/raw/` and `checkpoints/` are unreachable, the
suite is **516 passed, 8 skipped, 0 failed** — the two extra skips being
exactly the pre-existing checkpoint-gated tests
(`tests/unit/test_hypersigma_native_shapes.py`), which keep their `skipif`.
Both the datasets and the HyperSIGMA checkpoints *are* present on the dev
machine, so an ordinary run does not exercise this; the separate run above is
what backs the claim.

### Counts

| | Before | After |
|---|---|---|
| tests collected (`-m "not gpu and not data"`) | 134 | 525 |
| wall time, CPU | 4:23 | 3:21 - 4:08 |
| line coverage of `coffe/` | 52 % | 69 % |

The wall times are not a controlled comparison: five measurements of the new
suite spread over 3:21-4:08 on the same machine (and the `not slow` selection
over 2:06-4:12, depending on what else was running), while the 4:23 baseline
was the first, cold run of its session. Read them as "same order, still a few
minutes", not as a speed-up.

Two paper-path modules went from **0 %** to covered: `coffe/eval/hypersigma.py`
(74 %) — the module every Table 3 cell came through — and
`coffe/pretrain/hypersigma_adapt.py` (81 %), the label-free adaptation loop.

## Compatibility guarantees

**1. Checkpoints load unchanged.** No `nn.Module` attribute name was renamed, so
state_dict keys are identical (PAPER_CANON §7.2). A pre-rename checkpoint
fixture is committed under `tests/equivalence/fixtures/` and a test loads it
through the renamed classes on every run.

**2. Frozen experiment trees still read.** `coffe/compat.py` maps the retired
vocabulary and emits one `DeprecationWarning` per value, naming the artifact it
came from:

```python
from coffe_compat import normalize_model_name, normalize_objective

normalize_model_name("mft_cpea")  # -> "coffe"
normalize_objective("enhanced")  # -> "simmim"
normalize_variant("spatial")  # -> "simmim_token"
normalize_model_type("MFTCPEACosine")  # -> "CoFFE"
```

It is wired into every reader of a frozen artifact: the model dispatch in
`scripts/pretrain.py` and `scripts/evaluate.py`, the architecture auto-load in
`coffe/runners/eval_runner.py`, `scripts/compile_results.py`, and
`scripts/reports/build_experiment_metadata.py`. Writers emit canonical values only,
with one disclosed exception: `scripts/reports/aggregate_significance.py` reproduces the
significance experiment's own `group`/`variant` directory-name components (see
the table below).

**3. Old imports still work.** The retired class names still resolve from the
packages they used to live in, and from the compat module, each with a
deprecation warning:

```python
from models import MFTCPEACosine  # -> models.coffe.CoFFE
from models import MFTOriginalCosine  # -> MFTOriginal
from models.hypersigma import HyperSIGMACosine  # -> HyperSIGMAFewShot
from pretrain import EnhancedMaskedSpectralSpatialModel  # -> SimMIMPretrainModel
from coffe_compat import MFTCPEACosine  # same object
```

The alias *is* the canonical class, so `isinstance` checks and checkpoint loads
behave identically either way. `tests/unit/test_compat.py` (25 tests) pins all three guarantees, and
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
| `coffe/compat.py`, `tests/unit/test_compat.py` | the alias table itself, and its tests |
| `tests/unit/test_eval_runner.py` | its fixture is a frozen-vocabulary `pretrain_config.yaml`, because that is what the runner must keep reading (PAPER_CANON §7.3) |
| `tests/unit/test_config_parity.py` | asserts the retired names are **absent** from the committed configs — it names them to forbid them |
| `tests/equivalence/test_equivalence.py`'s `test_g1_legacy_vocabulary_config_trains_identically` | drives a frozen-vocabulary config end to end on purpose |
| `.claude/agents/*.md`, `.claude/skills/*` | the refactor tooling's own specification of the retirement (same category as `docs/refactor/`) |
| `_LEGACY_ALIASES` in `coffe/models/__init__.py`, `coffe/models/hypersigma/__init__.py`, `coffe/pretrain/__init__.py` | the package-level import shims, which delegate to that table |
| `tools/refactor/apply_renames.py`, `tools/refactor/build_manifest.py` | the rename table and the audit ledger — they exist to record old→new |
| `CHANGES.md`, `PAPER_CANON.md`, `docs/refactor/*`, `CLAUDE.md`, `WORKFLOW.md` | documents about the retirement |
| on-disk-name literals (`_ENHANCED_CANONICAL`, dir-name substring tests, `paths:` values, experiment names in notebook parameter cells) | they name files that exist |
| stored notebook **outputs** | execution records of runs made before the rename |
| `scripts/reports/aggregate_significance.py`'s emitted `group` / `variant` keys | they reproduce the significance experiment's own directory-name components and the frozen `significance_report.json` schema |
| `experiments/**` | frozen run artifacts |
| `docs/EVAL_PROTOCOL.md`'s legacy-config paragraph, `results/README.md`'s on-disk-name list | they document what the readers accept and what the frozen artifacts contain |
| the `normalize_model_name` call sites' comments (`coffe/pretrain/loop.py`) | they name the frozen value being normalised |

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
  `scripts/evaluate_hypersigma.py`, `coffe/runners/eval_runner.py`, and the `CoFFE` /
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
- `os.path` in the dataset loaders (phase 6). Converting those to `pathlib`
  would touch the code that decides which files a dataset reads, for no
  behavioural gain; `pathlib` is used in the signatures phase 6 did touch.
- The 21 one-line dataset property overrides (`num_classes`, `hsi_channels`,
  `aux_channels`) have no docstrings: the contract is documented on
  `MultimodalEODataset` and each subclass's docstring carries its Table 1
  numbers. Adding 21 restatements would be noise.

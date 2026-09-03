# `results/` — the JSONs behind the paper's tables

These eight files are **frozen artifacts**: they are the record of the runs that
produced Tables 2 and 3, exactly as those runs wrote them. They are never
regenerated to "fix" a number. `PAPER_CANON.md` §6 holds the published values;
`docs/refactor/AUDIT.md` §3 (D1, D2) holds the cell-by-cell provenance trace
this table summarises.

Until phase 5 these lived in `experiments/`, mixed in with the runtime output
tree. They moved here so the runtime tree can stay gitignored.

## Which file feeds which paper cell

| File | Feeds | Coverage |
|---|---|---|
| `aggregated_results.json` | **Table 2** OA means (CoFFE rows) and 3 **Table 3** cells | the CoFFE means; see `docs/refactor/AUDIT.md` §3 D1 for the per-cell source run |
| `experiment_metadata.json` | **Table 2** means — per-run distilled metadata | second source for the same cells |
| `mft_faithful_results.json` | **Table 2** MFT-control rows (SimMIM token, MAE) | all 6 MFT means, epoch-950 checkpoints |
| `significance_report.json` | **Table 2** ± (across-seed std, 5 seeds) | 18 of 30 cells: groups `hsi_only`, `enhanced_mae`, `mft_mae`, `mft_spatial` |
| `significance_report_enhanced_v1.json` | **Table 2** ± for the HSI+LiDAR SimMIM cells | the remaining 12 cells, **including all three headline CoFFE SimMIM-token numbers** |
| `_report_raw.json` | **Table 3** — verbatim configs / metadata / results | 20 of 24 cells, exact mean + CI match |
| `hypersigma_native_sem_pca100_report.json` | **Table 3** — the aggregated native-SEM / PCA-100 report | aggregation over `_report_raw.json` + `_analysis/` |
| `gathered_results.json` | supporting compilation for the paper write-up | output of `scripts/reports/gather_requested_results.py` |

### Two significance files, not one

`significance_report.json`'s `groups` list omits the `enhanced` (HSI+LiDAR
SimMIM) group; that group's std lives only in
`significance_report_enhanced_v1.json` (called `significance_report copy.json`
before phase 5 — it looks like a stray duplicate and is not: deleting it would
orphan 12 Table 2 ± values). The two files also use different cell-key schemas:
`houston/spatial` in the `_enhanced_v1` file, `hsi_only/houston/spatial` in the
other. See `docs/refactor/AUDIT.md` §3 D8.

### How a mean and a ± differ

The **mean** of a Table 2 cell comes from a single canonical run, evaluated at
a mid-schedule checkpoint — epoch 950 on Houston and 975 on Trento/MUUFL for the
CoFFE cells, with one exception (Houston SimMIM band HSI+LiDAR, epoch 800), and
epoch 950 for all six MFT cells including Trento and MUUFL. Not a
model-selection decision: those epochs are what a checkpoint sort that ordered
filenames as **strings** returned (`"...950" > "...1500"`), and the sort is
numeric since the phase-7 gate, so the epoch has to be passed explicitly now —
PAPER_CANON §8 D17. (D17 also records what the artifacts cannot settle: explicit `checkpoint=` paths were sometimes used, so no single run can be attributed either way from disk.) `AUDIT.md` §3 D1
lists the checkpoint of every cell. The **±** comes
from a *separate* 5-seed experiment pretrained fresh to 700 epochs
(`scripts/reproduce/sig_significance_config.py`, seeds `[42, 123, 456, 789, 1011]`)
— it is an across-seed std, not a CI over the mean run's episodes. Table 3's ±
is a within-run 95% CI over episodes. `PAPER_CANON.md` §8 D20 records this
composition.

### Legacy vocabulary inside

These files were written before the paper's naming was settled, so their keys
carry retired vocabulary (`mft_cpea`, `enhanced`, `spatial`, `spectral`, `both`,
and run/dir names like `houston_enhanced_spatial_run1`). That is deliberate and
frozen: readers translate through `coffe/compat.py`, writers emit canonical
names only (PAPER_CANON §7.3).

## Regenerating (don't, unless you mean it)

The builders under `scripts/reports/` write these paths. They walk
`experiments/<run>/…` on a machine that has the run trees, so they only work
where those runs exist; on a fresh clone they have nothing to read.

**Four of them used to be a hazard, and are not any more.**
`scripts/compile_results.py`, `scripts/reports/build_native_pca100_report.py`,
`scripts/reports/gather_native_pca100_raw.py` and
`scripts/reports/gather_requested_results.py` took **no command-line arguments**:
running them with *anything*, `--help` included, executed the full pipeline and
overwrote their committed output in place — which happened twice during the
cleanup. Since the phase-6 gate all four take `--out PATH` and `--force`, and
refuse to overwrite an existing output file without `--force`
(`SystemExit: refusing to overwrite …`, checked before any work is done).
`--help` is now just `--help`.

Regenerating one still needs the run trees, so on a fresh clone it has nothing
to read. Write somewhere else while you experiment:

```bash
python scripts/compile_results.py --out /tmp/RESULTS.json
```

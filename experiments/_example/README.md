# _example

Reference layout for an experiment directory. **Not a real run** — the
metadata/results values are placeholders. Every real run produced by
`coffe.runners.pretrain_runner.run_pretrain` (or `notebooks/pretrain.ipynb`) lives
under `experiments/<your_name>/` with the same structure.

Files you'll find in a real experiment:

- `README.md` — name + short description (you write at start of the run)
- `pretrain_config.yaml` — frozen copy of the training config
- `pretrain_metadata.json` — timestamps, git SHA, final/best loss, wallclock
- `pretrain.log` — full training log
- `checkpoints/checkpoint_epoch_*.pth` — saved encoder weights
- `evaluations/<eval_name>/` — one subdirectory per evaluation run
  - `eval_config.json` — every parameter passed to the evaluator
  - `eval_metadata.json` — timestamps, git SHA, OA/AA/Kappa summary
  - `results.json` — full results including per-class accuracy
  - `eval.log`
  - `plots/` — confusion matrix, per-class accuracy, t-SNE, ...

Created: 2026-05-13
Git SHA: (example — no real run)

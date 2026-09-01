# `fixtures/` — pre-refactor checkpoints (committed)

These are real checkpoints produced on the **pre-refactor** code, by what was
then `scripts.pretrain_enhanced.run_pretrain` (now `scripts.pretrain`), on the
synthetic `houston_mini` scene with the headline `simmim_token` objective,
3 epochs. They are the G4 artefact:
`test_equivalence.py` loads each one into the current model class on every run,
so a broken `state_dict` key or a broken loader fails the harness
(PAPER_CANON §7.2).

They carry the same shape as the paper's own checkpoints — a `model_state_dict`
holding the *pretraining* model's `encoder.`-prefixed keys, which is what
`scripts/evaluate.py:fix_state_dict_keys` consumes. Optimiser and
scheduler state is stripped so the directory stays under its 5 MB budget.

## Why `.pth.fixture` and not `.pth`

The repo `.gitignore` has a blanket `*.pth` rule, so a `.pth` file here would
silently fail to commit — the same trap PAPER_CANON §8 D5 records for `lib/`.
`torch.load` ignores the extension.

**Proposed for phase 5:** add `!/tests/equivalence/fixtures/*.pth` to
`.gitignore` and rename these files back to `.pth`. Phase 2 is scoped out of
touching `.gitignore`, so the workaround stands for now.

## Regenerating

Only when re-baselining, and only from a pre-refactor tree:

```bash
python tests/equivalence/make_golden.py --only g3
```

This rewrites both the fixtures and `golden/g3_episodic_eval.json` together —
they must always be regenerated as a pair.

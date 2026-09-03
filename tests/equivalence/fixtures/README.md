# `fixtures/` — pre-refactor checkpoints (committed)

These are real checkpoints produced on the **pre-refactor** code, by what was
then `scripts.pretrain_enhanced.run_pretrain` (now `coffe.pretrain.loop`), on the
synthetic `houston_mini` scene with the headline `simmim_token` objective,
3 epochs. They are the G4 artefact:
`test_equivalence.py` loads each one into the current model class on every run,
so a broken `state_dict` key or a broken loader fails the harness
(PAPER_CANON §7.2).

They carry the same shape as the paper's own checkpoints — a `model_state_dict`
holding the *pretraining* model's `encoder.`-prefixed keys, which is what
`coffe/eval/episodic.py:fix_state_dict_keys` consumes. Optimiser and
scheduler state is stripped so the directory stays under its 5 MB budget.

## Why these `.pth` files are tracked

The repo `.gitignore` has a blanket `*.pth` rule, which would silently drop
them. Phase 5 added an explicit `!/tests/equivalence/fixtures/*.pth` negation,
so they are committed under their real extension. In phases 2–4 they carried a
`.pth.fixture` suffix instead — `.gitignore` was out of scope then — which is
the name the golden `g3_episodic_eval.json` recorded before phase 5. The bytes
never changed; `torch.load` ignores the extension either way.

## Regenerating

Only when re-baselining, and only from a pre-refactor tree:

```bash
python tests/equivalence/make_golden.py --only g3
```

This rewrites both the fixtures and `golden/g3_episodic_eval.json` together —
they must always be regenerated as a pair.

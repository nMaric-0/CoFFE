"""Frozen-encoder few-shot evaluation (Euclidean nearest-class-mean, PAPER_CANON §4).

- :mod:`coffe.eval.episodic`   — the CoFFE / MFT episodic evaluator
- :mod:`coffe.eval.hypersigma` — the HyperSIGMA evaluator

The feature-width ablation (``docs/feature_width_ablation.md``) adds three
modules that reuse those evaluators rather than duplicating them:

- :mod:`coffe.eval.feature_cache`  — encode a scene once, index it per episode
- :mod:`coffe.eval.dim_reduction`  — label-free linear compressions of the feature
- :mod:`coffe.eval.reduced_ncm`    — the NCM episode loop over a cached feature

For a classification map rather than a distribution over episodes,
:mod:`coffe.eval.prediction_map` runs the same classifier over the same cached
feature once per labelled sample (``docs/classification_maps.md``).

Both modules were the bodies of ``scripts/evaluate.py`` and
``scripts/evaluate_hypersigma.py`` before phase 5; the scripts are now thin
CLIs that call :func:`main` here. Nothing is imported eagerly: each module
pulls in torch.
"""

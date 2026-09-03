"""Frozen-encoder few-shot evaluation (Euclidean nearest-class-mean, PAPER_CANON §4).

- :mod:`coffe.eval.episodic`   — the CoFFE / MFT episodic evaluator
- :mod:`coffe.eval.hypersigma` — the HyperSIGMA evaluator

Both modules were the bodies of ``scripts/evaluate.py`` and
``scripts/evaluate_hypersigma.py`` before phase 5; the scripts are now thin
CLIs that call :func:`main` here. Nothing is imported eagerly: each module
pulls in torch.
"""

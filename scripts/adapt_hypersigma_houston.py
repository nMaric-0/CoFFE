#!/usr/bin/env python
"""Backward-compatibility shim.

The adapt entry-point is now ``scripts/adapt_hypersigma.py`` (the logic
is dataset-generic — the old ``_houston`` name was misleading). This
module re-exports the public API so existing imports
(``from scripts.adapt_hypersigma_houston import run_adapt``) and
notebooks keep working. Prefer importing from ``scripts.adapt_hypersigma``.
"""

from scripts.adapt_hypersigma import main, run_adapt  # noqa: F401

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    main(parser.parse_args())

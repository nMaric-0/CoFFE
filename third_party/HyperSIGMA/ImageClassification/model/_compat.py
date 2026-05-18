"""Compatibility shims for the vendored HyperSIGMA model code.

Provides minimal stand-ins for `mmengine`/`mmcv` utilities so the
encoder modules can be imported in environments where those packages
are not installed. We never actually use distributed training here.
"""


def get_dist_info():
    """Single-process fallback for ``mmengine.dist.get_dist_info``.

    Returns ``(rank, world_size)``.
    """

    return 0, 1

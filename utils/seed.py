"""Unified seed utility for reproducibility."""
import torch
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
        deterministic: If True, enforce deterministic CUDA operations
                       (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    logger.debug(f"Set seed={seed}, deterministic={deterministic}")

"""Wrappers for backbone models.

PretrainWrapper: Adds self-supervised pretraining capabilities
"""
from .pretrain_wrapper import PretrainWrapper, MAEDecoder, ContrastiveHead

__all__ = [
    "PretrainWrapper",
    "MAEDecoder",
    "ContrastiveHead",
]

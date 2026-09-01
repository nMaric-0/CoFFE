"""Evaluation metrics."""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy."""
    return (preds == labels).float().mean().item() * 100


def confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    return sklearn_cm(labels, preds)

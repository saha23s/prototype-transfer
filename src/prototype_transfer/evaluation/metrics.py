"""Evaluation metrics for multi-label bird classification (cmAP, AUC-ROC, top-1 acc).

Implements the official BirdSet benchmark evaluation standards:
  - cmAP: class-mean Average Precision (macro-averaged over classes with >= 1 positive sample)
  - Macro ROC-AUC: Area under the ROC curve averaged across valid classes
  - Top-1 Accuracy: Fraction of predictions where argmax matches an active label
  - F1 Macro: Macro-averaged F1 score at a specified decision threshold
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

logger = logging.getLogger("metrics")


def compute_cmap(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[int, float]]:
    """Compute Class-mean Average Precision (cmAP) across classes with positive samples.

    Args:
        y_true: Binary ground-truth matrix of shape (N, C).
        y_pred: Predicted probability matrix of shape (N, C).

    Returns:
        Tuple of (cmAP score, dict mapping class_idx -> AP score).
    """
    num_classes = y_true.shape[1]
    per_class_ap: dict[int, float] = {}

    for c in range(num_classes):
        # Only evaluate classes with at least one positive instance
        if np.sum(y_true[:, c]) > 0:
            # If all are positive, AP is 1.0
            if np.all(y_true[:, c] == 1):
                ap = 1.0
            else:
                try:
                    ap = float(average_precision_score(y_true[:, c], y_pred[:, c]))
                except ValueError:
                    ap = 0.0
            per_class_ap[c] = ap

    if per_class_ap:
        cmap = float(np.mean(list(per_class_ap.values())))
    else:
        cmap = 0.0

    return cmap, per_class_ap


def compute_macro_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[int, float]]:
    """Compute Macro ROC-AUC across classes with both positive and negative samples."""
    num_classes = y_true.shape[1]
    per_class_auc: dict[int, float] = {}

    for c in range(num_classes):
        pos = np.sum(y_true[:, c] == 1)
        neg = np.sum(y_true[:, c] == 0)
        if pos > 0 and neg > 0:
            try:
                auc = float(roc_auc_score(y_true[:, c], y_pred[:, c]))
            except ValueError:
                auc = 0.5
            per_class_auc[c] = auc

    if per_class_auc:
        macro_auc = float(np.mean(list(per_class_auc.values())))
    else:
        macro_auc = 0.5

    return macro_auc, per_class_auc


def compute_top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute top-1 accuracy (whether highest scoring class is a true positive)."""
    top1_indices = np.argmax(y_pred, axis=1)
    correct = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i, top1_indices[i]] == 1:
            correct += 1
    return float(correct / max(1, total))


def evaluate_predictions(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute full evaluation metric suite for multi-label predictions."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = (y_true > 0.5).astype(int)
    num_samples, num_classes = y_true.shape

    cmap, per_class_ap = compute_cmap(y_true, y_pred)
    macro_auc, per_class_auc = compute_macro_roc_auc(y_true, y_pred)
    top1_acc = compute_top1_accuracy(y_true, y_pred)

    y_bin = (y_pred >= threshold).astype(int)
    f1_macro = float(f1_score(y_true, y_bin, average="macro", zero_division=0))
    f1_micro = float(f1_score(y_true, y_bin, average="micro", zero_division=0))

    class_metrics: dict[str, Any] = {}
    for c in range(num_classes):
        name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
        class_metrics[name] = {
            "ap": per_class_ap.get(c, None),
            "auc": per_class_auc.get(c, None),
            "support": int(np.sum(y_true[:, c])),
            "pred_positive": int(np.sum(y_bin[:, c])),
        }

    return {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "cmAP": cmap,
        "macro_roc_auc": macro_auc,
        "top1_accuracy": top1_acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "threshold": threshold,
        "per_class": class_metrics,
    }

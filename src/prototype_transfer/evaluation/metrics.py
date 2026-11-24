"""Evaluation metrics for multi-label bird classification.

  - cmAP: class-mean Average Precision (macro-averaged over classes with >= 1 positive sample), as in the BirdSet benchmark
  - Macro ROC-AUC: area under the ROC curve averaged across valid classes
  - column-wise AP / AUC and the cluster-bootstrap resampler used by the canonical evaluation (evaluation/canonical.py)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


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


# ----------------------------------------------------------------------------- column-wise / bootstrap helpers
def auc_cols(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """ROC-AUC of every column (Mann-Whitney with average ranks). S [N, P] scores, Y [N, P] bool. NaN if a column has no pos or no neg."""
    n = len(S)
    npos = Y.sum(0).astype(float)
    nneg = n - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (rankdata(S, axis=0) * Y).sum(0) - npos * (npos + 1) / 2
        out = u / (npos * nneg)
    out[(npos == 0) | (nneg == 0)] = np.nan
    return out


def ap_cols(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array([average_precision_score(Y[:, j], S[:, j]) if Y[:, j].any() else np.nan for j in range(S.shape[1])])


def cluster_indices(groups: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[np.ndarray]:
    """n_boot resamples of whole clusters (sites / recordists), each as a concatenated clip-index array."""
    u = np.unique(groups)
    members = {g: np.nonzero(groups == g)[0] for g in u}
    return [np.concatenate([members[g] for g in rng.choice(u, len(u))]) for _ in range(n_boot)]

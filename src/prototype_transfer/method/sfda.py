"""Losses and pseudo-labels of published source-free domain adaptation (SFDA) baselines, multi-label form.

PyTorch port of the audio / multi-label code path of the benchmark of Boudiaf et al., "In Search for a Generalizable
Method for Source Free Domain Adaptation" (ICML 2023, arXiv 2302.06658): github.com/google-research/perch, tag
sfda-codebase-snapshot (commit 7412704), chirp/projects/sfda. Every species is an independent binary problem: for clip i
and species c, p_ic = sigmoid(logit_ic). All 21 HSN species are in the loss. Losses are averaged over species, then over
clips.

  tent             entropy minimisation (Wang et al., ICLR 2021)                                     methods/tent.py
  pseudo_label     BCE towards the species above a confidence threshold (Lee, 2013)                  methods/pseudo_label.py
  dropout_student  noisy student matching the clean soft predictions (after Xie et al., CVPR 2020)   methods/dropout_student.py
  notela           dropout_student with Laplacian-adjusted teacher labels, Eq. 3 of Boudiaf et al.   methods/notela.py
  shot             information maximisation + beta * cluster pseudo-labels (Liang et al., ICML 2020) methods/shot.py

BCE and the per-clip entropy are computed from logits (the reference computes them from probabilities with eps=1e-10);
the two agree except where a probability is within about eps of 0 or 1. This module is only math (no model, no data);
the optimisation loop is prototype_transfer.training.adapt and the CLI scripts/method/adapt_sfda.py.
Tests: tests/test_sfda.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# The authors' audio settings (configs/<method>.py, get_audio_config), fixed here before any run: Adam, no weight decay,
# normalisation-layer scale and shift as the only trainable parameters. `dropout` = student noise during the update.
# Their paper selected hyperparameters on their own High Sierras split (the region of HSN), with a different model.
METHODS: dict[str, dict] = {
    "tent": {"lr": 1e-5, "cosine": True, "dropout": True},
    "pseudo_label": {"lr": 1e-5, "cosine": False, "dropout": True, "threshold": 0.5},
    "dropout_student": {"lr": 1e-4, "cosine": False, "dropout": True, "alpha": 1.0},
    "notela": {"lr": 1e-4, "cosine": False, "dropout": True, "alpha": 1.0, "lam": 1.0, "knn": 5},
    "shot": {"lr": 1e-5, "cosine": True, "dropout": False, "beta": 0.0},
}
PER_EPOCH_LABELS = ("dropout_student", "notela", "shot")   # targets are recomputed on all clips before every epoch
EPS = 1e-10         # losses.py
TEACHER_EPS = 1e-8  # NOTELA.teacher_step
SHOT_EPS = 1e-6     # SHOT.compute_pseudo_label


# --------------------------------------------------------------------------------------------- losses
def binary_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(p) of p = sigmoid(logits), elementwise. [B, C] -> [B, C]."""
    p = torch.sigmoid(logits)
    return -(p * F.logsigmoid(logits) + (1 - p) * F.logsigmoid(-logits))


def tent_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean binary entropy of the predictions."""
    return binary_entropy(logits).mean()


def soft_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy to soft targets [B, C] (Dropout Student and NOTELA student step)."""
    return F.binary_cross_entropy_with_logits(logits, targets)


def pseudo_label_loss(logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """BCE towards 1 on the selected (clip, species) entries only, [B, C] 0/1.

    As in the reference, the confidence mask equals the pseudo-label, so there are no pseudo-negatives. Each clip is
    averaged over its selected species; a clip with none contributes 0 but still counts in the batch mean.
    """
    bce = F.binary_cross_entropy_with_logits(logits, selected, reduction="none")
    return ((selected * bce).sum(dim=1) / (selected.sum(dim=1) + EPS)).mean()


def shot_loss(logits: torch.Tensor, targets: torch.Tensor | None, beta: float) -> torch.Tensor:
    """mean H(p_ic) - mean_c H(mean_i p_ic) + beta * BCE(p, targets); the marginal is taken over the batch.

    targets (the cluster pseudo-labels) may be None when beta == 0.
    """
    q = torch.sigmoid(logits).mean(dim=0)                                          # [C]
    marginal = -(q * torch.log(q + EPS) + (1 - q) * torch.log((1 - q) + EPS)).mean()
    loss = binary_entropy(logits).mean() - marginal
    return loss if beta == 0 else loss + beta * soft_bce_loss(logits, targets)


# --------------------------------------------------------------------------------------------- pseudo-labels
def threshold_labels(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """[B, C] 0/1: species whose clean probability is above the threshold."""
    return (probs > threshold).float()


def sharpen(probs: torch.Tensor, alpha: float) -> torch.Tensor:
    """Dropout Student targets: p^(1/alpha) / (p^(1/alpha) + (1-p)^(1/alpha)). alpha = 1 returns p."""
    pos, neg = probs ** (1 / alpha), (1 - probs) ** (1 / alpha)
    return pos / (pos + neg)


def squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[N, M] squared Euclidean distances, in float64 (nearest neighbours are sensitive to cancellation in float32)."""
    a, b = a.double(), b.double()
    return (a * a).sum(1, keepdim=True) - 2 * a @ b.T + (b * b).sum(1)


def knn_affinity(features: torch.Tensor, knn: int) -> torch.Tensor:
    """[N, N] 0/1: row i marks the neighbours of clip i among its `knn` closest clips, the closest (itself) dropped.

    So every clip has knn - 1 neighbours, as in the reference (compute_nearest_neighbors).
    """
    k = min(len(features), knn)
    idx = squared_distances(features, features).topk(k, dim=1, largest=False).indices[:, 1:]   # [N, k-1]
    affinity = torch.zeros(len(features), len(features), dtype=torch.float32, device=features.device)
    return affinity.scatter_(1, idx, 1.0)


def notela_labels(probs: torch.Tensor, features: torch.Tensor, knn: int, lam: float, alpha: float) -> torch.Tensor:
    """NOTELA teacher step, Eq. 3 of Boudiaf et al., one binary problem per species.

    y_ic is proportional to p_ic^(1/alpha) * exp((lam/alpha) * mean_{j in M(i)} p_jc), and the same expression with
    1 - p for the negative state; M(i) are the MUTUAL nearest neighbours of clip i. A clip without any has y = p^(1/alpha)
    normalised.

    Args:
        probs: clean probabilities of all adaptation clips, [N, C].
        features: their clip embeddings, [N, D].
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    affinity = knn_affinity(features, knn)
    affinity = affinity * affinity.T
    degree = affinity.sum(dim=1, keepdim=True) + TEACHER_EPS                     # [N, 1]
    pos = probs ** (1 / alpha) * torch.exp(lam / alpha * (affinity @ probs) / degree)
    neg = (1 - probs) ** (1 / alpha) * torch.exp(lam / alpha * (affinity @ (1 - probs)) / degree)
    return pos / (pos + neg + TEACHER_EPS)


def shot_labels(probs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """SHOT cluster pseudo-labels, one two-cluster problem per species (reference SHOT.compute_pseudo_label).

    Centroids start as the means weighted by 1-p and p (Eq. 4 of Liang et al.), clips are assigned to the nearest
    (squared Euclidean, as in the reference), centroids are recomputed from the assignment and clips reassigned.

    Returns:
        [N, C] 0/1.
    """
    f = features.double()
    labels = []
    for p in probs.double().unbind(dim=1):
        weights = torch.stack([1 - p, p])                                        # [2, N]
        centroids = weights @ f / (weights.sum(1, keepdim=True) + SHOT_EPS)      # [2, D]
        assign = F.one_hot(squared_distances(f, centroids).argmin(1), 2).double().T
        centroids = assign @ f / (assign.sum(1, keepdim=True) + SHOT_EPS)
        labels.append(squared_distances(f, centroids).argmin(1))
    return torch.stack(labels, dim=1).float()


# --------------------------------------------------------------------------------------------- dispatch
def epoch_targets(method: str, probs: torch.Tensor, features: torch.Tensor, cfg: dict) -> torch.Tensor | None:
    """Targets of all adaptation clips for the coming epoch, from clean predictions; None if the method needs none."""
    if method == "dropout_student":
        return sharpen(probs, cfg["alpha"])
    if method == "notela":
        return notela_labels(probs, features, cfg["knn"], cfg["lam"], cfg["alpha"])
    if method == "shot" and cfg["beta"] != 0:
        return shot_labels(probs, features)
    return None


def loss_fn(method: str, logits: torch.Tensor, targets: torch.Tensor | None, cfg: dict) -> torch.Tensor:
    """The method's loss on one batch of student logits [B, C]."""
    if method == "tent":
        return tent_loss(logits)
    if method == "pseudo_label":
        return pseudo_label_loss(logits, targets)
    if method in ("dropout_student", "notela"):
        return soft_bce_loss(logits, targets)
    if method == "shot":
        return shot_loss(logits, targets, cfg["beta"])
    raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")

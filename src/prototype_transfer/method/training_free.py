"""Training-free, label-free baselines that act on the pretrained model's prototype activations.

The class score is p_s = sigmoid(sum_k w_sk * a_sk + b_s), with a_sk the max activation of prototype k of species s.
Both baselines are fitted from two activation arrays only (no labels): unlabeled soundscape clips of the training sites
(target) and focal-train clips (reference). The fitted adaptation is then frozen and applied to every evaluated clip,
focal and soundscape alike. The backbone and the prototypes are unchanged.

quantile   a' = Q_focal(F_target(a)) per prototype (quantile mapping / histogram matching). Exact zeros stay zero:
           activations are ReLU outputs, and mapping the tied zeros through the CDF would turn "no match" into evidence.
           Values outside the fitted range are clipped to the focal end points. Weights and bias are unchanged.
reweight   a heuristic of this repository: g_D = mean(top fraction of a) - median(a) in domain D, r = g_target / g_focal,
           w' = w * r, rescaled so that every species keeps its total weight. Activations and bias are unchanged, so
           per-prototype AUCs (and the failing-prototype count) are those of the pretrained model.

Activation arrays are [N, 21, 5]; weights are the non-negative [21, 5] last-layer weights; bias is [21].
The CLI is scripts/method/apply_training_free.py.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def class_probs(acts: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """[N, 21] class probabilities from prototype activations [N, 21, 5]."""
    logits = (acts * weights).sum(axis=-1) + bias
    return 1.0 / (1.0 + np.exp(-logits))


def fit_quantile_map(target: np.ndarray, reference: np.ndarray, n_quantiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-prototype quantile knots (target, reference), each [n_quantiles, 21, 5], at the same probability levels."""
    levels = np.linspace(0.0, 1.0, n_quantiles)
    return np.quantile(target, levels, axis=0), np.quantile(reference, levels, axis=0)


def apply_quantile_map(acts: np.ndarray, knots_target: np.ndarray, knots_reference: np.ndarray) -> np.ndarray:
    """a' = Q_reference(F_target(a)) per prototype by linear interpolation between the knots; zeros stay zero."""
    out = np.empty_like(acts)
    for s in range(acts.shape[1]):
        for k in range(acts.shape[2]):
            out[:, s, k] = np.interp(acts[:, s, k], knots_target[:, s, k], knots_reference[:, s, k])
    return np.where(acts == 0, 0.0, out).astype(acts.dtype)


def contrast(acts: np.ndarray, top_fraction: float) -> np.ndarray:
    """[21, 5] gap between the mean of the top `top_fraction` of each prototype's activations and their median."""
    n_top = max(1, int(round(top_fraction * len(acts))))
    return np.sort(acts, axis=0)[-n_top:].mean(axis=0) - np.median(acts, axis=0)


def reweight(weights: np.ndarray, target: np.ndarray, reference: np.ndarray, top_fraction: float) -> np.ndarray:
    """w' = w * (g_target / g_reference), rescaled per species to keep sum_k w_sk. Returns [21, 5].

    Raises:
        ValueError: if a prototype has no contrast on the reference clips, or a species gets no weight at all.
    """
    g_ref = contrast(reference, top_fraction)
    if (g_ref <= _EPS).any():
        raise ValueError(f"{int((g_ref <= _EPS).sum())} prototypes have no contrast on the reference clips; r is undefined")
    scaled = weights * (contrast(target, top_fraction) / g_ref)
    total = scaled.sum(axis=1, keepdims=True)
    if (total <= _EPS).any():
        raise ValueError("a species would get zero total weight")
    return (scaled * weights.sum(axis=1, keepdims=True) / total).astype(weights.dtype)

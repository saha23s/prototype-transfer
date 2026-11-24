"""Training losses that are not part of the (remote-code) model."""

from __future__ import annotations

import torch
import torch.nn as nn


def orthogonality_loss(head: nn.Module) -> torch.Tensor:
    """||P_c P_c^T - I||_F summed over classes, P_c = L2-normalised prototypes of class c.
    (The remote get_prototype_orthogonalities references a missing attribute, so we compute it here.)"""
    K = head.prototypes_per_class
    P = nn.functional.normalize(head.prototype_vectors.flatten(1), dim=-1).view(head.num_classes, K, -1)
    G = P @ P.transpose(1, 2) - torch.eye(K, device=P.device)
    return torch.linalg.matrix_norm(G).sum()

"""Training losses that are not part of the (remote-code) model."""

from __future__ import annotations

import torch
import torch.nn as nn


ORTHO_FORMS = ("frobenius", "paper")


def orthogonality_loss(head: nn.Module, form: str = "frobenius") -> torch.Tensor:
    """Within-class orthogonality of the prototypes, P_c = L2-normalised prototypes of class c (C classes, J each).
    (The remote get_prototype_orthogonalities references a missing attribute, so we compute it here.)

    form "frobenius" (default, used by all runs so far): sum_c ||P_c P_c^T - I||_F. Its gradient does not vanish as the
        prototypes approach orthogonality.
    form "paper": sum_c ||P_c P_c^T - I||_F^2 / (C J^2), Eq. 4 of the later arXiv version of AudioProtoPNet
        (Heinrich et al., arXiv 2404.10420). Its gradient vanishes at orthogonality.

    Raises:
        ValueError: for an unknown form.
    """
    if form not in ORTHO_FORMS:
        raise ValueError(f"unknown orthogonality form {form!r}; expected one of {ORTHO_FORMS}")
    K = head.prototypes_per_class
    P = nn.functional.normalize(head.prototype_vectors.flatten(1), dim=-1).view(head.num_classes, K, -1)
    G = P @ P.transpose(1, 2) - torch.eye(K, device=P.device, dtype=P.dtype)
    if form == "paper":
        return G.pow(2).sum() / (head.num_classes * K ** 2)
    return torch.linalg.matrix_norm(G).sum()

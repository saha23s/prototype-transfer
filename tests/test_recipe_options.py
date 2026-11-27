"""Plain-assert tests for the opt-in recipe options (orthogonality form).  Run: python tests/test_recipe_options.py"""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from prototype_transfer.training.losses import orthogonality_loss

C, J, D = 3, 5, 8


def _head(vectors: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(num_classes=C, prototypes_per_class=J, prototype_vectors=nn.Parameter(vectors.view(C * J, D, 1, 1)))


def _random_head(seed: int) -> SimpleNamespace:
    return _head(torch.randn(C * J, D, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))


def _gram_gaps(head: SimpleNamespace) -> list[np.ndarray]:
    v = head.prototype_vectors.detach().numpy().reshape(C * J, D)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return [v[c * J:(c + 1) * J] @ v[c * J:(c + 1) * J].T - np.eye(J) for c in range(C)]


def test_default_form_is_unchanged():
    head = _random_head(0)
    want = sum(np.linalg.norm(g) for g in _gram_gaps(head))   # unsquared Frobenius norm, summed over classes
    assert np.isclose(float(orthogonality_loss(head)), want) and np.isclose(float(orthogonality_loss(head, "frobenius")), want)


def test_paper_form_is_the_squared_norm_divided_by_classes_times_prototypes_squared():
    head = _random_head(1)
    want = sum((g ** 2).sum() for g in _gram_gaps(head)) / (C * J ** 2)   # Eq. 4 of Heinrich et al. (arXiv 2404.10420)
    assert np.isclose(float(orthogonality_loss(head, "paper")), want)


def test_paper_gradient_vanishes_near_orthogonality_and_the_default_does_not():
    basis = torch.eye(D, dtype=torch.float64)[:J].repeat(C, 1)   # exactly orthogonal within every class
    near = basis + 1e-6 * torch.randn(basis.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    grads = {}
    for form in ("frobenius", "paper"):
        head = _head(near.clone())
        grads[form] = float(torch.autograd.grad(orthogonality_loss(head, form), head.prototype_vectors)[0].norm())
    assert grads["paper"] < 1e-4 and grads["frobenius"] > 0.5, grads


def test_unknown_form_is_refused():
    try:
        orthogonality_loss(_random_head(3), "other")
        raise AssertionError("expected a refusal")
    except ValueError as e:
        assert "form" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)

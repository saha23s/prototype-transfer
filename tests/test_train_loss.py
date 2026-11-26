"""Plain-assert test of the loss scale and the non-finite gradient check in training/trainer.run_epoch.  Run: python tests/test_train_loss.py

The model's loss class comes from its remote code, so a stand-in with the same `reduction` argument and the same
default ("mean") is used here; no model download is needed. The expected loss is: supervised term summed over
classes and averaged over labeled clips, plus pseudo_w times the pseudo-label term reduced the same way.
"""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype_transfer.training.trainer import run_epoch

N_CLASSES, N_FEAT = 21, 4


class AsymmetricLossMultiLabel(nn.Module):   # looked up by run_epoch in the model's module (this one)
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        return {"mean": loss.mean(), "sum": loss.sum(), "none": loss}[self.reduction]


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(N_FEAT, N_CLASSES)

    def forward(self, input_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.linear(input_values))


def _batch(n: int, soft: bool = False) -> dict:
    t = torch.rand(n, N_CLASSES) if soft else (torch.rand(n, N_CLASSES) < 0.1).float()
    return {"spectrogram": torch.randn(n, N_FEAT), "target": t, "audio_path": [str(i) for i in range(n)]}


def _per_clip_bce(model: nn.Module, b: dict) -> float:
    with torch.no_grad():
        loss = F.binary_cross_entropy_with_logits(model(b["spectrogram"]).logits, b["target"], reduction="none")
    return float(loss.sum(1).mean())


def _run(model: nn.Module, b: dict, pseudo: dict | None = None, pseudo_w: float = 0.0) -> float:
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    return run_epoch(model, [b], opt, None, "cpu", 0.0, iter([pseudo]) if pseudo else None, pseudo_w)


def test_supervised_loss_is_summed_over_classes_and_averaged_over_clips():
    torch.manual_seed(0)
    model = TinyModel()
    for n in (4, 32):   # the scale must not depend on the batch size
        b = _batch(n)
        got, want = _run(model, b), _per_clip_bce(model, b)
        assert np.isclose(got, want, rtol=1e-5), f"batch {n}: loss {got:.6g}, expected {want:.6g} (ratio {want / got:.1f})"


def test_pseudo_term_is_on_the_same_scale_as_the_supervised_term():
    torch.manual_seed(1)
    model = TinyModel()
    b, p = _batch(32), _batch(16, soft=True)
    got, want = _run(model, b, p, 0.2), _per_clip_bce(model, b) + 0.2 * _per_clip_bce(model, p)
    assert np.isclose(got, want, rtol=1e-5), f"loss {got:.6g}, expected {want:.6g}"

def test_a_non_finite_gradient_stops_before_the_optimizer_step():
    torch.manual_seed(2)
    for bad in (float("nan"), float("inf")):
        model = TinyModel()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.linear.weight.register_hook(lambda g: g * bad)          # the loss stays finite, only the gradient is bad
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        try:
            run_epoch(model, [_batch(4)], opt, None, "cpu", 0.0)
        except FloatingPointError as e:
            assert "non-finite gradient" in str(e), e
        else:
            raise AssertionError(f"a {bad} gradient must stop training")
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), "the weights were updated"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)

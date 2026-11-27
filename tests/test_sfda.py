"""Plain-assert tests of the source-free adaptation baselines (method/sfda.py, training/adapt.py). Run: python tests/test_sfda.py

Values are checked against hand calculations. The adaptation loop runs on a tiny Hugging Face ConvNeXt (random weights,
no download) wrapped like AudioProtoPNet: model.model.backbone, outputs .logits and .last_hidden_state.
The port was also compared numerically with the authors' JAX functions when it was written (see sfda.py for the commit).
"""

import copy
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from transformers import ConvNextConfig, ConvNextModel

from prototype_transfer.method import sfda
from prototype_transfer.training.adapt import (UnlabeledSoundscapes, adapt, add_stochastic_depth, norm_parameters,
                                               predict)

N_CLASSES, N_BLOCKS = 21, 5


# --------------------------------------------------------------------------------------------- math
def test_entropy_and_tent_loss_by_hand():
    logits = torch.tensor([[0.0, math.log(3.0)]])                      # p = 0.5, 0.75
    h = [math.log(2.0), -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))]
    assert torch.allclose(sfda.binary_entropy(logits), torch.tensor([h]))
    assert math.isclose(float(sfda.tent_loss(logits)), sum(h) / 2, rel_tol=1e-6)


def test_losses_stay_finite_with_finite_gradients_at_saturated_logits():
    for method, target in [("tent", None), ("pseudo_label", torch.tensor([[1.0, 0.0]])),
                           ("dropout_student", torch.tensor([[0.3, 0.7]])), ("shot", None)]:
        z = torch.tensor([[200.0, -200.0]], requires_grad=True)
        loss = sfda.loss_fn(method, z, target, sfda.METHODS[method])
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(z.grad).all(), method


def test_pseudo_label_loss_uses_confident_positives_only():
    logits = torch.tensor([[0.0, 5.0, -5.0], [1.0, 1.0, 1.0]], requires_grad=True)
    selected = sfda.threshold_labels(torch.tensor([[0.9, 0.8, 0.1], [0.2, 0.3, 0.4]]), 0.5)
    assert selected.tolist() == [[1, 1, 0], [0, 0, 0]]
    loss = sfda.pseudo_label_loss(logits, selected)
    first = (math.log(2.0) + math.log1p(math.exp(-5.0))) / 2           # averaged over the clip's two selected species
    assert math.isclose(float(loss), first / 2, rel_tol=1e-6)          # the empty clip counts in the batch mean
    loss.backward()
    assert logits.grad[0, 2] == 0 and (logits.grad[1] == 0).all()      # no pseudo-negatives
    assert (logits.grad[0, :2] < 0).all()                              # selected species are pushed up


def test_shot_loss_is_entropy_minus_marginal_entropy():
    z = torch.tensor([[20.0], [-20.0]])                                # confident and balanced: H = 0, H(mean) = log 2
    assert math.isclose(float(sfda.shot_loss(z, None, 0.0)), -math.log(2.0), rel_tol=1e-4)
    z = torch.tensor([[20.0], [20.0]])                                 # confident and one-sided: both 0
    assert abs(float(sfda.shot_loss(z, None, 0.0))) < 1e-4
    target = torch.tensor([[0.0], [0.0]])
    extra = float(sfda.shot_loss(z, target, 0.5) - sfda.shot_loss(z, None, 0.0))
    assert math.isclose(extra, 0.5 * float(sfda.soft_bce_loss(z, target)), rel_tol=1e-5)


def test_sharpen():
    p = torch.tensor([0.2, 0.5, 0.9])
    assert torch.allclose(sfda.sharpen(p, 1.0), p)
    assert torch.allclose(sfda.sharpen(p, 0.5), p ** 2 / (p ** 2 + (1 - p) ** 2))


def test_knn_affinity_drops_self_and_keeps_k_minus_one():
    x = torch.tensor([[0.0], [1.0], [3.0], [7.0]])
    a = sfda.knn_affinity(x, 3)                                        # two neighbours each, never itself
    assert a.tolist() == [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]]
    assert (a * a.T).sum(1).tolist() == [2, 2, 2, 0]                   # the outlier has no mutual neighbour


def test_notela_labels_by_hand():
    features = torch.tensor([[0.0], [1.0], [10.0]])                    # clips 0 and 1 are mutual; clip 2 has none
    probs = torch.tensor([[0.9], [0.6], [0.3]])
    y = sfda.notela_labels(probs, features, knn=2, lam=1.0, alpha=1.0)
    def eq3(p, q):                                                     # lambda = alpha = 1, one mutual neighbour q
        pos, neg = p * math.exp(q), (1 - p) * math.exp(1 - q)
        return pos / (pos + neg)
    assert torch.allclose(y[:, 0], torch.tensor([eq3(0.9, 0.6), eq3(0.6, 0.9), 0.3]), atol=1e-6)
    zero = sfda.notela_labels(probs, features, knn=2, lam=0.0, alpha=1.0)
    assert torch.allclose(zero, probs, atol=1e-6)                      # no Laplacian term: Dropout Student's labels


def test_shot_labels_split_two_clusters_per_species():
    features = torch.tensor([[0.0], [0.2], [5.0], [5.2]])
    probs = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
    assert sfda.shot_labels(probs, features).tolist() == [[0, 1], [0, 1], [1, 0], [1, 0]]


def test_epoch_targets():
    probs, feats = torch.rand(8, 3), torch.randn(8, 4)
    assert sfda.epoch_targets("tent", probs, feats, sfda.METHODS["tent"]) is None
    assert sfda.epoch_targets("shot", probs, feats, sfda.METHODS["shot"]) is None    # beta 0: pseudo-labels unused
    assert sfda.epoch_targets("shot", probs, feats, {"beta": 0.3}).shape == (8, 3)
    assert torch.allclose(sfda.epoch_targets("dropout_student", probs, feats, sfda.METHODS["dropout_student"]), probs)
    assert sfda.epoch_targets("notela", probs, feats, sfda.METHODS["notela"]).shape == (8, 3)


# --------------------------------------------------------------------------------------------- loop
class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = ConvNextConfig(num_channels=1, hidden_sizes=[8, 8, 8, 8], depths=[1, 1, 2, 1], drop_path_rate=0.0)
        self.model = nn.Module()
        self.model.backbone = ConvNextModel(cfg)
        self.head = nn.Linear(8, N_CLASSES)

    def forward(self, input_values: torch.Tensor) -> SimpleNamespace:
        hidden = self.model.backbone(input_values).last_hidden_state
        return SimpleNamespace(logits=self.head(hidden.amax(dim=(-2, -1))), last_hidden_state=hidden)


def tiny() -> tuple[TinyNet, TensorDataset]:
    torch.manual_seed(0)
    model = TinyNet()
    with torch.no_grad():
        for m in model.modules():   # a fresh ConvNeXt scales every block by 1e-6, so dropping a block would change nothing
            if getattr(m, "layer_scale_parameter", None) is not None:
                m.layer_scale_parameter.fill_(1.0)
    return model, TensorDataset(torch.randn(10, 1, 32, 64), torch.arange(10))


def run(model: nn.Module, data: TensorDataset, method: str, **over) -> list[dict]:
    cfg = {**sfda.METHODS[method], "lr": 1e-2, **over}
    return adapt(model, data, method, cfg, epochs=2, batch_size=4, drop_path=0.2, device="cpu", seed=0)


def test_every_method_changes_the_backbone_layernorms_and_nothing_else():
    for method in sfda.METHODS:
        model, data = tiny()
        before = copy.deepcopy(model.state_dict())
        over = {"threshold": 0.3} if method == "pseudo_label" else {"beta": 0.3} if method == "shot" else {}
        history = run(model, data, method, **over)
        assert len(history) == 2 and all(math.isfinite(r["loss"]) for r in history), method
        after = model.state_dict()
        assert after.keys() == before.keys(), "the added stochastic depth must not change the checkpoint keys"
        norm_ids = {id(p) for p in norm_parameters(model)}
        norms = {n for n, p in model.named_parameters() if id(p) in norm_ids}
        changed = {k for k in before if not torch.equal(before[k], after[k])}
        assert changed and changed <= norms, f"{method}: changed {changed - norms}"
        assert changed == norms, f"{method}: not updated {norms - changed}"
        assert not model.training, "the adapted model is returned frozen in eval mode"


def test_the_unused_pooler_norm_is_not_adapted():
    model, _ = tiny()
    ids = {id(p) for p in norm_parameters(model)}
    assert id(model.model.backbone.layernorm.weight) not in ids
    n_norms = sum("LayerNorm" in type(m).__name__ for m in model.model.backbone.modules())
    assert len(ids) == 2 * (n_norms - 1)


def test_stochastic_depth_is_the_only_noise_and_only_in_train_mode():
    model, data = tiny()
    x = data.tensors[0]
    run_logits = lambda: model(input_values=x).logits.detach()
    model.eval(); plain = run_logits()
    assert add_stochastic_depth(model, 0.5) == N_BLOCKS
    model.eval()
    assert torch.equal(run_logits(), plain), "eval-mode predictions (teacher, scoring) must have no noise"
    model.train()
    assert not torch.equal(run_logits(), run_logits()), "the student must be noisy"
    add_stochastic_depth(model, 0.0)
    assert torch.equal(run_logits(), plain)


def test_predictions_are_ordered_by_clip_index():
    model, data = tiny()
    x = data.tensors[0]
    order = torch.randperm(len(x))
    loader = torch.utils.data.DataLoader(TensorDataset(x[order], order), batch_size=3)
    probs, feats = predict(model, loader, "cpu")
    with torch.no_grad():
        out = model.eval()(input_values=x)
    assert torch.allclose(probs, torch.sigmoid(out.logits), atol=1e-6)
    assert torch.allclose(feats, out.last_hidden_state.mean(dim=(-2, -1)), atol=1e-6)


def test_each_batch_is_matched_with_its_own_clips_targets():
    """Dropout Student without noise and lr = 0: the student equals the teacher, so every batch's loss is the
    entropy of its own clips' predictions. Targets handed to the wrong clips would give a larger loss."""
    model, data = tiny()
    with torch.no_grad():
        model.head.weight *= 20                                        # spread the predictions apart
        probs = torch.sigmoid(model.eval()(input_values=data.tensors[0]).logits)
    want = float(-(probs * probs.log() + (1 - probs) * (1 - probs).log()).mean())
    history = run(model, data, "dropout_student", lr=0.0, dropout=False)
    assert all(math.isclose(r["loss"], want, rel_tol=1e-4) for r in history), (history, want)


def test_a_non_finite_gradient_stops_before_the_step():
    model, data = tiny()
    before = copy.deepcopy(model.state_dict())
    model.model.backbone.embeddings.layernorm.weight.register_hook(lambda g: g * float("nan"))
    try:
        run(model, data, "tent")
    except (FloatingPointError, RuntimeError) as e:
        assert "non-finite gradient" in str(e), e
    else:
        raise AssertionError("a NaN gradient must stop adaptation")
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_adaptation_data_never_reads_labels():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.parquet"
        pd.DataFrame({"audio_path": ["a.ogg"], "hsn_classes": [["daejun"]]}).to_parquet(path)
        seen = []
        original = pd.read_parquet
        pd.read_parquet = lambda *a, **k: seen.append(k.get("columns")) or original(*a, **k)
        try:
            ds = UnlabeledSoundscapes(path)
        finally:
            pd.read_parquet = original
        assert seen == [["audio_path"]] and ds.paths == ["a.ogg"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
